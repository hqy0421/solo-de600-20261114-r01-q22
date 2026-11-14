"""
Batched third-party profile (portrait) collection.

This module solves a very specific acquisition problem:

* A third-party profile API can only query **one set of IDs** for **one set of
  fields** per call.
* Each call is limited in the number of IDs and the number of fields.
* A call costs a fixed fee plus, for every ID in the call, the sum of the unit
  prices of every requested field.  Over-fetching (an ID/field combination that
  nobody asked for) is billed just the same.
* Different requirements ask for different fields.

The :class:`ProfileBatcher` stream operator buffers incoming requirements until
a batch is complete, asks :func:`plan_calls` for the cheapest feasible set of
API calls covering every required ``(id, field)`` pair, executes the plan, and
reconstructs each original requirement from the returned values.  Partial
failures only trigger re-fetches of the still-missing pairs; values that were
already fetched successfully are kept and never paid for twice.

Requirements carrying different *contexts* (different credentials or snapshot
scopes) are always buffered, planned and executed separately; data fetched under
one context is never mixed into a result of another context.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Callable, Hashable, Mapping, Optional

from tornado import gen

from .core import Stream, sync

logger = logging.getLogger(__name__)

__all__ = [
    "MISSING",
    "ProfileRequest",
    "CallSpec",
    "BatchPlan",
    "BatchResult",
    "Pricing",
    "plan_calls",
    "plan_requests",
    "ProfileBatcher",
]


class _Missing:
    """Sentinel placed in a reconstructed result for pairs that could not be
    fetched after all retries."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<MISSING>"

    def __bool__(self):
        return False


MISSING = _Missing()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileRequest:
    """One collection requirement.

    Parameters
    ----------
    request_id: hashable
        Identifier used to reconstruct this requirement later.
    ids: sequence of hashable
        The IDs this requirement needs values for.  Duplicates are ignored.
    fields: sequence of hashable
        The profile fields this requirement needs.  Duplicates are ignored.
    context: hashable, optional
        Credential / snapshot scope.  Requirements with different contexts are
        never combined into the same API call.
    """

    request_id: Hashable
    ids: tuple
    fields: tuple
    context: Hashable = None

    @classmethod
    def create(cls, request_id, ids, fields, context=None):
        ids = tuple(dict.fromkeys(ids))
        fields = tuple(dict.fromkeys(fields))
        if not ids or not fields:
            raise ValueError(
                "ProfileRequest %r requires at least one id and one field"
                % (request_id,)
            )
        return cls(
            request_id=request_id,
            ids=ids,
            fields=fields,
            context=context,
        )

    @property
    def pairs(self):
        return {(i, f) for i in self.ids for f in self.fields}


@dataclass(frozen=True)
class CallSpec:
    """One concrete API call: query ``ids`` for ``fields`` under ``context``."""

    ids: tuple
    fields: tuple
    cost: float
    context: Hashable = None

    def __post_init__(self):
        if not self.ids or not self.fields:
            raise ValueError("CallSpec requires at least one id and one field")


@dataclass(frozen=True)
class Pricing:
    """Pricing and per-call limits of the profile API.

    Parameters
    ----------
    field_prices: mapping
        Unit price per field.  A call with ``n`` IDs and fields ``F`` costs
        ``fixed_fee + n * sum(field_prices[f] for f in F)``.
    fixed_fee: float
        Fixed fee charged for every call.
    max_ids: int, optional
        Maximum number of IDs allowed in one call.
    max_fields: int, optional
        Maximum number of fields allowed in one call.
    """

    field_prices: Mapping
    fixed_fee: float = 0.0
    max_ids: Optional[int] = None
    max_fields: Optional[int] = None

    def field_price(self, name):
        try:
            return self.field_prices[name]
        except KeyError:
            raise ValueError(
                "No unit price configured for field %r" % (name,)
            ) from None

    def call_cost(self, n_ids, fields):
        """Cost of a single (un-chunked) call with ``n_ids`` IDs."""
        return self.fixed_fee + n_ids * sum(
            self.field_price(f) for f in fields
        )


@dataclass(frozen=True)
class BatchPlan:
    """The cheapest feasible plan found for one batch of requirements."""

    context: Hashable
    calls: tuple
    total_cost: float
    requests: tuple
    coverage: Mapping = field(default_factory=dict)

    def __iter__(self):
        return iter(self.calls)

    def __len__(self):
        return len(self.calls)


@dataclass
class BatchResult:
    """Result emitted once a whole batch has been fetched.

    Successfully fetched values live in :attr:`values` keyed by
    ``(id, field)``; pairs that failed on every attempt are in
    :attr:`failures`.  Use :meth:`get` to reconstruct one original requirement.
    """

    context: Hashable
    requests: tuple
    plan: BatchPlan
    values: dict
    failures: dict
    calls_made: list
    planned_cost: float
    charged_cost: float

    def _request(self, request_id):
        for req in self.requests:
            if req.request_id == request_id:
                return req
        raise KeyError("Unknown request_id %r" % (request_id,))

    def get(self, request_id):
        """Reconstruct one requirement.

        Returns a nested mapping ``{id: {field: value}}``.  Pairs that could
        not be fetched are filled with the :data:`MISSING` sentinel.
        """
        req = self._request(request_id)
        return {
            i: {
                f: self.values.get((i, f), MISSING)
                for f in req.fields
            }
            for i in req.ids
        }

    def missing_pairs(self, request_id=None):
        """The ``(id, field)`` pairs that were never fetched successfully."""
        if request_id is None:
            return set(self.failures)
        req = self._request(request_id)
        return req.pairs & set(self.failures)

    def is_complete(self, request_id=None):
        """Whether all (or one requirement's) pairs were fetched."""
        return not self.missing_pairs(request_id)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _chunk(items, size):
    """Split an ordered sequence into chunks of at most ``size``."""
    if size is None:
        return [tuple(items)]
    return [tuple(items[i:i + size]) for i in range(0, len(items), size)]


def _calls_for_field_blocks(blocks, block_ids, pricing, context):
    """Materialize chunked calls for a field partition.

    ``block_ids`` maps each block to the ordered IDs needing one of its fields.
    """
    calls = []
    for block in blocks:
        fields = tuple(sorted(block, key=repr))
        for ids in _chunk(block_ids[block], pricing.max_ids):
            calls.append(
                CallSpec(
                    ids=ids,
                    fields=fields,
                    cost=pricing.call_cost(len(ids), fields),
                    context=context,
                )
            )
    return calls


def _block_cost(block, ids, pricing):
    """Total cost of serving one field block for ``ids`` (incl. chunking)."""
    fields = tuple(block)
    return sum(
        pricing.call_cost(len(chunk), fields)
        for chunk in _chunk(ids, pricing.max_ids)
    )


def _greedy_field_partition(pairs, pricing, context):
    """Candidate plan: start with one field per block, greedily merge blocks
    while merging strictly lowers the total cost."""
    all_fields = tuple(dict.fromkeys(f for _, f in pairs))
    field_ids = {f: tuple(dict.fromkeys(i for i, ff in pairs if ff == f))
                 for f in all_fields}
    blocks = [frozenset((f,)) for f in all_fields]
    block_ids = {frozenset((f,)): list(ids) for f, ids in field_ids.items()}

    def cost_of(blocks):
        return sum(
            _block_cost(b, block_ids[b], pricing) for b in blocks
        )

    total = cost_of(blocks)
    merged = True
    while merged and len(blocks) > 1:
        merged = False
        best = None  # (new_total, idx_a, idx_b, merged_block, merged_ids)
        for ai in range(len(blocks)):
            for bi in range(ai + 1, len(blocks)):
                new_block = blocks[ai] | blocks[bi]
                if (pricing.max_fields is not None
                        and len(new_block) > pricing.max_fields):
                    continue
                new_ids = tuple(dict.fromkeys(
                    list(block_ids[blocks[ai]]) + list(block_ids[blocks[bi]])
                ))
                new_total = (
                    total
                    - _block_cost(blocks[ai], block_ids[blocks[ai]], pricing)
                    - _block_cost(blocks[bi], block_ids[blocks[bi]], pricing)
                    + _block_cost(new_block, new_ids, pricing)
                )
                if new_total < total and (best is None or new_total < best[0]):
                    best = (new_total, ai, bi, new_block, new_ids)
        if best is not None:
            new_total, ai, bi, new_block, new_ids = best
            keep = [b for j, b in enumerate(blocks) if j not in (ai, bi)]
            blocks = keep + [new_block]
            block_ids[new_block] = list(new_ids)
            total = new_total
            merged = True
    return _calls_for_field_blocks(blocks, block_ids, pricing, context)


def _pattern_group_calls(requests, pricing, context):
    """Candidate plan: group requirements with identical field sets, so no
    field is ever over-fetched; only IDs are shared."""
    groups = {}
    order = []
    for req in requests:
        key = frozenset(req.fields)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(req)

    calls = []
    for key in order:
        fields = tuple(dict.fromkeys(
            f for req in groups[key] for f in req.fields
        ))
        ids = tuple(dict.fromkeys(
            i for req in groups[key] for i in req.ids
        ))
        # split both dimensions so every emitted call respects the API limits
        for field_chunk in _chunk(fields, pricing.max_fields):
            for id_chunk in _chunk(ids, pricing.max_ids):
                calls.append(
                    CallSpec(
                        ids=id_chunk,
                        fields=field_chunk,
                        cost=pricing.call_cost(len(id_chunk), field_chunk),
                        context=context,
                    )
                )
    return calls


def _merge_calls(calls, pricing):
    """Local search: repeatedly merge two calls into one while the merge is
    feasible (within both limits) and strictly lowers the total cost."""
    calls = list(calls)
    total = sum(c.cost for c in calls)
    merged = True
    while merged and len(calls) > 1:
        merged = False
        best = None
        for ai in range(len(calls)):
            for bi in range(ai + 1, len(calls)):
                a, b = calls[ai], calls[bi]
                ids = tuple(dict.fromkeys(a.ids + b.ids))
                fields = tuple(dict.fromkeys(a.fields + b.fields))
                if pricing.max_ids is not None and len(ids) > pricing.max_ids:
                    continue
                if (pricing.max_fields is not None
                        and len(fields) > pricing.max_fields):
                    continue
                new_cost = pricing.call_cost(len(ids), fields)
                new_total = total - a.cost - b.cost + new_cost
                if new_total < total and (best is None or new_total < best[0]):
                    best = (new_total, ai, bi, ids, fields, new_cost)
        if best is not None:
            new_total, ai, bi, ids, fields, new_cost = best
            merged_call = CallSpec(
                ids=ids, fields=fields, cost=new_cost,
                context=calls[ai].context,
            )
            calls = [c for j, c in enumerate(calls) if j not in (ai, bi)]
            calls.append(merged_call)
            total = new_total
            merged = True
    return calls


def _rectangles(pairs, pricing, context):
    """Every feasible call rectangle over the rows/columns of ``pairs``.

    A rectangle is an ``(ids, fields)`` combination within both per-call
    limits.  Rectangles with identical pair coverage are deduplicated keeping
    the cheapest one.  Yields ``(mask, cost, ids, fields)`` tuples where
    ``mask`` is a bitmask over the sorted pair list.
    """
    from itertools import combinations

    cells = tuple(sorted(pairs, key=repr))
    index = {pair: bit for bit, pair in enumerate(cells)}
    all_ids = tuple(dict.fromkeys(i for i, _ in cells))
    all_fields = tuple(dict.fromkeys(f for _, f in cells))
    id_limit = pricing.max_ids or len(all_ids)
    field_limit = pricing.max_fields or len(all_fields)

    def _subsets(seq, limit):
        for size in range(1, min(limit, len(seq)) + 1):
            for combo in combinations(seq, size):
                yield combo

    best = {}
    for ids in _subsets(all_ids, id_limit):
        for fields in _subsets(all_fields, field_limit):
            mask = 0
            for f in fields:
                for i in ids:
                    if (i, f) in index:
                        mask |= 1 << index[(i, f)]
            if not mask:
                continue
            cost = pricing.call_cost(len(ids), fields)
            if mask not in best or cost < best[mask][0]:
                best[mask] = (cost, tuple(ids), tuple(fields))
    for mask, (cost, ids, fields) in best.items():
        yield mask, cost, ids, fields


def _exact_set_cover_plan(pairs, pricing, context, max_cells):
    """Globally optimal plan via feasible-rectangle enumeration plus an exact
    weighted set-cover DP.  Only practical for small pair sets.

    Each rectangle may be chosen at most once (0-1 set cover).  Ties on total
    cost are broken towards fewer calls: at the same price a plan with fewer
    API calls has less overhead and fewer failure points.
    """
    pairs = set(pairs)
    n = len(pairs)
    if n > max_cells:
        raise ValueError(
            "exact solver is limited to %d required pairs (%d present); use "
            "solver='greedy'" % (max_cells, n)
        )
    full = (1 << n) - 1
    inf = float("inf")
    dp_cost = [inf] * (full + 1)
    dp_count = [0] * (full + 1)
    parent = [None] * (full + 1)
    dp_cost[0] = 0.0
    rectangles = list(_rectangles(pairs, pricing, context))
    for mask, cost, ids, fields in rectangles:
        # snapshot: every rectangle is either skipped or taken exactly once
        prev_cost = dp_cost.copy()
        prev_count = dp_count.copy()
        for state in range(full + 1):
            if prev_cost[state] == inf:
                continue
            new_state = state | mask
            candidate_cost = prev_cost[state] + cost
            candidate_count = prev_count[state] + 1
            incumbent = (dp_cost[new_state], dp_count[new_state])
            if (candidate_cost, candidate_count) < incumbent:
                dp_cost[new_state] = candidate_cost
                dp_count[new_state] = candidate_count
                parent[new_state] = (state, ids, fields)

    if dp_cost[full] == inf:  # pragma: no cover - singletons always feasible
        raise ValueError("No feasible call plan covers the required pairs")

    chosen = []
    state = full
    while state:
        prev_state, ids, fields = parent[state]
        chosen.append(CallSpec(
            ids=ids, fields=fields,
            cost=pricing.call_cost(len(ids), fields), context=context,
        ))
        state = prev_state
    return tuple(reversed(chosen))


def plan_calls(pairs, pricing, context=None, solver="auto",
               exact_max_cells=12):
    """Plan cheap calls covering a set of ``(id, field)`` pairs.

    Parameters
    ----------
    pairs: iterable of tuples
        The ``(id, field)`` combinations that must be returned.
    pricing: Pricing
        API prices and per-call limits.
    context: hashable, optional
        Attached to every produced :class:`CallSpec`.
    solver: {"auto", "exact", "greedy"}
        The underlying problem is a weighted rectangle-cover problem, which is
        NP-hard in general.

        * ``"exact"`` enumerates every feasible call rectangle and solves the
          resulting weighted set-cover problem with a bitmask DP, returning the
          globally minimum-cost plan.  It raises ``ValueError`` when the number
          of required pairs exceeds ``exact_max_cells``.
        * ``"greedy"`` always runs in polynomial time.  It builds two
          over-fetch-free candidate plans (identical-field-set grouping and
          greedy field-block merging), improves both with pairwise call
          merging, and keeps the cheapest.
        * ``"auto"`` (default) uses the exact solver up to
          ``exact_max_cells`` pairs and falls back to the greedy heuristic for
          larger batches, so ordinary batches are billed at the true minimum.
    exact_max_cells: int
        Maximum number of required ``(id, field)`` pairs handled by the exact
        solver.  The DP state space grows as ``2 ** n_cells``.

    Returns
    -------
    tuple of CallSpec
    """
    pairs = set(pairs)
    if not pairs:
        return ()
    all_fields = frozenset(f for _, f in pairs)
    # validate prices up front
    for f in all_fields:
        pricing.field_price(f)

    if solver in ("exact", "auto"):
        if solver == "exact" or len(pairs) <= exact_max_cells:
            return _exact_set_cover_plan(pairs, pricing, context,
                                         exact_max_cells)
    elif solver != "greedy":
        raise ValueError(
            "solver must be 'auto', 'exact' or 'greedy', got %r" % (solver,))

    # Synthetic single-request view lets us reuse the pattern-group candidate.
    synthetic = [ProfileRequest.create(
        request_id="_pairs",
        ids=tuple(dict.fromkeys(i for i, _ in pairs)),
        fields=tuple(all_fields),
        context=context,
    )]
    candidate_a = _pattern_group_calls(synthetic, pricing, context)
    candidate_b = _greedy_field_partition(pairs, pricing, context)

    candidate_a = _merge_calls(candidate_a, pricing)
    candidate_b = _merge_calls(candidate_b, pricing)

    cost_a = sum(c.cost for c in candidate_a)
    cost_b = sum(c.cost for c in candidate_b)
    best = candidate_a if cost_a <= cost_b else candidate_b
    return tuple(best)


def plan_requests(requests, pricing, solver="auto", exact_max_cells=12):
    """Plan the cheapest calls covering a batch of :class:`ProfileRequest`.

    All requests must share the same ``context``; use one planner invocation
    per context (the stream operator does this automatically).

    Returns
    -------
    BatchPlan
    """
    requests = tuple(requests)
    if not requests:
        return BatchPlan(context=None, calls=(), total_cost=0.0,
                         requests=(), coverage={})

    contexts = {req.context for req in requests}
    if len(contexts) > 1:
        raise ValueError(
            "Cannot plan requirements from different contexts together: %r"
            % (sorted(contexts, key=repr),)
        )
    context = requests[0].context

    seen_ids = set()
    for req in requests:
        if req.request_id in seen_ids:
            raise ValueError(
                "Duplicate request_id %r within one batch" % (req.request_id,)
            )
        seen_ids.add(req.request_id)

    pairs = set()
    coverage = {}
    for req in requests:
        for pair in req.pairs:
            pairs.add(pair)
            coverage.setdefault(pair, []).append(req.request_id)
    coverage = {pair: tuple(rids) for pair, rids in coverage.items()}

    calls = plan_calls(pairs, pricing, context=context, solver=solver,
                       exact_max_cells=exact_max_cells)
    total = sum(c.cost for c in calls)
    return BatchPlan(
        context=context,
        calls=calls,
        total_cost=total,
        requests=requests,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Stream operator
# ---------------------------------------------------------------------------


def _as_getter(spec, default_key, missing=...):
    """Return a callable extracting a value from an event dict.

    ``spec`` may be a callable or a dict key (defaulting to ``default_key``).
    If ``missing`` is not ``...``, a missing key returns ``missing`` instead
    of raising (used for the optional context field).
    """
    if callable(spec):
        return spec
    key = default_key if spec is None else spec

    if missing is ...:
        def _getter(x):
            return x[key]
    else:
        def _getter(x):
            return x.get(key, missing)

    return _getter


@Stream.register_api(attribute_name="profile_batch")
class ProfileBatcher(Stream):
    """Batch profile requirements into the cheapest feasible API calls.

    Incoming stream elements are dictionaries describing one requirement each
    (equivalently, pass extractor callables):

    ``{"request_id": ..., "ids": [...], "fields": [...], "context": ...}``

    Requirements are buffered per ``context`` (credentials / snapshot scope).
    A batch flushes when the number of buffered requirements reaches
    ``max_requests``, when an event declares the expected batch size, or after
    ``timeout`` seconds of idling.  On flush the operator:

    1. Computes the minimum-cost set of API calls covering every required
       ``(id, field)`` pair and invokes ``plan_callback(plan)``.
    2. Executes the calls through ``fetcher``.
    3. On partial failure, re-plans and re-fetches only the still-missing
       pairs, up to ``retries`` extra rounds.  Already fetched values are
       retained.
    4. Emits one :class:`BatchResult` per batch; reconstruct any original
       requirement with ``result.get(request_id)``.

    Requirements from different contexts are never mixed, and values fetched
    under one context never enter another context's result.

    Parameters
    ----------
    fetcher: callable
        ``fetcher(ids, fields, context=...)`` performs one API call and
        returns a nested mapping ``{id: {field: value}}``.  Pairs that the API
        could not return are simply omitted; raising an exception marks the
        whole call as failed (eligible for retry).  Both sync callables and
        coroutine functions are supported.
    field_prices: mapping
        Unit price per field (see :class:`Pricing`).
    fixed_fee: float
        Fixed fee per API call.
    max_ids, max_fields: int, optional
        Per-call limits enforced by the API.
    request_key, id_key, fields_key, context_key: hashable or callable
        Event-dict keys (or extractor callables) for the requirement parts.
    max_requests: int, optional
        Flush a context's batch once this many requirements are buffered.
    expected_count_key: hashable, optional
        Event-dict key holding the expected number of requirements in this
        batch; the batch flushes once that many have arrived.
    timeout: float, optional
        Flush a context's batch this many seconds after its first requirement
        arrived, if nothing else triggered a flush earlier.
    retries: int
        Extra fetch rounds for missing/failed pairs (default 1).
    solver, exact_max_cells:
        Passed through to :func:`plan_requests`.  Defaults to ``"auto"``,
        which returns the globally minimum-cost plan for typical (small)
        batches and falls back to the greedy heuristic beyond
        ``exact_max_cells`` required pairs.
    plan_callback: callable, optional
        Called with the :class:`BatchPlan` before any API call is made, e.g.
        for logging the plan and its cost.

    Examples
    --------
    >>> def fetcher(ids, fields, context=None):
    ...     return {i: {f: f"{i}:{f}" for f in fields} for i in ids}
    >>> prices = {"age": 0.1, "city": 0.2}
    >>> source = Stream()
    >>> out = source.profile_batch(fetcher, prices, fixed_fee=1.0,
    ...                            max_ids=2, max_fields=2,
    ...                            max_requests=2).sink_to_list()
    >>> source.emit({"request_id": "r1", "ids": [1, 2], "fields": ["age"]})
    >>> source.emit({"request_id": "r2", "ids": [2, 3], "fields": ["city"]})
    >>> result = out[0]
    >>> result.get("r1")
    {1: {'age': '1:age'}, 2: {'age': '2:age'}}
    """

    _graphviz_shape = "diamond"

    def __init__(self, upstream, fetcher, field_prices, *, fixed_fee=0.0,
                 max_ids=None, max_fields=None, request_key="request_id",
                 id_key="ids", fields_key="fields", context_key="context",
                 max_requests=None, expected_count_key=None, timeout=None,
                 retries=1, solver="auto", exact_max_cells=12,
                 plan_callback=None, **kwargs):
        self.fetcher = fetcher
        self.pricing = Pricing(
            field_prices=field_prices,
            fixed_fee=fixed_fee,
            max_ids=max_ids,
            max_fields=max_fields,
        )
        self._get_request_id = _as_getter(request_key, "request_id")
        self._get_ids = _as_getter(id_key, "ids")
        self._get_fields = _as_getter(fields_key, "fields")
        self._get_context = _as_getter(context_key, "context", missing=None)
        self._get_expected = (
            _as_getter(expected_count_key, expected_count_key)
            if expected_count_key is not None else None
        )
        self.max_requests = max_requests
        self.timeout = timeout
        self.retries = retries
        self.solver = solver
        self.exact_max_cells = exact_max_cells
        self.plan_callback = plan_callback

        self._buffer = {}          # context -> list[ProfileRequest]
        self._metadata = {}        # context -> list[metadata]
        self._expected = {}        # context -> declared full size
        self._callbacks = {}       # context -> timeout handle
        self._flushing = set()     # contexts currently being flushed

        kwargs["ensure_io_loop"] = True
        Stream.__init__(self, upstream, **kwargs)

    # -- buffering ----------------------------------------------------------

    @gen.coroutine
    def update(self, x, who=None, metadata=None):
        context = self._get_context(x)
        # validate before retaining any metadata references
        request = ProfileRequest.create(
            request_id=self._get_request_id(x),
            ids=self._get_ids(x),
            fields=self._get_fields(x),
            context=context,
        )
        self._retain_refs(metadata)
        buf = self._buffer.setdefault(context, [])
        buf.append(request)
        meta_buf = self._metadata.setdefault(context, [])
        if isinstance(metadata, list):
            meta_buf.extend(metadata)
        else:
            meta_buf.append(metadata)

        if self._get_expected is not None and self._get_expected(x) is not None:
            self._expected[context] = self._get_expected(x)

        should_flush = False
        if context in self._expected and len(buf) >= self._expected[context]:
            should_flush = True
        elif self.max_requests is not None and len(buf) >= self.max_requests:
            should_flush = True

        if should_flush:
            self._cancel_timeout(context)
            try:
                yield self._flush(context)
            except Exception:
                # _flush releases metadata only on the happy path; make sure a
                # planning/execution failure cannot leak the retained refs
                self._release_refs(metadata)
                raise
            return
        if self.timeout is not None and len(buf) == 1:
            self._callbacks[context] = self.loop.call_later(
                self.timeout, self._flush, context
            )

    def flush(self):
        """Flush every buffered context immediately (manual trigger).

        Runs the flushes on this operator's own IOLoop and blocks until they
        finish, so it is safe to call from a plain synchronous thread even when
        no event loop is current there.
        """
        contexts = [context for context in list(self._buffer)
                    if self._buffer[context]]

        @gen.coroutine
        def _run():
            yield [self._flush(context) for context in contexts]

        sync(self.loop, _run)

    def _cancel_timeout(self, context):
        handle = self._callbacks.pop(context, None)
        if handle is not None:
            handle.cancel()

    # -- planning & execution ----------------------------------------------

    @gen.coroutine
    def _flush(self, context):
        if context in self._flushing:
            return
        if not self._buffer.get(context):
            return
        self._flushing.add(context)
        self._cancel_timeout(context)
        metadata_result = None
        released = False
        try:
            requests = tuple(self._buffer.pop(context))
            metadata_result = self._metadata.pop(context)
            self._expected.pop(context, None)

            plan = plan_requests(
                requests, self.pricing,
                solver=self.solver,
                exact_max_cells=self.exact_max_cells,
            )
            if self.plan_callback is not None:
                self.plan_callback(plan)

            result = yield self._execute(plan)
            emitted = self._emit(result, metadata=metadata_result)
            if emitted:
                yield emitted
            self._release_refs(metadata_result)
            released = True
        except Exception:
            logger.exception("profile_batch flush failed for context %r",
                             context)
            raise
        finally:
            # release here as well so the timeout-triggered path (which has no
            # caller-side cleanup) cannot leak retained metadata references
            if not released and metadata_result is not None:
                self._release_refs(metadata_result)
            self._flushing.discard(context)

    @gen.coroutine
    def _invoke_fetcher(self, call):
        outcome = self.fetcher(list(call.ids), list(call.fields),
                               context=call.context)
        if inspect.isawaitable(outcome):
            outcome = yield outcome
        return outcome

    @gen.coroutine
    def _execute(self, plan):
        needed = set(plan.coverage)
        values = {}
        failures = {}
        calls_made = []
        last_error = {}

        calls = list(plan.calls)
        for attempt in range(self.retries + 1):
            is_last = attempt == self.retries
            for call in calls:
                calls_made.append(call)
                try:
                    data = yield self._invoke_fetcher(call)
                except Exception as exc:
                    logger.warning(
                        "profile fetch call failed (attempt %d/%d): %s",
                        attempt + 1, self.retries + 1, exc,
                    )
                    for pair in ((i, f) for i in call.ids
                                 for f in call.fields):
                        if pair in needed:
                            last_error[pair] = exc
                    continue
                if not isinstance(data, Mapping):
                    raise TypeError(
                        "fetcher must return a mapping {id: {field: value}}, "
                        "got %r" % (type(data).__name__,)
                    )
                for i in call.ids:
                    if i not in data:
                        continue
                    row = data[i]
                    if not isinstance(row, Mapping):
                        raise TypeError(
                            "fetcher row for id %r must be a mapping, got %r"
                            % (i, type(row).__name__)
                        )
                    for f in call.fields:
                        pair = (i, f)
                        if pair in needed and f in row:
                            values[pair] = row[f]
                            needed.discard(pair)
                            last_error.pop(pair, None)
            if not needed:
                break
            if not is_last:
                # Re-plan only what is still missing.
                calls = list(plan_calls(
                    needed, self.pricing, context=plan.context,
                    solver=self.solver,
                    exact_max_cells=self.exact_max_cells,
                ))

        for pair in needed:
            err = last_error.get(pair)
            failures[pair] = (
                "missing after %d fetch round(s)" % (self.retries + 1,)
                if err is None else repr(err)
            )

        charged = sum(c.cost for c in calls_made)
        result = BatchResult(
            context=plan.context,
            requests=plan.requests,
            plan=plan,
            values=values,
            failures=failures,
            calls_made=calls_made,
            planned_cost=plan.total_cost,
            charged_cost=charged,
        )
        return result
