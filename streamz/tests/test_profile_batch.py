import asyncio
from itertools import combinations
from time import sleep

import pytest

from streamz import Stream
from streamz.profile_batch import (
    MISSING,
    BatchResult,
    Pricing,
    ProfileRequest,
    plan_calls,
    plan_requests,
)


PRICES = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}


def make_pricing(fixed_fee=1.0, max_ids=None, max_fields=None):
    return Pricing(PRICES, fixed_fee=fixed_fee, max_ids=max_ids,
                   max_fields=max_fields)


def req(request_id, ids, fields, context=None):
    return ProfileRequest.create(
        request_id=request_id, ids=ids, fields=fields, context=context
    )


def assert_plan_respects_limits(plan, pricing):
    assert plan.calls
    for call in plan.calls:
        assert len(call.ids) >= 1
        assert len(call.fields) >= 1
        if pricing.max_ids is not None:
            assert len(call.ids) <= pricing.max_ids
        if pricing.max_fields is not None:
            assert len(call.fields) <= pricing.max_fields


def assert_plan_covers(plan):
    covered = set()
    for call in plan.calls:
        covered |= {(i, f) for i in call.ids for f in call.fields}
    assert set(plan.coverage) <= covered


# ---------------------------------------------------------------------------
# Planner: pricing, chunking and merging
# ---------------------------------------------------------------------------


def test_call_cost_formula():
    pricing = make_pricing()
    # 2 IDs * (0.1 + 0.2) + fixed fee 1.0
    assert pricing.call_cost(2, ("a", "b")) == pytest.approx(1.6)


def test_plan_chunks_ids_by_max_ids():
    pricing = make_pricing(max_ids=2)
    plan = plan_requests([req("r", [1, 2, 3, 4, 5], ["a"])], pricing)
    assert_plan_respects_limits(plan, pricing)
    assert_plan_covers(plan)
    # exact DP returns chunks in backtrack order; compare as multisets
    assert sorted(len(c.ids) for c in plan.calls) == [1, 2, 2]
    assert all(c.fields == ("a",) for c in plan.calls)
    assert {i for c in plan.calls for i in c.ids} == {1, 2, 3, 4, 5}
    # three calls: 3 fixed fees + 5 * 0.1 variable
    assert plan.total_cost == pytest.approx(3.5)


def test_plan_single_field_single_call_without_limits():
    pricing = make_pricing()
    plan = plan_requests(
        [req("r1", [1, 2], ["a"]), req("r2", [2, 3], ["a"])], pricing
    )
    assert len(plan.calls) == 1
    call = plan.calls[0]
    assert set(call.ids) == {1, 2, 3}
    assert call.fields == ("a",)
    # merging saves fixed fees and never over-fetches a field
    assert call.cost == pytest.approx(1.3)
    assert plan.total_cost == pytest.approx(1.3)


def test_plan_field_merge_when_fixed_fee_dominates():
    # fixed fee large -> sharing one call for both fields is cheaper
    pricing = make_pricing(fixed_fee=10.0)
    plan = plan_requests(
        [req("r1", [1], ["a"]), req("r2", [2], ["b"])], pricing
    )
    assert len(plan.calls) == 1
    call = plan.calls[0]
    assert set(call.ids) == {1, 2}
    assert set(call.fields) == {"a", "b"}
    # 10 + 2 * 0.3 = 10.6; two separate calls would cost 20.4
    assert plan.total_cost == pytest.approx(10.6)


def test_plan_keeps_fields_separate_when_variable_dominates():
    # fixed fee small, IDs differ -> merging over-fetches and costs more
    pricing = make_pricing(fixed_fee=0.01)
    plan = plan_requests(
        [req("r1", [1], ["a"]), req("r2", [2], ["b"])], pricing
    )
    assert len(plan.calls) == 2
    assert all(len(c.fields) == 1 for c in plan.calls)
    # two calls: fixed fees 0.02 + variable 0.1 + 0.2 = 0.32
    assert plan.total_cost == pytest.approx(0.32)


def test_plan_merges_overlapping_ids_when_cheaper():
    # same IDs, different fields, no fixed fee: merging costs nothing extra
    pricing = make_pricing(fixed_fee=0.0)
    plan = plan_requests(
        [req("r1", [1, 2], ["a"]), req("r2", [1, 2], ["b"])], pricing
    )
    assert len(plan.calls) == 1
    assert set(plan.calls[0].fields) == {"a", "b"}
    assert plan.total_cost == pytest.approx(2 * (0.1 + 0.2))


def test_plan_respects_max_fields():
    pricing = make_pricing(fixed_fee=10.0, max_fields=2)
    plan = plan_requests([req("r", [1], ["a", "b", "c"])], pricing)
    assert_plan_respects_limits(plan, pricing)
    assert_plan_covers(plan)
    assert max(len(c.fields) for c in plan.calls) == 2


def test_plan_dedupes_ids_and_fields():
    pricing = make_pricing()
    plan = plan_requests(
        [req("r", [1, 1, 2, 2], ["a", "a", "b"])], pricing
    )
    call = plan.calls[0]
    assert set(call.ids) == {1, 2}
    assert set(call.fields) == {"a", "b"}


def test_plan_empty_requests():
    plan = plan_requests([], make_pricing())
    assert plan.calls == ()
    assert plan.total_cost == 0.0
    assert plan_calls([], make_pricing()) == ()


def test_plan_rejects_mixed_contexts():
    with pytest.raises(ValueError):
        plan_requests(
            [req("r1", [1], ["a"], context="cred1"),
             req("r2", [1], ["a"], context="cred2")],
            make_pricing(),
        )


def test_plan_rejects_duplicate_request_id():
    with pytest.raises(ValueError):
        plan_requests(
            [req("r", [1], ["a"]), req("r", [2], ["b"])], make_pricing()
        )


def test_plan_splits_fields_over_limit_and_rejects_unknown_price():
    # more fields than one call allows -> split across calls, never rejected
    pricing = make_pricing(max_fields=1)
    plan = plan_requests([req("r", [1], ["a", "b"])], pricing)
    assert_plan_respects_limits(plan, pricing)
    assert_plan_covers(plan)
    assert max(len(c.fields) for c in plan.calls) == 1
    with pytest.raises(ValueError):
        plan_calls({(1, "zzz")}, make_pricing())


def optimal_small_instance(pairs, pricing):
    """Independent true-optimum reference (brute-force weighted set cover).

    Kept deliberately separate from the production DP so the two
    implementations can only fail together by coincidence.
    """
    pairs = set(pairs)
    cells = tuple(sorted(pairs, key=repr))
    index = {pair: bit for bit, pair in enumerate(cells)}
    n = len(cells)
    all_ids = tuple(dict.fromkeys(i for i, _ in cells))
    all_fields = tuple(dict.fromkeys(f for _, f in cells))
    id_limit = pricing.max_ids or len(all_ids)
    field_limit = pricing.max_fields or len(all_fields)

    def subsets(seq, limit):
        from itertools import combinations
        for size in range(1, min(limit, len(seq)) + 1):
            for combo in combinations(seq, size):
                yield combo

    rectangles = {}
    for ids in subsets(all_ids, id_limit):
        for fields in subsets(all_fields, field_limit):
            mask = 0
            for i in ids:
                for f in fields:
                    if (i, f) in index:
                        mask |= 1 << index[(i, f)]
            if not mask:
                continue
            cost = pricing.call_cost(len(ids), fields)
            if mask not in rectangles or cost < rectangles[mask]:
                rectangles[mask] = cost

    full = (1 << n) - 1
    inf = float("inf")
    dp = [inf] * (full + 1)
    dp[0] = 0.0
    for mask, cost in rectangles.items():
        prev = dp.copy()  # 0-1: each rectangle at most once
        for state in range(full + 1):
            if prev[state] == inf:
                continue
            new_state = state | mask
            candidate = prev[state] + cost
            if candidate < dp[new_state]:
                dp[new_state] = candidate
    return dp[full]


def test_exact_solver_matches_independent_optimum():
    pricing = make_pricing(fixed_fee=2.0)
    pairs = {(1, "a"), (2, "a"), (3, "b"), (1, "b")}
    exact = plan_calls(pairs, pricing, solver="exact")
    exact_cost = sum(c.cost for c in exact)
    assert exact_cost == pytest.approx(optimal_small_instance(pairs, pricing))
    # feasibility of the produced calls
    covered = set()
    for call in exact:
        covered |= {(i, f) for i in call.ids for f in call.fields}
    assert pairs <= covered


def test_exact_solver_beats_greedy_under_binding_id_limit():
    # the case where field-partition / greedy plans leave money on the table
    pricing = make_pricing(fixed_fee=0.5, max_ids=2, max_fields=3)
    pairs = {
        (0, "a"), (0, "d"), (1, "b"), (1, "d"), (2, "a"), (2, "d"),
        (3, "c"), (3, "d"), (4, "a"), (4, "b"), (4, "d"),
    }
    exact = plan_calls(pairs, pricing, solver="exact")
    greedy = plan_calls(pairs, pricing, solver="greedy")
    exact_cost = sum(c.cost for c in exact)
    greedy_cost = sum(c.cost for c in greedy)
    assert exact_cost == pytest.approx(optimal_small_instance(pairs, pricing))
    assert exact_cost < greedy_cost
    # exact still respects every per-call limit
    for call in exact:
        assert len(call.ids) <= 2
        assert len(call.fields) <= 3


def test_auto_solver_uses_exact_for_small_batches():
    pricing = make_pricing(fixed_fee=0.5, max_ids=2, max_fields=3)
    pairs = {
        (0, "a"), (0, "d"), (1, "b"), (1, "d"), (2, "a"), (2, "d"),
        (3, "c"), (3, "d"), (4, "a"), (4, "b"), (4, "d"),
    }
    auto = plan_calls(pairs, pricing)  # 11 cells <= default 12
    assert sum(c.cost for c in auto) == pytest.approx(
        optimal_small_instance(pairs, pricing)
    )


def test_auto_solver_falls_back_to_greedy_for_large_batches():
    pricing = make_pricing()
    pairs = {(i, f) for i in range(7) for f in ("a", "b")}  # 14 cells
    # exact refuses at this size ...
    with pytest.raises(ValueError):
        plan_calls(pairs, pricing, solver="exact", exact_max_cells=12)
    # ... but auto transparently falls back and stays feasible
    calls = plan_calls(pairs, pricing, exact_max_cells=12)
    covered = set()
    for call in calls:
        covered |= {(i, f) for i in call.ids for f in call.fields}
    assert pairs <= covered


def test_greedy_feasible_and_bounded_against_true_optimum():
    import random
    rng = random.Random(42)
    fields = ["a", "b", "c", "d"]
    worst_ratio = 1.0
    for fixed_fee in (0.0, 0.5, 5.0):
        for _ in range(25):
            pricing = make_pricing(
                fixed_fee=fixed_fee,
                max_ids=rng.choice([None, 2, 3]),
                max_fields=rng.choice([None, 2, 3]),
            )
            # keep instances small enough for the exact set-cover reference
            pairs = set()
            n_ids = rng.randint(1, 4)
            for i in range(n_ids):
                for f in rng.sample(fields, rng.randint(1, 3)):
                    pairs.add((i, f))
            optimum = optimal_small_instance(pairs, pricing)
            greedy = plan_calls(pairs, pricing, solver="greedy")
            greedy_cost = sum(c.cost for c in greedy)

            # the heuristic plan must always be feasible ...
            covered = set()
            for call in greedy:
                if pricing.max_ids is not None:
                    assert len(call.ids) <= pricing.max_ids
                if pricing.max_fields is not None:
                    assert len(call.fields) <= pricing.max_fields
                covered |= {(i, f) for i in call.ids for f in call.fields}
            assert pairs <= covered

            # ... never cheaper than the true optimum ...
            assert greedy_cost >= optimum - 1e-12
            # ... and stay within a modest approximation gap on these sizes
            ratio = greedy_cost / optimum if optimum else 1.0
            worst_ratio = max(worst_ratio, ratio)
            assert ratio < 1.5
    # the heuristic is a fallback only; document its observed worst case here
    assert worst_ratio < 1.45


def test_exact_solver_refuses_oversized_pair_sets():
    pricing = make_pricing()
    pairs = {(i, f) for i in range(7) for f in ("a", "b")}  # 14 cells
    assert len(pairs) == 14
    with pytest.raises(ValueError):
        plan_calls(pairs, pricing, solver="exact", exact_max_cells=12)


def test_invalid_solver_name_rejected():
    with pytest.raises(ValueError):
        plan_calls({(1, "a")}, make_pricing(), solver="bogus")


def test_coverage_maps_pairs_to_requests():
    plan = plan_requests(
        [req("r1", [1, 2], ["a"]), req("r2", [2, 3], ["a", "b"])],
        make_pricing(),
    )
    assert set(plan.coverage[(2, "a")]) == {"r1", "r2"}
    assert plan.coverage[(3, "b")] == ("r2",)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


def make_fetcher(fail_pairs=(), exc_pairs=(), fail_once=()):
    """In-memory fetcher recording every call.

    fail_pairs: pairs permanently omitted from responses
    exc_pairs:  whole call raises while any of these pairs are requested
    fail_once:  pairs omitted the first time they are requested, then returned
    """
    calls = []
    seen_once = set()

    def fetcher(ids, fields, context=None):
        calls.append((tuple(ids), tuple(fields), context))
        exc = {(i, f) for i in ids for f in fields} & set(exc_pairs)
        if exc:
            raise RuntimeError("boom for %r" % (exc,))
        data = {}
        for i in ids:
            row = {}
            for f in fields:
                pair = (i, f)
                if pair in fail_pairs:
                    continue
                if pair in fail_once and pair not in seen_once:
                    seen_once.add(pair)
                    continue
                row[f] = "%s:%s:%s" % (context, i, f)
            data[i] = row
        return data

    fetcher.calls = calls
    return fetcher


def emit_req(source, request_id, ids, fields, context=None, **extra):
    event = {"request_id": request_id, "ids": ids, "fields": fields,
             "context": context}
    event.update(extra)
    source.emit(event)


def test_operator_flushes_on_max_requests_and_reconstructs():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, fixed_fee=1.0, max_requests=2
    ).sink_to_list()

    emit_req(source, "r1", [1, 2], ["a"])
    assert out == []  # batch not complete yet
    emit_req(source, "r2", [2, 3], ["b"])

    assert len(out) == 1
    result = out[0]
    assert isinstance(result, BatchResult)
    assert result.is_complete()
    assert result.get("r1") == {
        1: {"a": "None:1:a"},
        2: {"a": "None:2:a"},
    }
    assert result.get("r2")[3] == {"b": "None:3:b"}
    # the shared ID 2 is fetched once and reused for both requirements
    assert result.charged_cost == result.planned_cost


def test_operator_flushes_on_expected_count():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, expected_count_key="n"
    ).sink_to_list()

    emit_req(source, "r1", [1], ["a"], n=3)
    emit_req(source, "r2", [2], ["a"], n=3)
    assert out == []
    emit_req(source, "r3", [3], ["a"], n=3)
    assert len(out) == 1
    assert {r.request_id for r in out[0].requests} == {"r1", "r2", "r3"}


def test_operator_timeout_flush():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, timeout=0.05
    ).sink_to_list()

    emit_req(source, "r1", [1], ["a"])
    assert out == []
    sleep(0.15)
    assert len(out) == 1
    assert out[0].get("r1")[1]["a"] == "None:1:a"


def test_operator_manual_flush():
    fetcher = make_fetcher()
    source = Stream()
    batcher = source.profile_batch(fetcher, PRICES)
    out = batcher.sink_to_list()

    emit_req(source, "r1", [1], ["a"])
    batcher.flush()
    assert len(out) == 1


def test_operator_contexts_are_never_mixed():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, max_requests=1
    ).sink_to_list()

    emit_req(source, "r1", [1], ["a"], context="cred1")
    emit_req(source, "r2", [2], ["a"], context="cred2")

    contexts = {result.context for result in out}
    assert contexts == {"cred1", "cred2"}
    for result in out:
        for call in result.calls_made:
            assert call.context == result.context
        for (i, f), value in result.values.items():
            assert value.startswith(str(result.context) + ":")
        assert all(req.context == result.context for req in result.requests)


def test_operator_partial_failure_only_refetches_missing_pairs():
    fetcher = make_fetcher(fail_once={(2, "a")})
    plans = []
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, fixed_fee=1.0, retries=1, max_requests=1,
        plan_callback=plans.append,
    ).sink_to_list()

    emit_req(source, "r1", [1, 2], ["a", "b"])

    result = out[0]
    assert result.is_complete()
    # initial call had both fields for both IDs; the retry call only covers
    # the one missing pair
    assert len(fetcher.calls) == 2
    retry_ids, retry_fields, _ = fetcher.calls[1]
    assert retry_ids == (2,)
    assert retry_fields == ("a",)
    # values that succeeded on round one are kept, not refetched
    assert result.values[(1, "a")] == "None:1:a"
    # the initial plan callback fires before execution; the retry round is
    # internal, so exactly one plan was announced
    assert len(plans) == 1
    assert result.charged_cost > result.planned_cost


def test_operator_permanent_failure_marks_missing_and_keeps_rest():
    fetcher = make_fetcher(fail_pairs={(2, "a")})
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, retries=1, max_requests=1
    ).sink_to_list()

    emit_req(source, "r1", [1, 2], ["a", "b"])
    result = out[0]
    assert not result.is_complete()
    assert result.is_complete("r1") is False
    assert (2, "a") in result.failures
    assert result.missing_pairs("r1") == {(2, "a")}
    reconstructed = result.get("r1")
    assert reconstructed[2]["a"] is MISSING
    assert reconstructed[2]["b"] == "None:2:b"
    assert reconstructed[1]["a"] == "None:1:a"


def test_operator_call_exception_retries_then_reports_failure():
    fetcher = make_fetcher(exc_pairs={(1, "a")})
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, retries=1, max_requests=1
    ).sink_to_list()

    emit_req(source, "r1", [1], ["a"])
    result = out[0]
    # one attempt + one retry, both failed
    assert len(fetcher.calls) == 2
    assert (1, "a") in result.failures
    assert result.get("r1")[1]["a"] is MISSING


def test_operator_keeps_falsy_values():
    # 0 / False / "" are valid profile values, not "missing"
    def fetcher(ids, fields, context=None):
        return {1: {"a": 0, "b": False, "c": ""}}

    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, max_requests=1
    ).sink_to_list()
    emit_req(source, "r1", [1], ["a", "b", "c"])
    row = out[0].get("r1")[1]
    assert row == {"a": 0, "b": False, "c": ""}
    assert out[0].is_complete()


def test_operator_manual_flush_keeps_subsequent_batches_working():
    fetcher = make_fetcher()
    source = Stream()
    batcher = source.profile_batch(fetcher, PRICES, timeout=0.05)
    out = batcher.sink_to_list()

    emit_req(source, "r1", [1], ["a"])
    batcher.flush()
    assert len(out) == 1

    # the timeout scheduled for r1 must not resurrect an empty flush
    sleep(0.1)
    assert len(out) == 1

    # and the next batch still flushes normally
    emit_req(source, "r2", [2], ["a"])
    batcher.flush()
    assert len(out) == 2
    assert out[1].get("r2")[2]["a"] == "None:2:a"


def test_operator_custom_extractors():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES,
        request_key="req", id_key="entities", fields_key="attrs",
        context_key="cred", max_requests=1,
    ).sink_to_list()

    source.emit({"req": "x", "entities": [7], "attrs": ["a"],
                 "cred": "tenant"})
    result = out[0]
    assert result.context == "tenant"
    assert result.get("x")[7]["a"] == "tenant:7:a"


def test_operator_callable_extractors():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES,
        request_key=lambda e: e["id_"],
        id_key=lambda e: e["entity_ids"],
        fields_key=lambda e: e["field_names"],
        max_requests=1,
    ).sink_to_list()

    source.emit({"id_": "q", "entity_ids": [9], "field_names": ["b"]})
    assert out[0].get("q")[9]["b"] == "None:9:b"


def test_operator_enforces_limits_on_wire_calls():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, fixed_fee=0.01, max_ids=2, max_fields=2,
        max_requests=1,
    ).sink_to_list()

    emit_req(source, "r1", [1, 2, 3, 4, 5], ["a", "b", "c"])
    for ids, fields, _ in fetcher.calls:
        assert len(ids) <= 2
        assert len(fields) <= 2
    assert out[0].is_complete()


def test_operator_async_fetcher():
    calls = []

    async def fetcher(ids, fields, context=None):
        await asyncio.sleep(0)
        calls.append((tuple(ids), tuple(fields)))
        return {i: {f: (i, f) for f in fields} for i in ids}

    source = Stream(asynchronous=False)
    out = source.profile_batch(
        fetcher, PRICES, max_requests=1
    ).sink_to_list()
    emit_req(source, "r1", [1, 2], ["a"])
    result = out[0]
    assert result.get("r1") == {
        1: {"a": (1, "a")},
        2: {"a": (2, "a")},
    }
    assert calls


def test_operator_bad_fetcher_return_raises():
    def fetcher(ids, fields, context=None):
        return ["not", "a", "mapping"]

    source = Stream()
    source.profile_batch(fetcher, PRICES, max_requests=1).sink_to_list()
    with pytest.raises(TypeError):
        emit_req(source, "r1", [1], ["a"])


def test_operator_rejects_empty_requirement():
    source = Stream()
    source.profile_batch(make_fetcher(), PRICES, max_requests=1).sink_to_list()
    with pytest.raises(ValueError):
        emit_req(source, "r1", [], ["a"])
    with pytest.raises(ValueError):
        emit_req(source, "r2", [1], [])


def test_operator_releases_metadata_refs_when_all_calls_fail():
    from streamz import RefCounter

    def fetcher(ids, fields, context=None):
        raise RuntimeError("api down")

    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, retries=0, max_requests=1
    ).sink_to_list()
    ref = RefCounter()
    source.emit(
        {"request_id": "r1", "ids": [1], "fields": ["a"]},
        metadata=[{"ref": ref}],
    )
    # failed pairs are reported, not raised ...
    result = out[0]
    assert not result.is_complete()
    assert (1, "a") in result.failures
    # ... and every retained metadata reference has been released
    assert ref.count == 0


def test_operator_buffer_is_empty_after_completed_flush():
    source = Stream()
    batcher = source.profile_batch(
        make_fetcher(), PRICES, max_requests=1
    )
    out = batcher.sink_to_list()
    emit_req(source, "r1", [1], ["a"])
    assert len(out) == 1
    assert batcher._buffer == {}
    assert batcher._metadata == {}


def test_operator_plan_callback_receives_plan_with_cost():
    fetcher = make_fetcher()
    plans = []
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, fixed_fee=1.0, max_requests=1,
        plan_callback=plans.append,
    ).sink_to_list()

    emit_req(source, "r1", [1, 2], ["a"])
    assert len(plans) == 1
    plan = plans[0]
    assert plan.total_cost > 0
    assert all(c.cost > 0 for c in plan.calls)
    assert out[0].planned_cost == plan.total_cost


def test_operator_dedupes_repeated_pairs():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, max_requests=2
    ).sink_to_list()

    emit_req(source, "r1", [1, 1], ["a", "a"])
    emit_req(source, "r2", [1], ["a"])
    result = out[0]
    # only one (id, field) pair on the wire
    assert len(fetcher.calls) == 1
    assert fetcher.calls[0][0] == (1,)
    assert fetcher.calls[0][1] == ("a",)
    assert result.get("r1") == {1: {"a": "None:1:a"}}
    assert result.get("r2") == {1: {"a": "None:1:a"}}


def test_solver_exact_end_to_end():
    fetcher = make_fetcher()
    source = Stream()
    out = source.profile_batch(
        fetcher, PRICES, fixed_fee=2.0, max_requests=1,
        solver="exact", exact_max_cells=12,
    ).sink_to_list()

    emit_req(source, "r1", [1, 2], ["a", "b", "c"])
    result = out[0]
    assert result.is_complete()
    assert max(len(c.fields) for c in result.calls_made) <= 4
