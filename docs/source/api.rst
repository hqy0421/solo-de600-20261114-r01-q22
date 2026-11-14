API
===

Stream
------

.. currentmodule:: streamz

.. autosummary::
   Stream

.. autosummary::
   Stream.connect
   Stream.destroy
   Stream.disconnect
   Stream.visualize
   accumulate
   buffer
   collect
   combine_latest
   delay
   filter
   flatten
   map
   map_async
   partition
   profile_batch
   rate_limit
   scatter
   sink
   sink_to_textfile
   slice
   sliding_window
   starmap
   timed_window
   union
   unique
   pluck
   zip
   zip_latest

.. automethod:: Stream.connect
.. automethod:: Stream.disconnect
.. automethod:: Stream.destroy
.. automethod:: Stream.emit
.. automethod:: Stream.frequencies
.. automethod:: Stream.register_api
.. automethod:: Stream.sink
.. automethod:: Stream.sink_to_list
.. automethod:: Stream.sink_to_textfile
.. automethod:: Stream.to_websocket
.. automethod:: Stream.to_mqtt
.. automethod:: Stream.update
.. automethod:: Stream.visualize

Sources
-------

.. autosummary::
   from_iterable
   filenames
   from_kafka
   from_kafka_batched
   from_mqtt
   from_process
   from_websocket
   from_textfile
   from_tcp
   from_http_server

DaskStream
----------

.. currentmodule:: streamz.dask

.. autosummary::
   DaskStream
   gather


Definitions
-----------

.. currentmodule:: streamz

.. autofunction:: accumulate
.. autofunction:: buffer
.. autofunction:: collect
.. autofunction:: combine_latest
.. autofunction:: delay
.. autofunction:: filter
.. autofunction:: flatten
.. autofunction:: map
.. autofunction:: partition
.. autofunction:: profile_batch
.. autofunction:: rate_limit
.. autofunction:: sink
.. autofunction:: sink_to_textfile
.. autofunction:: sliding_window
.. autofunction:: Stream
.. autofunction:: timed_window
.. autofunction:: union
.. autofunction:: unique
.. autofunction:: pluck
.. autofunction:: zip
.. autofunction:: zip_latest

.. autofunction:: from_iterable
.. autofunction:: filenames
.. autofunction:: from_kafka
.. autofunction:: from_kafka_batched
.. autofunction:: from_textfile

.. currentmodule:: streamz.dask

.. autofunction:: DaskStream
.. autofunction:: gather


Profile batching
----------------

.. currentmodule:: streamz

Planning and executing cost-optimal batches of third-party profile calls.
Requirements are grouped by credential/snapshot context, combined into the
cheapest feasible API calls, reconstructed per original requirement, and
re-fetched only for the pairs that failed.

.. autosummary::
   ProfileRequest
   CallSpec
   Pricing
   BatchPlan
   BatchResult
   plan_calls
   plan_requests
   MISSING

.. autofunction:: ProfileRequest
.. autoclass:: CallSpec
.. autoclass:: Pricing
.. autoclass:: BatchPlan
.. autoclass:: BatchResult
.. autofunction:: plan_calls
.. autofunction:: plan_requests
