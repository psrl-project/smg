"""Integration smoke test: PreemptionStatLogger → SubscribePreemptionEvents."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_servicer.vllm.preemption import PreemptionStatLogger
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
from vllm.outputs import STREAM_FINISHED


def _make_stats(req_ids):
    s = MagicMock()
    s.gateway_preemption_req_ids = req_ids
    return s


@pytest.mark.asyncio
async def test_stat_logger_feeds_subscribe_stream():
    """Events put by PreemptionStatLogger appear in SubscribePreemptionEvents."""
    preemption_queue: asyncio.Queue = asyncio.Queue()
    logger = PreemptionStatLogger(MagicMock(), 0, preemption_queue=preemption_queue)

    # Simulate two scheduler steps with preempted requests
    logger.record(_make_stats(["req-1"]), iteration_stats=None)
    logger.record(_make_stats(["req-2", "req-3"]), iteration_stats=None)

    # Build minimal servicer
    mock_llm = MagicMock()
    servicer = VllmEngineServicer(
        mock_llm, start_time=time.time(), preemption_queue=preemption_queue
    )
    mock_context = MagicMock()
    # SubscribePreemptionEvents uses `while not context.cancelled()`.
    # MagicMock().cancelled() returns a truthy MagicMock by default, which
    # would skip the loop body entirely. Return False so the loop body runs.
    mock_context.cancelled.return_value = False
    mock_request = MagicMock()

    # Collect one yielded event then stop
    gen = servicer.SubscribePreemptionEvents(mock_request, mock_context)
    event = await gen.__anext__()

    assert isinstance(event, vllm_engine_pb2.PreemptionEvent)
    assert set(event.request_ids) == {"req-1", "req-2", "req-3"}
    assert event.timestamp_ns > 0
    await gen.aclose()


@pytest.mark.asyncio
async def test_subscribe_terminates_on_cancellation():
    """SubscribePreemptionEvents terminates cleanly when context is cancelled."""
    preemption_queue: asyncio.Queue = asyncio.Queue()
    # Pre-load the queue so the generator yields one event before cancellation
    preemption_queue.put_nowait(["req-x"])

    mock_llm = MagicMock()
    servicer = VllmEngineServicer(
        mock_llm, start_time=time.time(), preemption_queue=preemption_queue
    )
    mock_context = MagicMock()
    mock_request = MagicMock()

    # First call: not cancelled → yields one event
    mock_context.cancelled.return_value = False
    gen = servicer.SubscribePreemptionEvents(mock_request, mock_context)
    event = await gen.__anext__()
    assert event.request_ids == ["req-x"]

    # Second call: cancelled → generator should stop
    mock_context.cancelled.return_value = True
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    await gen.aclose()


@pytest.mark.asyncio
async def test_generate_admission_close_waits_for_registered_requests():
    servicer = VllmEngineServicer(MagicMock(), start_time=time.time())
    assert servicer.begin_generate_admission()

    close_task = asyncio.create_task(servicer.close_generate_admission())
    await asyncio.sleep(0)
    assert not close_task.done()
    assert not servicer.begin_generate_admission()

    servicer.finish_generate_admission()
    await close_task

    await servicer.open_generate_admission()
    assert servicer.begin_generate_admission()
    servicer.finish_generate_admission()


def _make_park_servicer(default_request_seed=None):
    """Build a servicer whose engine.add_request is mocked, suitable for
    driving the Generate() park path with a `text` input (bypasses renderer)."""
    mock_llm = MagicMock()

    # add_request returns an output_collector that finishes immediately.
    async def _add_request(**kwargs):
        collector = MagicMock()
        collector.request_id = kwargs["request_id"]
        collector.get_nowait.return_value = STREAM_FINISHED
        collector.close = MagicMock()
        return collector

    mock_llm.add_request = MagicMock(side_effect=_add_request)
    servicer = VllmEngineServicer(
        mock_llm,
        start_time=time.time(),
        default_request_seed=default_request_seed,
    )
    # Stub the proto→params converters so we don't depend on real vLLM types.
    servicer._sampling_params_from_proto = MagicMock(
        return_value=MagicMock(logprobs=None, prompt_logprobs=None)
    )
    servicer._tokenization_kwargs_from_proto = MagicMock(return_value={})
    return servicer, mock_llm


def _make_text_request(request_id="req-park"):
    req = vllm_engine_pb2.GenerateRequest(request_id=request_id, text="hello", stream=False)
    return req


def _make_generate_context():
    ctx = MagicMock()

    async def _send_initial_metadata(_md):
        return None

    ctx.send_initial_metadata = MagicMock(side_effect=_send_initial_metadata)
    return ctx


@pytest.mark.parametrize(("rank", "expected_rank"), [(None, 0), (0, 0), (3, 3)])
@pytest.mark.asyncio
async def test_generate_forwards_data_parallel_rank_with_zero_default(rank, expected_rank):
    servicer, mock_llm = _make_park_servicer()
    request = _make_text_request()
    if rank is not None:
        request.data_parallel_rank = rank

    async for _ in servicer.Generate(request, _make_generate_context()):
        pass

    add_request_kwargs = mock_llm.add_request.call_args.kwargs
    assert add_request_kwargs["data_parallel_rank"] == expected_rank
    assert servicer.get_active_request_count(expected_rank) == 0


@pytest.mark.parametrize(("priority", "expected_priority"), [(None, None), (0, 0), (-7, -7)])
@pytest.mark.asyncio
async def test_generate_forwards_only_present_priority(priority, expected_priority):
    servicer, mock_llm = _make_park_servicer()
    request = _make_text_request()
    if priority is not None:
        request.priority = priority

    async for _ in servicer.Generate(request, _make_generate_context()):
        pass

    add_request_kwargs = mock_llm.add_request.call_args.kwargs
    if expected_priority is None:
        assert "priority" not in add_request_kwargs
    else:
        assert add_request_kwargs["priority"] == expected_priority


@pytest.mark.asyncio
async def test_generate_tracks_active_request_until_stream_finishes():
    servicer, mock_llm = _make_park_servicer()
    release = asyncio.Event()
    collector = MagicMock()
    collector.request_id = "req-active"
    collector.get_nowait.return_value = None

    async def _get_output():
        await release.wait()
        return STREAM_FINISHED

    collector.get = MagicMock(side_effect=_get_output)

    async def _add_request(**kwargs):
        return collector

    mock_llm.add_request = MagicMock(side_effect=_add_request)
    request = _make_text_request(request_id="req-active")
    request.data_parallel_rank = 2
    generator = servicer.Generate(request, _make_generate_context())
    pending = asyncio.create_task(generator.__anext__())
    await asyncio.sleep(0)

    assert servicer.get_active_request_count(2) == 1

    release.set()
    with pytest.raises(StopAsyncIteration):
        await pending
    assert servicer.get_active_request_count(2) == 0
    await generator.aclose()


@pytest.mark.asyncio
async def test_generate_propagates_default_request_seed():
    servicer, _ = _make_park_servicer(default_request_seed=17)

    async for _ in servicer.Generate(_make_text_request(), _make_generate_context()):
        pass

    assert servicer._sampling_params_from_proto.call_args.kwargs["default_seed"] == 17


@pytest.mark.asyncio
async def test_generate_parks_until_resume():
    """A Generate call arriving while the resume event is cleared must park:
    it sends initial metadata but does not call add_request until resumed."""
    servicer, mock_llm = _make_park_servicer()
    servicer.pause_generation_admission()  # clear resume event (engine paused)

    ctx = _make_generate_context()
    gen = servicer.Generate(_make_text_request(), ctx)
    step = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0)  # let it run up to the park

    # Parked: initial metadata sent, but add_request NOT called yet.
    assert ctx.send_initial_metadata.called
    assert not mock_llm.add_request.called
    assert not step.done()

    # Resume → parked request proceeds to add_request and completes.
    servicer.resume_generation_admission()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(step, timeout=1.0)
        await gen.__anext__()
    assert mock_llm.add_request.called
    await gen.aclose()


@pytest.mark.asyncio
async def test_park_does_not_block_admission_drain():
    """A parked request must not hold the admission counter, so
    close_generate_admission() completes immediately."""
    servicer, _ = _make_park_servicer()
    servicer.pause_generation_admission()

    ctx = _make_generate_context()
    gen = servicer.Generate(_make_text_request(), ctx)
    step = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0)  # park

    # Drain must not block (parked request released/never took the counter).
    await asyncio.wait_for(servicer.close_generate_admission(), timeout=1.0)

    # Cleanup: cancel the parked request.
    step.cancel()
    with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
        await step
    await gen.aclose()


@pytest.mark.asyncio
async def test_fail_sync_unblocks_parked_with_abort():
    """fail_generation_admission() wakes a parked request, which aborts
    (finish_reason='abort') without ever calling add_request."""
    servicer, mock_llm = _make_park_servicer()
    servicer.pause_generation_admission()

    ctx = _make_generate_context()
    gen = servicer.Generate(_make_text_request(), ctx)
    step = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0)  # park

    servicer.fail_generation_admission()  # sync failed → wake with failed flag
    resp = await asyncio.wait_for(step, timeout=1.0)
    assert resp.complete.finish_reason == "abort"
    assert not mock_llm.add_request.called
    await gen.aclose()


@pytest.mark.asyncio
async def test_park_cancellation_clean():
    """Cancelling a parked request leaks no admission counter and does not
    call engine.abort (no output_collector yet)."""
    servicer, mock_llm = _make_park_servicer()
    mock_llm.abort = MagicMock()
    servicer.pause_generation_admission()

    ctx = _make_generate_context()
    gen = servicer.Generate(_make_text_request(), ctx)
    step = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0)  # park

    step.cancel()
    with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
        await step
    await gen.aclose()

    # No counter leak: a fresh admission cycle drains immediately.
    assert servicer.active_generate_admissions == 0
    assert not mock_llm.abort.called
