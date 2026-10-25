# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import MagicMock

import pytest

from smg_grpc_servicer.vllm.preemption import PreemptionStatLogger


def _make_stats(req_ids: list[str]):
    stats = MagicMock()
    stats.preemption_req_ids = req_ids
    return stats


def _make_logger(queue=None):
    if queue is None:
        queue = asyncio.Queue()
    # PreemptionStatLogger.__init__ takes (vllm_config, engine_index=0, *, preemption_queue)
    logger = PreemptionStatLogger(MagicMock(), 0, preemption_queue=queue)
    return logger, queue


def test_record_enqueues_nonempty_ids():
    logger, queue = _make_logger()
    logger.record(_make_stats(["req-1", "req-2"]), iteration_stats=None)
    assert not queue.empty()
    assert queue.get_nowait() == ["req-1", "req-2"]


def test_record_skips_empty_ids():
    logger, queue = _make_logger()
    logger.record(_make_stats([]), iteration_stats=None)
    assert queue.empty()


def test_record_skips_none_scheduler_stats():
    logger, queue = _make_logger()
    logger.record(None, iteration_stats=None)
    assert queue.empty()


def test_record_multiple_batches_enqueues_separately():
    logger, queue = _make_logger()
    logger.record(_make_stats(["req-a"]), iteration_stats=None)
    logger.record(_make_stats(["req-b", "req-c"]), iteration_stats=None)
    assert queue.qsize() == 2


@pytest.mark.asyncio
async def test_subscribe_yields_batched_event():
    """SubscribePreemptionEvents drains queue and yields one batched event."""
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(["req-1"])
    queue.put_nowait(["req-2", "req-3"])

    from smg_grpc_servicer.vllm.preemption import drain_preemption_queue

    abort_event = asyncio.Event()
    result = await drain_preemption_queue(queue, abort_event)
    assert set(result) == {"req-1", "req-2", "req-3"}
