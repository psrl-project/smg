"""Focused validation tests for vLLM routed-expert proto payloads."""

import numpy as np
import pytest

pytest.importorskip("smg_grpc_proto")
pytest.importorskip("vllm")

from smg_grpc_servicer.vllm.servicer import VllmEngineServicer


def test_build_routed_experts_accepts_compact_uint8_and_uint16():
    for dtype in (np.uint8, np.uint16):
        routed_experts = np.arange(24, dtype=dtype).reshape(4, 2, 3)

        proto = VllmEngineServicer._build_routed_experts_tensor(routed_experts)

        assert proto.num_layers == 2
        assert proto.top_k == 3
        assert proto.dtype == np.dtype(dtype).name
        assert proto.data == routed_experts.tobytes()


def test_build_routed_experts_accepts_c_contiguous_singleton_dimension_stride():
    storage = np.arange(12, dtype=np.uint8)
    routed_experts = np.lib.stride_tricks.as_strided(
        storage,
        shape=(4, 1, 3),
        strides=(3, 99, 1),
    )
    assert routed_experts.flags.c_contiguous

    proto = VllmEngineServicer._build_routed_experts_tensor(routed_experts)

    assert proto.num_layers == 1
    assert proto.top_k == 3
    assert proto.data == routed_experts.tobytes()


@pytest.mark.parametrize(
    "routed_experts",
    [
        np.zeros((4, 2), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((4, 2, 3), dtype=np.int16),
    ],
)
def test_build_routed_experts_drops_malformed_present_payload(routed_experts):
    # Routed experts are auxiliary training data: a malformed payload must not
    # abort the stream, so the servicer drops it and the gateway sees a missing
    # field.
    assert VllmEngineServicer._build_routed_experts_tensor(routed_experts) is None


def test_build_routed_experts_copies_non_contiguous_payload():
    routed_experts = np.zeros((4, 2, 3), dtype=np.uint8)[:, :, ::-1]

    proto = VllmEngineServicer._build_routed_experts_tensor(routed_experts)

    expected = np.ascontiguousarray(routed_experts)
    assert proto.dtype == "uint8"
    assert proto.num_layers == 2
    assert proto.top_k == 3
    assert proto.data == expected.tobytes()
