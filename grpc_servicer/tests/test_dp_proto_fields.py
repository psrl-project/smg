"""Unit tests for DP-aware routing proto fields (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_dp_proto_fields.py
"""

import pytest

pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402


class TestGenerateRequestDataParallelRank:
    def test_defaults_to_zero(self):
        request = vllm_engine_pb2.GenerateRequest()
        assert request.data_parallel_rank == 0

    def test_set_rank_roundtrips(self):
        request = vllm_engine_pb2.GenerateRequest(data_parallel_rank=3)
        parsed = vllm_engine_pb2.GenerateRequest.FromString(request.SerializeToString())
        assert parsed.data_parallel_rank == 3


class TestGenerateRequestPriority:
    def test_unset_by_default(self):
        request = vllm_engine_pb2.GenerateRequest()
        assert not request.HasField("priority")

    def test_zero_is_distinguishable_from_unset(self):
        request = vllm_engine_pb2.GenerateRequest(priority=0)
        assert request.HasField("priority")
        assert request.priority == 0

    def test_signed_priority_roundtrips(self):
        request = vllm_engine_pb2.GenerateRequest(priority=-7)
        parsed = vllm_engine_pb2.GenerateRequest.FromString(request.SerializeToString())
        assert parsed.HasField("priority")
        assert parsed.priority == -7


class TestGetServerInfoResponseDataParallelSize:
    def test_defaults_to_zero_when_unreported(self):
        info = vllm_engine_pb2.GetServerInfoResponse()
        assert info.data_parallel_size == 0

    def test_size_roundtrips(self):
        info = vllm_engine_pb2.GetServerInfoResponse(data_parallel_size=4)
        parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(info.SerializeToString())
        assert parsed.data_parallel_size == 4
