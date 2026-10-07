"""Validate ranked head selection without importing an accelerator runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_sampled_token_outside_head_does_not_displace_a_ranked_token():
    source = Path(__file__).parents[1] / "smg_grpc_servicer/vllm/servicer.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_build_top_logprobs")
    method.decorator_list = []
    namespace = {
        "vllm_engine_pb2": SimpleNamespace(
            # `_build_top_logprobs` bulk-fills via the constructor, so the stand-in must accept kwargs.
            TopLogProbs=lambda token_ids=(), values=(): SimpleNamespace(
                token_ids=list(token_ids), values=list(values)
            )
        )
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    entries = {
        99: SimpleNamespace(rank=30, logprob=-9.0),
        3: SimpleNamespace(rank=1, logprob=-0.4),
        7: SimpleNamespace(rank=2, logprob=-1.2),
    }
    result = namespace["_build_top_logprobs"](entries, 2)
    assert result.token_ids == [3, 7]
    assert result.values == [-0.4, -1.2]
    result = namespace["_build_top_logprobs"](entries, 0)
    assert result.token_ids == []
