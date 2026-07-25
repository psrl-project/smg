# mypy: ignore-errors
"""
vLLM gRPC Servicer

Implements the VllmEngine gRPC service on top of vLLM's EngineClient.
"""

import asyncio
import hashlib
import itertools
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import grpc
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_servicer.vllm.kv_event_replay import KvEventReplayHub
from smg_grpc_servicer.vllm.preemption import drain_preemption_queue
from smg_grpc_proto.generated import common_pb2
from transformers import BatchFeature
from vllm import PoolingParams, SamplingParams, TokensPrompt
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.engine.protocol import EngineClient
from vllm.inputs.engine import MultiModalInput as VllmMultiModalInput
from vllm.inputs.engine import mm_input, tokens_input
from vllm.logger import init_logger
from vllm.logprobs import PromptLogprobs, SampleLogprobs
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import STREAM_FINISHED, CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

from smg_grpc_servicer.tokenizer_bundle import CHUNK_SIZE, build_tokenizer_zip

logger = init_logger(__name__)
SAMPLING_DEFAULT_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
)


def _filtered_sampling_defaults(params: dict | None) -> dict:
    if not params:
        return {}
    return {
        key: params[key]
        for key in SAMPLING_DEFAULT_KEYS
        if key in params and params[key] is not None
    }


# Proto dtype string → torch dtype
_PROTO_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "int64": torch.int64,
    "uint32": torch.uint32,
}


def _tensor_from_proto(td: vllm_engine_pb2.TensorData) -> torch.Tensor:
    """Deserialize a TensorData proto message into a torch.Tensor."""
    torch_dtype = _PROTO_DTYPE_MAP.get(td.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported proto tensor dtype: {td.dtype!r}")
    return torch.frombuffer(bytearray(td.data), dtype=torch_dtype).reshape(*td.shape)


# Must match model_gateway `PROMPT_OVERFLOW_ERROR_CODE` / PSRL header recognition.
_PROMPT_OVERFLOW_ERROR_CODE = "prompt_overflow"


class PromptOverflowError(ValueError):
    """Prompt exceeds the engine max_model_len (expected RL truncation)."""


async def _abort_invalid_argument(
    context: grpc.aio.ServicerContext,
    exc: BaseException,
) -> None:
    """Abort with INVALID_ARGUMENT; attach prompt_overflow metadata when applicable."""
    metadata = ()
    if isinstance(exc, PromptOverflowError):
        metadata = (("x-smg-error-code", _PROMPT_OVERFLOW_ERROR_CODE),)
    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc), trailing_metadata=metadata)


class VllmEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """
    gRPC servicer implementing the VllmEngine service.

    Handles 8 RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings
    - HealthCheck: Health probe
    - Abort: Cancel requests out-of-band
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    - GetTokenizer: Stream tokenizer artifacts
    - SubscribePreemptionEvents: Stream scheduler-preempted request IDs to SMG
    """

    def __init__(self, async_llm: EngineClient, start_time: float, preemption_queue: asyncio.Queue | None = None, kv_cache_manager=None, kv_transfer_stats_log_interval_s: float = 30.0, enable_kv_event_replay: bool = False,):
        """
        Initialize the servicer.

        Args:
            async_llm: The EngineClient instance (e.g. AsyncLLM)
            start_time: The server start time, in seconds since epoch
            preemption_queue: Optional queue for preemption events; if None,
                a new empty queue is created (no events will be delivered
                unless a PreemptionStatLogger is wired to the same queue)
            kv_cache_manager: Optional PSRL ``KVCacheManager`` enabling the
                TransferKv/PinKv/UnpinKv RPCs (cross-instance KV migration).
                When None those RPCs return UNIMPLEMENTED.
            enable_kv_event_replay: When True, ``SubscribeKvEvents`` is served by
                a long-lived ``KvEventReplayHub`` that buffers recent batches so
                gateway gap-replay can recover missed sequences. When False
                (default), fall back to the original per-subscription inline ZMQ
                loop with no buffering — the known-good baseline.
        """
        self.enable_kv_event_replay = enable_kv_event_replay
        self.engine = async_llm
        self.start_time = start_time
        self.preemption_queue: asyncio.Queue[list[str]] = (
            preemption_queue if preemption_queue is not None else asyncio.Queue()
        )
        self.kv_cache_manager = kv_cache_manager
        self.generate_admission_open = True
        self.active_generate_admissions = 0
        self.generate_admissions_drained = asyncio.Event()
        self.generate_admissions_drained.set()
        # Cleared while paused for a weight sync; new Generate calls park on it.
        self.generation_resume_event = asyncio.Event()
        self.generation_resume_event.set()
        self.generation_resume_failed = False

        # Parse the native vLLM KV-event publisher config so SubscribeKvEvents
        # can bridge the local ZMQ stream to gRPC (event-driven cache-aware
        # routing). None → events not enabled → SubscribeKvEvents is UNIMPLEMENTED.
        self.kv_events_config = None
        try:
            cfg = getattr(self.engine.vllm_config, "kv_events_config", None)
            if cfg is not None and getattr(cfg, "publisher", None) == "zmq":
                self.kv_events_config = cfg
                logger.info(
                    "KV events enabled: endpoint=%s topic=%s",
                    cfg.endpoint,
                    cfg.topic,
                )
        except Exception as e:  # noqa: BLE001 - defensive: never block startup
            logger.warning("Failed to read kv_events_config: %s", e)

        # Monotonic event-id counter for converted KvCacheEvents.
        self.kv_event_id_counter = 0

        self._kv_replay_hub: KvEventReplayHub | None = None
        if self.kv_events_config is not None and self.enable_kv_event_replay:
            parallel_config = self.engine.vllm_config.parallel_config
            dp_size = int(parallel_config.data_parallel_size)
            kv_decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
            self._kv_replay_hub = KvEventReplayHub(
                endpoint=self.kv_events_config.endpoint,
                topic=self.kv_events_config.topic,
                dp_size=dp_size,
                decode=kv_decoder.decode,
                convert=self._convert_kv_event_batch,
                offset_endpoint_port=ZmqEventPublisher.offset_endpoint_port,
            )
            self._kv_replay_hub.start()
            logger.info(
                "KV event replay hub started dp_size=%d endpoint=%s",
                dp_size,
                self.kv_events_config.endpoint,
            )

        # KV-transfer statistics, mirroring psrl_agent's RolloutRouter counters.
        # SMG drives transfers from the Rust gateway, so this servicer's
        # TransferKv handler is the single source-side choke point where every
        # migration's outcome is observable. _kv_transfer_stats_logged_at gates
        # the periodic summary so it prints at most once per interval.
        self._kv_transfer_stats = {
            "succeeded": 0,        # transfer_direct moved >0 tokens on all ranks
            "returned_false": 0,   # transfer_direct returned False (src miss / layout mismatch)
            "exception": 0,        # transfer_direct raised
            "empty_tokens": 0,     # request carried no tokens
        }
        self._kv_transfer_stats_logged_at = time.time()
        self._kv_transfer_stats_log_interval_s = kv_transfer_stats_log_interval_s
        logger.info("VllmEngineServicer initialized")

    def _record_kv_transfer(self, outcome: str) -> None:
        """Increment a TransferKv outcome counter and periodically log a summary.

        Args:
            outcome: one of the keys in ``self._kv_transfer_stats``.
        """
        self._kv_transfer_stats[outcome] = self._kv_transfer_stats.get(outcome, 0) + 1
        if self._kv_transfer_stats_log_interval_s <= 0:
            return
        now = time.time()
        if now - self._kv_transfer_stats_logged_at < self._kv_transfer_stats_log_interval_s:
            return
        self._kv_transfer_stats_logged_at = now
        s = self._kv_transfer_stats
        total = sum(s.values())
        attempted = total - s["empty_tokens"]
        rate = (s["succeeded"] / attempted * 100.0) if attempted else 0.0
        logger.warning(
            "[KVTransfer] instance=%s stats: ok=%d returned_false=%d exc=%d "
            "empty=%d (total=%d, success_rate=%.1f%%)",
            getattr(self.kv_cache_manager.config, "lmcache_instance_id", "?")
            if self.kv_cache_manager is not None else "?",
            s["succeeded"],
            s["returned_false"],
            s["exception"],
            s["empty_tokens"],
            total,
            rate,
        )

    async def close_generate_admission(self) -> None:
        """Reject new Generate calls and wait for admitted calls to register."""
        self.generate_admission_open = False
        await self.generate_admissions_drained.wait()

    async def open_generate_admission(self) -> None:
        self.generate_admission_open = True

    def pause_generation_admission(self) -> None:
        """Clear the park gate so new Generate calls wait for engine resume.

        Must be called BEFORE close_generate_admission() to preserve the
        invariant (event set ⟹ engine live or sync-failed).
        """
        self.generation_resume_failed = False
        self.generation_resume_event.clear()

    def resume_generation_admission(self) -> None:
        """Set the park gate after a successful sync, waking parked Generate
        calls so they proceed to add_request on the now-live engine."""
        self.generation_resume_failed = False
        self.generation_resume_event.set()

    def fail_generation_admission(self) -> None:
        """Set the park gate after a FAILED sync (replica quarantined).

        Wakes parked Generate calls with generation_resume_failed=True so they
        abort → gateway loopback re-routes them to a healthy instance instead
        of hanging forever on the resume event that will never come.
        """
        self.generation_resume_failed = True
        self.generation_resume_event.set()

    def begin_generate_admission(self) -> bool:
        if not self.generate_admission_open:
            return False
        self.active_generate_admissions += 1
        self.generate_admissions_drained.clear()
        return True

    def finish_generate_admission(self) -> None:
        self.active_generate_admissions -= 1
        if self.active_generate_admissions == 0:
            self.generate_admissions_drained.set()

    @staticmethod
    def abort_response() -> vllm_engine_pb2.GenerateResponse:
        return vllm_engine_pb2.GenerateResponse(
            complete=vllm_engine_pb2.GenerateComplete(
                output_ids=[],
                finish_reason="abort",
                prompt_tokens=0,
                completion_tokens=0,
                cached_tokens=0,
                index=0,
            )
        )

    async def Generate(
        self,
        request: vllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[vllm_engine_pb2.GenerateResponse, None]:
        """
        Handle streaming generation requests.

        Supports n>1 by sending separate chunk/complete messages for each output index.
        When streaming with n>1, chunks for different indices are interleaved.

        Args:
            request: The GenerateRequest protobuf
            context: gRPC context

        Yields:
            GenerateResponse protobuf messages (streaming)
        """
        request_id = request.request_id
        input_type = request.WhichOneof("input")
        has_preprocessed_mm = request.HasField("mm_inputs") and request.mm_inputs.HasField(
            "pixel_values"
        )
        logger.info(
            "Generate request %s: input_type=%s, stream=%s, preprocessed_mm=%s",
            request_id,
            input_type,
            request.stream,
            has_preprocessed_mm,
        )

        output_collector = None
        registration_pending = False
        try:
            arrival_time = time.time()

            if has_preprocessed_mm and input_type == "tokenized":
                # Preprocessed multimodal from Rust router.
                # Token IDs already have expanded placeholders; tensors are
                # ready for the model. Bypass the renderer entirely.
                prompt = self._build_preprocessed_mm_inputs(request.tokenized, request.mm_inputs)
                prompt["arrival_time"] = arrival_time
            elif input_type == "tokenized":
                prompt: TokensPrompt = {"prompt_token_ids": list(request.tokenized.input_ids)}
                if request.tokenized.original_text:
                    prompt["prompt"] = request.tokenized.original_text
                prompt = self.engine.renderer.process_for_engine(prompt, arrival_time=arrival_time)
            else:
                prompt = request.text

            # Validate prompt length before any park/admission/engine work.
            # This catches overlong prompts at the SMG boundary and returns
            # INVALID_ARGUMENT (400) with x-smg-error-code=prompt_overflow
            # instead of letting them reach the engine as EngineGenerateError (500).
            if input_type == "tokenized":
                prompt_len = len(request.tokenized.input_ids)
                max_model_len = self.engine.model_config.max_model_len
                if prompt_len > max_model_len:
                    raise PromptOverflowError(
                        f"The prompt (length {prompt_len}) is longer than the "
                        f"maximum model length of {max_model_len}."
                    )
                if prompt_len == max_model_len and self.engine.model_config.runner_type == "generate":
                    raise PromptOverflowError(
                        f"The prompt (length {prompt_len}) plus the number of "
                        f"requested output tokens (at least 1) is longer than the "
                        f"maximum model length of {max_model_len}."
                    )

            # Build sampling params with detokenize=False
            _version_tag = (
                str(self.kv_cache_manager.current_version)
                if self.kv_cache_manager is not None
                and getattr(self.kv_cache_manager.config, "multi_version_kv", False)
                else None
            )
            sampling_params = self._sampling_params_from_proto(
                request.sampling_params,
                stream=request.stream,
                kv_transfer_params=request.kv_transfer_params
                if request.HasField("kv_transfer_params")
                else None,
                model_version_tag=_version_tag,
            )
            tokenization_kwargs = self._tokenization_kwargs_from_proto(request.sampling_params)

            # Extract logprobs configuration
            num_logprobs = sampling_params.logprobs
            num_prompt_logprobs = sampling_params.prompt_logprobs

            # Track which indices have sent their first chunk
            seen_indices: set[int] = set()

            # Send gRPC response headers immediately so the tonic client's
            # `client.generate().await` resolves without waiting for the first
            # output token. This is critical for the SMG pause barrier: the
            # dispatch handoff permit is held until `generate().await` returns,
            # and without this call the permit would be blocked until the vLLM
            # scheduler produces the first token. Sending it BEFORE the park
            # below ensures the permit is released before we wait, so a
            # concurrent weight-sync pause barrier is never stalled by a parked
            # request.
            await context.send_initial_metadata(())

            # Park-and-wait: if the engine is mid weight-sync (resume event
            # cleared), wait here instead of calling add_request on a not-ready
            # engine (which would raise EngineDead -> 500 -> circuit breaker).
            if not self.generation_resume_event.is_set():
                logger.info("Generate %s parking: engine paused for PS sync", request_id)
                await self.generation_resume_event.wait()
                if self.generation_resume_failed:
                    # Woken by fail_sync: the replica is quarantined (sync
                    # failed), so abort and let the gateway re-route elsewhere.
                    logger.warning(
                        "Generate %s aborting: sync failed, replica quarantined", request_id
                    )
                    yield self.abort_response()
                    return
                logger.info("Generate %s resumed from park", request_id)

            # Admission counter guards only the add_request registration window.
            admitted = self.begin_generate_admission()
            if not admitted:
                # Rare: a new pause raced in between wake and here.
                yield self.abort_response()
                return
            registration_pending = True

            output_collector = await self.engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=sampling_params,
                tokenization_kwargs=tokenization_kwargs,
            )
            registration_pending = False
            self.finish_generate_admission()

            finished = False
            while not finished:
                output = output_collector.get_nowait() or await output_collector.get()
                if output is STREAM_FINISHED:
                    break
                finished = output.finished
                # For streaming, send chunks for EACH completion output (n outputs)
                if request.stream:
                    for completion in output.outputs:
                        idx = completion.index
                        is_first = idx not in seen_indices
                        seen_indices.add(idx)

                        # Send chunk with delta data (Rust accumulates for vLLM)
                        yield self._chunk_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                            is_first_chunk=is_first,
                        )

                        # Send Complete when sequence finishes (n>1 support)
                        if completion.finish_reason:
                            yield self._complete_response(
                                output,
                                completion=completion,
                                num_logprobs=num_logprobs,
                                num_prompt_logprobs=num_prompt_logprobs,
                            )

                # For non-streaming, send complete response when finished
                if output.finished and not request.stream:
                    for completion in output.outputs:
                        yield self._complete_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                        )

        except (asyncio.CancelledError, GeneratorExit):
            if output_collector is not None:
                await self.engine.abort(output_collector.request_id, internal=True)
            raise
        except ValueError as e:
            # Invalid request error (equiv to 400). PromptOverflowError attaches
            # x-smg-error-code=prompt_overflow so the gateway can demote logs.
            await _abort_invalid_argument(context, e)
        except Exception as e:
            logger.exception("Error in Generate for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            if registration_pending:
                self.finish_generate_admission()
            if output_collector is not None:
                output_collector.close()

    async def Embed(
        self,
        request: vllm_engine_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.EmbedResponse:
        """
        Handle embedding requests.

        Calls vLLM's encode() API with PoolingParams and returns the embedding vector.

        Args:
            request: The EmbedRequest protobuf
            context: gRPC context

        Returns:
            EmbedResponse protobuf
        """
        request_id = request.request_id
        logger.info("Embed request %s", request_id)

        try:
            if not request.HasField("tokenized"):
                raise ValueError("EmbedRequest requires tokenized input")

            prompt = tokens_input(
                prompt_token_ids=list(request.tokenized.input_ids),
                prompt=request.tokenized.original_text or None,
            )

            pooling_params = PoolingParams(task="embed")

            # encode() is an async generator; collect the final result
            final_output = None
            async for output in self.engine.encode(
                prompt=prompt,
                pooling_params=pooling_params,
                request_id=request_id,
            ):
                final_output = output

            if final_output is None or not final_output.finished:
                msg = f"Embed request {request_id} did not produce a result"
                logger.warning(msg)
                await context.abort(grpc.StatusCode.INTERNAL, msg)

            embedding = final_output.outputs.data.tolist()

            return vllm_engine_pb2.EmbedResponse(
                embedding=embedding,
                prompt_tokens=len(final_output.prompt_token_ids),
                embedding_dim=len(embedding),
            )

        except grpc.aio.AbortError:
            raise
        except ValueError as e:
            logger.warning("Embed invalid request %s: %s", request_id, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Embed failed for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def HealthCheck(
        self,
        request: vllm_engine_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.HealthCheckResponse:
        """
        Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        is_healthy = not self.engine.errored
        message = "Health" if is_healthy else "Engine is not alive"

        logger.info("HealthCheck request: healthy=%s, message=%s", is_healthy, message)

        return vllm_engine_pb2.HealthCheckResponse(healthy=is_healthy, message=message)

    async def Abort(
        self,
        request: vllm_engine_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.AbortResponse:
        """
        Out-of-band abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_ids = request.request_ids
        logger.info("Abort requests: %s", request_ids)

        await self.engine.abort(request_ids)
        return vllm_engine_pb2.AbortResponse()

    async def GetModelInfo(
        self,
        request: vllm_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetModelInfoResponse:
        """
        Handle model info requests.

        Args:
            request: The GetModelInfoRequest protobuf
            context: gRPC context

        Returns:
            GetModelInfoResponse protobuf
        """
        model_config = self.engine.model_config
        hf_config = model_config.hf_config

        # eos_token_id can be int or list[int]
        eos = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list):
            eos_token_ids = eos
        else:
            eos_token_ids = []

        sampling_defaults = _filtered_sampling_defaults(
            model_config.get_diff_sampling_param() or {}
        )

        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.model,
            is_generation=model_config.runner_type == "generate",
            max_context_length=model_config.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            supports_vision=model_config.is_multimodal_model,
            served_model_name=model_config.served_model_name or model_config.model,
            tokenizer_path=model_config.tokenizer or "",
            model_type=getattr(hf_config, "model_type", "") or "",
            architectures=model_config.architectures or [],
            eos_token_ids=eos_token_ids,
            pad_token_id=getattr(hf_config, "pad_token_id", None) or 0,
            bos_token_id=getattr(hf_config, "bos_token_id", None) or 0,
            max_req_input_len=model_config.max_model_len,
            default_sampling_params_json=(
                json.dumps(sampling_defaults, separators=(",", ":")) if sampling_defaults else ""
            ),
        )

    async def GetServerInfo(
        self,
        request: vllm_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetServerInfoResponse:
        """
        Handle server info requests.

        Args:
            request: The GetServerInfoRequest protobuf
            context: gRPC context

        Returns:
            GetServerInfoResponse protobuf
        """
        kv_connector = ""
        kv_role = ""
        kv_transfer_config = self.engine.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            kv_connector = kv_transfer_config.kv_connector or ""
            kv_role = kv_transfer_config.kv_role or ""

        parallel_config = self.engine.vllm_config.parallel_config
        data_parallel_size = parallel_config.data_parallel_size
        tensor_parallel_size = parallel_config.tensor_parallel_size
        pipeline_parallel_size = parallel_config.pipeline_parallel_size

        return vllm_engine_pb2.GetServerInfoResponse(
            kv_connector=kv_connector,
            kv_role=kv_role,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

    async def GetTokenizer(
        self,
        request: common_pb2.GetTokenizerRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.GetTokenizerChunk]:
        """Stream tokenizer artifacts as a ZIP bundle.

        Resolves the tokenizer directory from model_config, zips all relevant
        tokenizer files, and streams them as GetTokenizerChunk messages.
        The final chunk carries the SHA-256 fingerprint of the full archive.
        """
        logger.info("Receive GetTokenizer request")

        tokenizer_path = self.engine.model_config.tokenizer
        if not tokenizer_path:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Tokenizer path is not configured on this server.",
            )
        # TODO: model_config.tokenizer may be an HF model ID (e.g. "meta-llama/...")
        # rather than a local path. vLLM does not resolve it on the config object.
        # For now, GetTokenizer only works when vLLM is started with a local path.
        tokenizer_dir = Path(tokenizer_path)

        # Build ZIP archive in memory
        try:
            zip_buffer = build_tokenizer_zip(tokenizer_dir)
        except Exception as e:
            logger.exception("Failed to build tokenizer ZIP")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        zip_data = zip_buffer.getbuffer()
        sha256 = hashlib.sha256(zip_data).hexdigest()

        logger.info(
            "Streaming tokenizer bundle: %d bytes, sha256=%s",
            len(zip_data),
            sha256,
        )

        # Stream chunks; SHA-256 only on the final chunk
        offset = 0
        total = len(zip_data)
        while offset < total:
            end = min(offset + CHUNK_SIZE, total)
            is_last = end == total
            yield common_pb2.GetTokenizerChunk(
                data=bytes(zip_data[offset:end]),
                sha256=sha256 if is_last else "",
            )
            offset = end

    async def SubscribePreemptionEvents(
        self,
        request: vllm_engine_pb2.SubscribePreemptionEventsRequest,
        context: grpc.aio.ServicerContext,
    ):
        """Stream preemption events to the SMG gateway.

        Blocks until preempted request IDs are available, then yields a batched
        ``PreemptionEvent``.  Runs indefinitely until the client disconnects.

        ``drain_preemption_queue`` races the blocking ``queue.get()`` against
        an abort event that is set when the gRPC context is cancelled.  This
        guarantees the handler coroutine exits promptly on disconnect, preventing
        the zombie-consumer / dual-consumer race that would arise if we awaited
        the queue unconditionally.
        """
        abort_event = asyncio.Event()
        context.add_done_callback(lambda _ctx: abort_event.set())

        while not context.cancelled():
            req_ids = await drain_preemption_queue(self.preemption_queue, abort_event)
            if req_ids is None:
                # abort_event fired — client disconnected.
                break
            yield vllm_engine_pb2.PreemptionEvent(
                request_ids=req_ids,
                timestamp_ns=time.time_ns(),
            )

    # ========== KV cache event streaming (event-driven cache-aware routing) ==========

    async def SubscribeKvEvents(
        self,
        request: common_pb2.SubscribeKvEventsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.KvEventBatch]:
        """Bridge vLLM's internal ZMQ KV cache event stream to gRPC.

        vLLM publishes one ZMQ stream per data-parallel rank (port offset by
        ``dp_rank``) with an independent monotonic sequence counter. The router
        subscribes once per ``(url, dp_rank)`` and passes the rank in the request,
        so we connect to exactly that rank's publisher endpoint.
        The publisher's native sequence numbers are forwarded as-is so SMG's gap
        detection and ``start_sequence_number`` replay work unchanged.
        """
        if self._kv_replay_hub is None:
            # Replay disabled: fall back to the original per-subscription inline
            # ZMQ loop (no buffering). Requires kv_events_config to be set.
            if self.kv_events_config is None:
                await context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    "KV cache events not enabled. Launch vLLM with a kv_events_config "
                    'whose publisher == "zmq".',
                )
                return

            config = self.kv_events_config
            dp_rank = request.dp_rank if request.HasField("dp_rank") else 0

            # The publisher binds e.g. "tcp://*:5557"; connect on localhost and offset
            # the port by the requested dp_rank to reach that rank's stream.
            pub_endpoint = config.endpoint.replace("*", "127.0.0.1")
            pub_endpoint = ZmqEventPublisher.offset_endpoint_port(pub_endpoint, dp_rank)

            zmq_ctx = zmq.asyncio.Context.instance()
            sub_socket = zmq_ctx.socket(zmq.SUB)
            sub_socket.subscribe(config.topic.encode("utf-8"))
            sub_socket.connect(pub_endpoint)

            logger.info(
                "SubscribeKvEvents: connected to ZMQ endpoint %s (dp_rank=%d, no replay)",
                pub_endpoint,
                dp_rank,
            )

            # Send headers immediately so the tonic client's subscribe future resolves
            # before the first event arrives.
            await context.send_initial_metadata(())

            decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

            try:
                while not context.cancelled():
                    try:
                        frames = await asyncio.wait_for(sub_socket.recv_multipart(), timeout=1.0)
                    except (TimeoutError, asyncio.TimeoutError):
                        continue

                    # ZMQ multipart layout: [topic, seq(8B big-endian), msgpack payload]
                    if len(frames) < 3:
                        continue

                    zmq_seq = int.from_bytes(frames[1], "big")
                    payload = frames[2]
                    try:
                        raw_batch = decoder.decode(payload)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to decode KV event batch: %s", e)
                        continue

                    yield self._convert_kv_event_batch(raw_batch, zmq_seq)
            except asyncio.CancelledError:
                pass
            finally:
                sub_socket.close(linger=0)
                logger.info("SubscribeKvEvents: stream closed (dp_rank=%d)", dp_rank)
            return

        dp_rank = request.dp_rank if request.HasField("dp_rank") else 0
        start_seq = int(request.start_sequence_number)

        logger.info(
            "SubscribeKvEvents: dp_rank=%d start_sequence_number=%d (replay then live)",
            dp_rank,
            start_seq,
        )

        try:
            async for batch in self._kv_replay_hub.subscribe(
                dp_rank,
                start_seq,
                send_initial_metadata=lambda: context.send_initial_metadata(()),
                is_cancelled=context.cancelled,
            ):
                yield batch
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("SubscribeKvEvents: stream closed (dp_rank=%d)", dp_rank)

    def _convert_kv_event_batch(
        self, raw_batch: KVEventBatch, seq_num: int
    ) -> common_pb2.KvEventBatch:
        """Convert a vLLM ZMQ ``KVEventBatch`` to the proto ``KvEventBatch``."""
        proto_batch = common_pb2.KvEventBatch(
            sequence_number=seq_num,
            timestamp=raw_batch.ts,
        )
        if raw_batch.data_parallel_rank is not None:
            proto_batch.dp_rank = raw_batch.data_parallel_rank

        for event in raw_batch.events:
            proto_event = self._convert_kv_event(event)
            if proto_event is not None:
                proto_batch.events.append(proto_event)
        return proto_batch

    @staticmethod
    def _block_hash_to_int(block_hash) -> int:
        """Coerce a vLLM ``ExternalBlockHash`` (``bytes | int``) to int64.

        The proto carries block hashes as ``int64``; vLLM may emit raw ``bytes``.
        We fold bytes into a stable 63-bit int (positive, fits signed int64).
        """
        if isinstance(block_hash, (bytes, bytearray)):
            return int.from_bytes(block_hash[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
        return int(block_hash) & 0x7FFF_FFFF_FFFF_FFFF

    def _convert_kv_event(self, event) -> common_pb2.KvCacheEvent | None:
        """Convert a single vLLM raw KV event to a proto ``KvCacheEvent``.

        The block's ``medium`` is mapped to ``cache_level`` so the router can
        score GPU vs LMCache (off-GPU) hits with different weights:
          - ``medium == "GPU"`` (or None for legacy single-tier) → cache_level 0
          - anything else (LMCache CPU/disk offload) → cache_level 1
        """
        self.kv_event_id_counter += 1
        event_id = self.kv_event_id_counter

        if isinstance(event, BlockStored):
            cache_level = 0 if (event.medium is None or event.medium == "GPU") else 1

            # region agent log
            try:
                _dbg_a = globals().setdefault(
                    "_DBG_KVEVENT_AGG",
                    {"stored_level0": 0, "stored_level1": 0, "mediums": {}},
                )
                _dbg_a["stored_level0" if cache_level == 0 else "stored_level1"] += 1
                _dbg_mk = f"{type(event.medium).__name__}:{event.medium!r}"
                _dbg_a["mediums"][_dbg_mk] = _dbg_a["mediums"].get(_dbg_mk, 0) + 1
                _dbg_tot = _dbg_a["stored_level0"] + _dbg_a["stored_level1"]
                if _dbg_tot % 2000 == 0 or (
                    cache_level == 1 and _dbg_a["stored_level1"] <= 3
                ):
                    import json as _dbg_json
                    import os as _dbg_os
                    import time as _dbg_time

                    with open(
                        "/apdcephfs_zwfy10/share_303541817/lhy/.cursor/debug-48f20e.log",
                        "a",
                    ) as _dbg_f:
                        _dbg_f.write(
                            _dbg_json.dumps(
                                {
                                    "sessionId": "48f20e",
                                    "runId": "run1",
                                    "hypothesisId": "H1+H5",
                                    "location": "smg_grpc_servicer/vllm/servicer.py:905",
                                    "message": "KV BlockStored events bridged to router, by cache_level",
                                    "data": dict(
                                        _dbg_a,
                                        pid=_dbg_os.getpid(),
                                        last_block_size=int(event.block_size),
                                        last_n_hashes=len(event.block_hashes),
                                        last_n_token_ids=len(event.token_ids or []),
                                    ),
                                    "timestamp": int(_dbg_time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
            except Exception:
                pass
            # endregion

            blocks = []
            for i, bh in enumerate(event.block_hashes):
                start = i * event.block_size
                end = start + event.block_size
                block = common_pb2.KvBlock(
                    block_hash=self._block_hash_to_int(bh),
                    token_ids=event.token_ids[start:end],
                    block_size=event.block_size,
                    cache_level=cache_level,
                )
                if event.lora_id is not None:
                    block.lora_id = event.lora_id
                blocks.append(block)

            stored = common_pb2.KvBlocksStored(blocks=blocks)
            if event.parent_block_hash is not None:
                stored.parent_block_hash = self._block_hash_to_int(event.parent_block_hash)
            return common_pb2.KvCacheEvent(event_id=event_id, stored=stored)

        elif isinstance(event, BlockRemoved):
            cache_level = 0 if (event.medium is None or event.medium == "GPU") else 1
            removed = common_pb2.KvBlocksRemoved(
                block_hashes=[self._block_hash_to_int(h) for h in event.block_hashes],
                cache_level=cache_level,
            )
            return common_pb2.KvCacheEvent(event_id=event_id, removed=removed)

        elif isinstance(event, AllBlocksCleared):
            return common_pb2.KvCacheEvent(
                event_id=event_id, cleared=common_pb2.KvCacheCleared()
            )

        return None

    # ========== KV cache transfer (cross-instance migration) ==========

    async def TransferKv(
        self,
        request: vllm_engine_pb2.TransferKvRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.TransferKvResponse:
        """Push this instance's cached prefix for ``tokens`` to a peer instance.

        Calls the local ``KVCacheManager.transfer_direct`` (ZMQ MoveWorkerMsg to
        the local LMCacheWorker), addressing the destination by its
        ``dst_instance_id`` (resolved to a peer URL via the broadcast registry).
        """
        if self.kv_cache_manager is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV transfer not enabled: no KVCacheManager attached.",
            )
            return vllm_engine_pb2.TransferKvResponse()

        tokens = list(request.tokens)
        if not tokens:
            self._record_kv_transfer("empty_tokens")
            return vllm_engine_pb2.TransferKvResponse(
                success=False, num_tokens=0, error="empty token sequence"
            )

        src_instance = self.kv_cache_manager.config.lmcache_instance_id
        src_backend = request.src_backend or "LocalCPUBackend"
        dst_backend = request.dst_backend or "LocalCPUBackend"

        # Fallback seed: the authoritative per-rank peer registry is installed once
        # at init by RolloutCoordinator._broadcast_peer_registry, so in normal
        # operation every instance is already present here. The only legitimate case
        # for seeding from the request is an instance added AFTER that broadcast.
        # The request carries a single (rank-0) dst_peer_url, so we can only build a
        # one-rank list — correct just for world_size==1. We therefore only seed when
        # the instance is absent (never clobbering a fuller broadcast list) and log
        # loudly, because hitting this path for an existing instance indicates the
        # broadcast did not run / did not reach this replica and should be fixed.
        if request.dst_peer_url and request.dst_instance_id not in self.kv_cache_manager.peer_registry:
            logger.error(
                "TransferKv seeding peer registry for %s from request dst_peer_url: "
                "this should not normally trigger (broadcast registry is the source "
                "of truth) unless this is a newly-added instance. Seeding a single-rank "
                "list, which is only correct for world_size==1.",
                request.dst_instance_id,
            )
            try:
                self.kv_cache_manager.set_peer_registry(
                    {request.dst_instance_id: [request.dst_peer_url]}
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to seed peer registry: %s", e)

        try:
            ok = await self.kv_cache_manager.transfer_direct(
                tokens,
                (src_instance, src_backend),
                (request.dst_instance_id, dst_backend),
                request.copy,
                dst_model_version=(
                    request.dst_model_version
                    if getattr(self.kv_cache_manager.config, "multi_version_kv", False)
                    else -1
                ),
            )
        except Exception as e:  # noqa: BLE001
            self._record_kv_transfer("exception")
            logger.warning("TransferKv failed: %s", e)
            return vllm_engine_pb2.TransferKvResponse(
                success=False, num_tokens=0, error=str(e)
            )

        if ok:
            self._record_kv_transfer("succeeded")
            return vllm_engine_pb2.TransferKvResponse(success=True, num_tokens=len(tokens))
        self._record_kv_transfer("returned_false")
        err = getattr(self.kv_cache_manager, "_last_transfer_error", "transfer returned False")
        return vllm_engine_pb2.TransferKvResponse(success=False, num_tokens=0, error=str(err))

    async def PinKv(
        self,
        request: vllm_engine_pb2.PinKvRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.PinKvResponse:
        """Pin the cached prefix for ``tokens`` to protect it from LRU eviction."""
        if self.kv_cache_manager is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV pin not enabled: no KVCacheManager attached.",
            )
            return vllm_engine_pb2.PinKvResponse()

        tokens = list(request.tokens)
        targets = list(request.targets) or ["gpu", "backend"]
        try:
            ok = await self.kv_cache_manager.pin(tokens, targets)
            return vllm_engine_pb2.PinKvResponse(success=bool(ok))
        except Exception as e:  # noqa: BLE001
            logger.warning("PinKv failed: %s", e)
            return vllm_engine_pb2.PinKvResponse(success=False, error=str(e))

    async def UnpinKv(
        self,
        request: vllm_engine_pb2.UnpinKvRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.UnpinKvResponse:
        """Unpin a previously-pinned prefix, releasing the pin budget."""
        if self.kv_cache_manager is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV unpin not enabled: no KVCacheManager attached.",
            )
            return vllm_engine_pb2.UnpinKvResponse()

        tokens = list(request.tokens)
        targets = list(request.targets) or ["gpu", "backend"]
        try:
            ok = await self.kv_cache_manager.unpin(tokens, targets)
            return vllm_engine_pb2.UnpinKvResponse(success=bool(ok))
        except Exception as e:  # noqa: BLE001
            logger.warning("UnpinKv failed: %s", e)
            return vllm_engine_pb2.UnpinKvResponse(success=False, error=str(e))

    # ========== Helper methods ==========

    def _build_preprocessed_mm_inputs(
        self,
        tokenized: vllm_engine_pb2.TokenizedInput,
        mm_proto: vllm_engine_pb2.MultimodalInputs,
    ) -> VllmMultiModalInput:
        """Build vLLM MultiModalInput from preprocessed proto data.

        Bypasses HF processor entirely — pixel values and model-specific
        tensors were already computed by the Rust router.  Field layouts
        (batched / flat / shared) are also determined by the router via
        ``batched_keys`` and ``flat_keys`` proto fields.
        """
        prompt_token_ids = list(tokenized.input_ids)
        num_images = len(mm_proto.mm_placeholders)

        # Deserialize all tensors from proto
        hf_dict: dict[str, torch.Tensor] = {
            "pixel_values": _tensor_from_proto(mm_proto.pixel_values),
        }
        for key, td in mm_proto.model_specific_tensors.items():
            hf_dict[key] = _tensor_from_proto(td)

        # Cast floating-point tensors to model dtype (e.g. bfloat16).
        # This mirrors _postprocess_output in multimodal/processing/context.py
        # which is skipped when bypassing the HF processor.
        model_dtype = self.engine.model_config.dtype
        for key in hf_dict:
            if hf_dict[key].is_floating_point():
                hf_dict[key] = hf_dict[key].to(dtype=model_dtype)

        cpu_keys = set(mm_proto.keep_on_cpu_keys)

        # Field configs are fully determined by the Rust router.
        batched = set(mm_proto.batched_keys)
        flat = dict(mm_proto.flat_keys)
        fields_config: dict[str, MultiModalFieldConfig] = {}
        flat_sizes_cache: dict[str, torch.Tensor] = {}
        for key in hf_dict:
            on_cpu = key in cpu_keys
            if key in batched:
                fields_config[key] = MultiModalFieldConfig.batched("image", keep_on_cpu=on_cpu)
            elif key in flat:
                sizes_key = flat[key]
                if sizes_key not in flat_sizes_cache:
                    flat_sizes_cache[sizes_key] = hf_dict[sizes_key].flatten().to(torch.int64)
                fields_config[key] = MultiModalFieldConfig.flat_from_sizes(
                    "image", flat_sizes_cache[sizes_key], keep_on_cpu=on_cpu
                )
            else:
                fields_config[key] = MultiModalFieldConfig.shared("image", num_images)

        batch_feature = BatchFeature(hf_dict, tensor_type="pt")
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(batch_feature, fields_config)

        # Build mm_hashes: dict[str, list[str]]
        mm_hashes: dict[str, list[str]] = {}
        if mm_proto.mm_hashes:
            mm_hashes["image"] = list(mm_proto.mm_hashes)

        # Build mm_placeholders: dict[str, list[PlaceholderRange]]
        # When structural tokens (e.g. <|image_start|>, separators) are present
        # in the placeholder range, we must set is_embed so vLLM only scatters
        # encoder embeddings into patch-token positions (im_token_id).
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if mm_proto.mm_placeholders:
            im_token_id = mm_proto.im_token_id if mm_proto.HasField("im_token_id") else None
            # Pre-convert to tensor for vectorized mask building
            prompt_ids_tensor = (
                torch.tensor(prompt_token_ids, dtype=torch.int64)
                if im_token_id is not None
                else None
            )
            placeholders = []
            for p in mm_proto.mm_placeholders:
                is_embed = None
                if prompt_ids_tensor is not None:
                    mask = prompt_ids_tensor[p.offset : p.offset + p.length] == im_token_id
                    # Only set is_embed when there are non-embed positions,
                    # otherwise None means "all positions are embeds" which is
                    # both correct and avoids unnecessary overhead.
                    if not mask.all():
                        is_embed = mask
                placeholders.append(
                    PlaceholderRange(offset=p.offset, length=p.length, is_embed=is_embed)
                )
            mm_placeholders["image"] = placeholders

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            prompt=tokenized.original_text or None,
        )

    @staticmethod
    def _sampling_params_from_proto(
        params: vllm_engine_pb2.SamplingParams,
        stream: bool = True,
        kv_transfer_params: vllm_engine_pb2.KvTransferParams | None = None,
        model_version_tag: str | None = None,
    ) -> SamplingParams:
        """
        Convert protobuf SamplingParams to vLLM SamplingParams.

        Args:
            params: Protobuf SamplingParams message
            stream: Whether streaming is enabled
            kv_transfer_params: KV transfer params proto for Mooncake PD
            model_version_tag: LMCache model-version tag to inject into
                ``extra_args["kv_transfer_params"]`` when multi-version KV is
                enabled (e.g. ``"3"`` for version 3).  ``None`` = no injection.

        Returns:
            vLLM SamplingParams with detokenize=False and structured_outputs
        """
        # Build stop sequences
        stop = list(params.stop) if params.stop else None
        stop_token_ids = list(params.stop_token_ids) if params.stop_token_ids else None

        # Handle structured outputs constraints
        structured_outputs = None
        constraint_field = params.WhichOneof("constraint")
        if constraint_field:
            if constraint_field == "json_schema":
                structured_outputs = StructuredOutputsParams(json=params.json_schema)
            elif constraint_field == "regex":
                structured_outputs = StructuredOutputsParams(regex=params.regex)
            elif constraint_field == "grammar":
                structured_outputs = StructuredOutputsParams(grammar=params.grammar)
            elif constraint_field == "structural_tag":
                structured_outputs = StructuredOutputsParams(structural_tag=params.structural_tag)
            elif constraint_field == "json_object":
                structured_outputs = StructuredOutputsParams(json_object=params.json_object)
            elif constraint_field == "choice":
                structured_outputs = StructuredOutputsParams(choice=list(params.choice.choices))

        # Build extra_args for kv_transfer_params (Mooncake PD)
        extra_args = None
        if kv_transfer_params:
            remote_host = kv_transfer_params.remote_host
            remote_port = kv_transfer_params.remote_port
            if not remote_host or not (1 <= remote_port <= 65535):
                raise ValueError(
                    "Invalid kv_transfer_params: remote_host must be set and remote_port must be in [1, 65535]."
                )
            logger.debug(
                "kv_transfer_params={remote_host=%s, remote_port=%d}",
                remote_host,
                remote_port,
            )
            extra_args = {
                "kv_transfer_params": {
                    "remote_host": remote_host,
                    "remote_port": remote_port,
                }
            }

        # Inject model-version tag for LMCache version-aware KV store/retrieve.
        if model_version_tag is not None:
            if extra_args is None:
                extra_args = {}
            extra_args.setdefault("kv_transfer_params", {})
            extra_args["kv_transfer_params"]["lmcache.tag.model_version"] = model_version_tag

        # Create SamplingParams
        # output_kind=DELTA: Return only new tokens in each chunk (for streaming)
        return SamplingParams(
            temperature=params.temperature if params.HasField("temperature") else 1.0,
            top_p=params.top_p if params.top_p != 0.0 else 1.0,
            top_k=params.top_k,
            min_p=params.min_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            repetition_penalty=params.repetition_penalty
            if params.repetition_penalty != 0.0
            else 1.0,
            max_tokens=params.max_tokens if params.HasField("max_tokens") else None,
            min_tokens=params.min_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            ignore_eos=params.ignore_eos,
            n=params.n if params.n > 0 else 1,
            logprobs=params.logprobs if params.HasField("logprobs") else None,
            prompt_logprobs=params.prompt_logprobs if params.HasField("prompt_logprobs") else None,
            seed=params.seed if params.HasField("seed") else None,
            include_stop_str_in_output=params.include_stop_str_in_output,
            logit_bias=dict(params.logit_bias) if params.logit_bias else None,
            structured_outputs=structured_outputs,
            extra_args=extra_args,
            # detokenize must be True if stop strings are used
            detokenize=bool(stop),
            output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
            routed_experts_prompt_start=params.routed_experts_prompt_start,
        )

    @staticmethod
    def _build_top_logprobs(
        logprob_entry: dict,
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.TopLogProbs:
        """Build TopLogProbs proto from a logprob entry dict."""
        top = vllm_engine_pb2.TopLogProbs()
        if num_top_logprobs and num_top_logprobs > 0 and logprob_entry:
            for tid, lp in itertools.islice(logprob_entry.items(), num_top_logprobs):
                top.token_ids.append(tid)
                top.values.append(lp.logprob)
        return top

    @staticmethod
    def _build_routed_experts_tensor(
        routed_experts: "np.ndarray | None",
        index: int = 0,
    ) -> "vllm_engine_pb2.RoutedExpertsTensor | None":
        """Convert vLLM's routed_experts ndarray into the proto tensor.

        vLLM populates ``CompletionOutput.routed_experts`` only on
        ``finished=True`` (including the abort path); it is an
        ``np.ndarray`` of shape ``[num_tokens, num_layers, top_k]`` with
        dtype ``uint8``/``uint16``, C-contiguous (built from
        ``np.concatenate(..., axis=0)`` over per-step chunks).

        We pass the raw bytes through a single ``.tobytes()`` call so the
        proto carries the layout exactly as the GPU produced it; the SMG
        gateway and downstream trainers reuse the same ``np.frombuffer``/
        ``np.load`` decoders without reshaping.  Returns ``None`` for
        engines started without ``--enable-return-routed-experts`` (the
        attribute is absent or ``None``) and for empty arrays.
        """
        if routed_experts is None:
            return None
        # Defensive: the engine should never hand us a 0-row array, but
        # guard anyway so the gateway sees a missing field rather than an
        # empty-but-present payload.
        if getattr(routed_experts, "size", 0) == 0:
            return None
        if routed_experts.ndim != 3:
            logger.warning(
                "routed_experts has unexpected ndim=%d; dropping",
                routed_experts.ndim,
            )
            return None
        if not routed_experts.flags["C_CONTIGUOUS"]:
            routed_experts = np.ascontiguousarray(routed_experts)
        _, num_layers, top_k = routed_experts.shape
        return vllm_engine_pb2.RoutedExpertsTensor(
            data=routed_experts.tobytes(),
            num_layers=int(num_layers),
            top_k=int(top_k),
            dtype=str(routed_experts.dtype),
            index=int(index),
        )

    @staticmethod
    def _build_output_logprobs(
        logprobs: SampleLogprobs | None,
        token_ids: list[int],
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.OutputLogProbs | None:
        """
        Convert vLLM SampleLogprobs to proto OutputLogProbs.

        Args:
            logprobs: vLLM logprobs (list of dict[int, Logprob])
            token_ids: Token IDs for each position
            num_top_logprobs: Number of top logprobs to include

        Returns:
            OutputLogProbs proto or None
        """
        if not logprobs:
            return None

        proto = vllm_engine_pb2.OutputLogProbs()

        for token_id, logprob_entry in zip(token_ids, logprobs):
            if logprob := logprob_entry.get(token_id):
                proto.token_logprobs.append(logprob.logprob)
                proto.token_ids.append(token_id)

                if num_top_logprobs:
                    proto.top_logprobs.append(
                        VllmEngineServicer._build_top_logprobs(logprob_entry, num_top_logprobs)
                    )

        return proto if proto.token_ids else None

    @staticmethod
    def _build_input_logprobs(
        prompt_logprobs: PromptLogprobs | None,
        prompt_token_ids: list[int],
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.InputLogProbs | None:
        """
        Convert vLLM PromptLogprobs to proto InputLogProbs.

        Args:
            prompt_logprobs: vLLM prompt logprobs (list of dict[int, Logprob] | None)
            prompt_token_ids: Prompt token IDs
            num_top_logprobs: Number of top logprobs to include

        Returns:
            InputLogProbs proto or None
        """
        if not prompt_logprobs:
            return None

        proto = vllm_engine_pb2.InputLogProbs()

        for token_id, logprob_entry in zip(prompt_token_ids, prompt_logprobs):
            token_logprob = vllm_engine_pb2.InputTokenLogProb()

            # First token has no logprob (None)
            if logprob_entry is not None and token_id in logprob_entry:
                token_logprob.value = logprob_entry[token_id].logprob

            proto.token_logprobs.append(token_logprob)
            proto.token_ids.append(token_id)
            if num_top_logprobs:
                proto.top_logprobs.append(
                    VllmEngineServicer._build_top_logprobs(logprob_entry, num_top_logprobs)
                )

        return proto if proto.token_ids else None

    @staticmethod
    def _tokenization_kwargs_from_proto(
        params: vllm_engine_pb2.SamplingParams,
    ) -> dict[str, int] | None:
        if params.HasField("truncate_prompt_tokens"):
            return {"truncate_prompt_tokens": params.truncate_prompt_tokens}
        return None

    @staticmethod
    def _chunk_response(
        output: RequestOutput,
        completion: "CompletionOutput | None" = None,
        num_logprobs: int | None = None,
        num_prompt_logprobs: int | None = None,
        is_first_chunk: bool = False,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a streaming chunk response from vLLM output.
        When output_kind=DELTA, vLLM returns only new tokens automatically.

        Note: This sends DELTA logprobs (only for new tokens in this chunk).
        The Rust side is responsible for accumulating if needed.

        Args:
            output: vLLM RequestOutput (with delta tokens when output_kind=DELTA)
            completion: Specific CompletionOutput to use (for n>1 support).
                       If None, uses output.outputs[0] for backwards compatibility.
            num_logprobs: Number of top logprobs for output tokens
            num_prompt_logprobs: Number of top logprobs for prompt tokens
            is_first_chunk: Whether this is the first chunk for this index
                           (include input_logprobs only on first chunk)

        Returns:
            GenerateResponse with chunk field set
        """
        # Use provided completion or fall back to first output
        if completion is None:
            completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty chunk
            return vllm_engine_pb2.GenerateResponse(
                chunk=vllm_engine_pb2.GenerateStreamChunk(
                    token_ids=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                    index=0,
                ),
            )

        # Build output logprobs for this chunk's tokens (delta, not cumulative)
        output_logprobs = VllmEngineServicer._build_output_logprobs(
            completion.logprobs, completion.token_ids, num_logprobs
        )

        # Build input logprobs only on first chunk for this index
        input_logprobs = None
        if is_first_chunk:
            input_logprobs = VllmEngineServicer._build_input_logprobs(
                output.prompt_logprobs,
                output.prompt_token_ids,
                num_prompt_logprobs,
            )

        # When output_kind=DELTA, completion.token_ids contains only new tokens
        # vLLM handles the delta logic internally
        # completion_tokens = delta count (client will accumulate)
        return vllm_engine_pb2.GenerateResponse(
            chunk=vllm_engine_pb2.GenerateStreamChunk(
                token_ids=completion.token_ids,
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                completion_tokens=len(completion.token_ids),  # Delta count
                cached_tokens=output.num_cached_tokens,
                output_logprobs=output_logprobs,
                input_logprobs=input_logprobs,
                index=completion.index,
            ),
        )

    @staticmethod
    def _complete_response(
        output: RequestOutput,
        completion: "CompletionOutput | None" = None,
        num_logprobs: int | None = None,
        num_prompt_logprobs: int | None = None,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a final completion response from vLLM output.

        For non-streaming (FINAL_ONLY): completion has all tokens and logprobs.
        For streaming (DELTA): completion has last delta; Rust accumulates.

        Args:
            output: vLLM RequestOutput (finished=True)
            completion: Specific CompletionOutput to use (for n>1 support).
                       If None, uses output.outputs[0] for backwards compatibility.
            num_logprobs: Number of top logprobs for output tokens
            num_prompt_logprobs: Number of top logprobs for prompt tokens

        Returns:
            GenerateResponse with complete field set
        """
        # Use provided completion or fall back to first output
        if completion is None:
            completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty completion
            return vllm_engine_pb2.GenerateResponse(
                complete=vllm_engine_pb2.GenerateComplete(
                    output_ids=[],
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                    index=0,
                ),
            )

        # Build output logprobs from completion's data
        # For non-streaming: this has all logprobs
        # For streaming: this has only last delta (Rust accumulates from chunks)
        output_logprobs = VllmEngineServicer._build_output_logprobs(
            completion.logprobs, completion.token_ids, num_logprobs
        )

        # Build input logprobs
        input_logprobs = VllmEngineServicer._build_input_logprobs(
            output.prompt_logprobs,
            output.prompt_token_ids,
            num_prompt_logprobs,
        )

        # Build kv_transfer_params if present (Mooncake PD)
        kv_transfer_params = None
        if output.kv_transfer_params:
            kv_transfer_params = vllm_engine_pb2.KvTransferParams(
                remote_host=output.kv_transfer_params.get("remote_host", ""),
                remote_port=output.kv_transfer_params.get("remote_port", 0),
            )

        # Build matched_stop kwargs from stop_reason (int token ID or str stop sequence)
        stop_kwargs = {}
        if completion.stop_reason is not None:
            if isinstance(completion.stop_reason, int):
                stop_kwargs["matched_token_id"] = completion.stop_reason
            else:
                stop_kwargs["matched_stop_str"] = str(completion.stop_reason)

        routed_experts_kwargs = {}
        re_tensor = VllmEngineServicer._build_routed_experts_tensor(
            getattr(completion, "routed_experts", None),
            index=completion.index,
        )
        if re_tensor is not None:
            routed_experts_kwargs["routed_experts"] = re_tensor

        # Build complete response
        # When streaming (DELTA mode): completion.token_ids will be empty/last delta
        # When non-streaming (FINAL_ONLY mode): completion.token_ids has all tokens
        # Client will accumulate token counts for streaming
        return vllm_engine_pb2.GenerateResponse(
            complete=vllm_engine_pb2.GenerateComplete(
                output_ids=completion.token_ids,
                finish_reason=completion.finish_reason or "stop",
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                completion_tokens=len(completion.token_ids),
                cached_tokens=output.num_cached_tokens,
                output_logprobs=output_logprobs,
                input_logprobs=input_logprobs,
                index=completion.index,
                kv_transfer_params=kv_transfer_params,
                **stop_kwargs,
                **routed_experts_kwargs,
            ),
        )
