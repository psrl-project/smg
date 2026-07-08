//! Unified gRPC client wrapper for SGLang, vLLM, and TensorRT-LLM backends

use std::collections::HashMap;

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{
    tokenizer_bundle, tokenizer_bundle::StreamBundle, MlxEngineClient,
    SglangGenerateRequestOptions, SglangSchedulerClient, TokenSpeedSchedulerClient,
    TrtllmServiceClient, VllmEngineClient,
};

use crate::routers::grpc::{
    proto_wrapper::{
        cleanup_tokenspeed_shm_handles, collect_tokenspeed_generate_request_shm_handles,
        finish_tokenspeed_request, ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest,
        ProtoStream,
    },
    MultimodalData,
};

/// Health check response (common across backends)
#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

/// Polymorphic gRPC client that wraps SGLang, vLLM, TensorRT-LLM, or MLX
#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
    Trtllm(TrtllmServiceClient),
    Mlx(MlxEngineClient),
    TokenSpeed(TokenSpeedSchedulerClient),
}

#[derive(Default)]
pub struct GenerateRequestBuildOptions {
    pub multimodal_inputs: Option<MultimodalData>,
    pub tool_constraints: Option<(String, String)>,
    pub require_reasoning: bool,
}

impl GrpcClient {
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm_mut(&mut self) -> &mut VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm_mut(&mut self) -> &mut TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_mlx() check"
    )]
    pub fn as_mlx(&self) -> &MlxEngineClient {
        match self {
            Self::Mlx(client) => client,
            _ => panic!("Expected MLX client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_mlx() check"
    )]
    pub fn as_mlx_mut(&mut self) -> &mut MlxEngineClient {
        match self {
            Self::Mlx(client) => client,
            _ => panic!("Expected MLX client"),
        }
    }

    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    pub async fn connect(
        url: &str,
        runtime_type: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match runtime_type {
            "sglang" => Ok(Self::Sglang(SglangSchedulerClient::connect(url).await?)),
            "vllm" => Ok(Self::Vllm(VllmEngineClient::connect(url).await?)),
            "trtllm" | "tensorrt-llm" => Ok(Self::Trtllm(TrtllmServiceClient::connect(url).await?)),
            "mlx" => Ok(Self::Mlx(MlxEngineClient::connect(url).await?)),
            "tokenspeed" => Ok(Self::TokenSpeed(
                TokenSpeedSchedulerClient::connect(url).await?,
            )),
            _ => Err(format!("Unknown runtime type: {runtime_type}").into()),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Vllm(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Trtllm(client) => {
                let resp = client.health_check().await?;
                let healthy = resp.status.to_lowercase().contains("ok")
                    || resp.status.to_lowercase().contains("healthy");
                Ok(HealthCheckResponse {
                    healthy,
                    message: resp.status,
                })
            }
            Self::Mlx(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::TokenSpeed(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Sglang(client) => Ok(ModelInfo::Sglang(Box::new(client.get_model_info().await?))),
            Self::Vllm(client) => Ok(ModelInfo::Vllm(client.get_model_info().await?)),
            Self::Trtllm(client) => Ok(ModelInfo::Trtllm(client.get_model_info().await?)),
            Self::Mlx(client) => Ok(ModelInfo::Mlx(client.get_model_info().await?)),
            Self::TokenSpeed(client) => Ok(ModelInfo::TokenSpeed(Box::new(
                client.get_model_info().await?,
            ))),
        }
    }

    /// Get the full load response from the backend.
    /// Only supported for SGLang backends. Returns per-DP-rank load metrics.
    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        // Optional sections beyond `core` (disagg/queues/memory) are dropped by
        // engines that do not report them, so requesting them is always safe and
        // leaves routing consumers, which only read `core`, unaffected.
        let include = || {
            ["core", "disagg", "queues", "memory"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        };
        match self {
            Self::Sglang(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            Self::TokenSpeed(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            Self::Vllm(client) => {
                let resp = client.get_loads(include()).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            _ => Err(tonic::Status::unimplemented(
                "GetLoads RPC not supported for this backend",
            )),
        }
    }

    /// Subscribe to KV cache events (all backends).
    ///
    /// `dp_rank` selects the data-parallel rank when the backend fronts
    /// multiple DP ranks behind one gRPC server. `None` → rank 0.
    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
        dp_rank: Option<u32>,
    ) -> Result<tonic::Streaming<smg_grpc_client::common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Sglang(client) => client.subscribe_kv_events(start_seq, dp_rank).await,
            Self::Vllm(client) => client.subscribe_kv_events(start_seq, dp_rank).await,
            Self::Trtllm(client) => client.subscribe_kv_events(start_seq, dp_rank).await,
            Self::TokenSpeed(client) => client.subscribe_kv_events(start_seq, dp_rank).await,
            Self::Mlx(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents RPC not supported for MLX backend",
            )),
        }
    }

    /// Transfer a trajectory's cached KV prefix to a destination instance.
    /// Only supported by the vLLM backend (PSRL cross-instance migration).
    pub async fn transfer_kv(
        &self,
        request: smg_grpc_client::vllm_proto::TransferKvRequest,
    ) -> Result<smg_grpc_client::vllm_proto::TransferKvResponse, tonic::Status> {
        match self {
            Self::Vllm(client) => client.transfer_kv(request).await,
            _ => Err(tonic::Status::unimplemented(
                "TransferKv RPC only supported for vLLM backend",
            )),
        }
    }

    /// Pin a trajectory's cached prefix (vLLM backend only).
    pub async fn pin_kv(
        &self,
        request: smg_grpc_client::vllm_proto::PinKvRequest,
    ) -> Result<smg_grpc_client::vllm_proto::PinKvResponse, tonic::Status> {
        match self {
            Self::Vllm(client) => client.pin_kv(request).await,
            _ => Err(tonic::Status::unimplemented(
                "PinKv RPC only supported for vLLM backend",
            )),
        }
    }

    /// Unpin a trajectory's cached prefix (vLLM backend only).
    pub async fn unpin_kv(
        &self,
        request: smg_grpc_client::vllm_proto::UnpinKvRequest,
    ) -> Result<smg_grpc_client::vllm_proto::UnpinKvResponse, tonic::Status> {
        match self {
            Self::Vllm(client) => client.unpin_kv(request).await,
            _ => Err(tonic::Status::unimplemented(
                "UnpinKv RPC only supported for vLLM backend",
            )),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Sglang(client) => Ok(ServerInfo::Sglang(Box::new(
                client.get_server_info().await?,
            ))),
            Self::Vllm(client) => Ok(ServerInfo::Vllm(client.get_server_info().await?)),
            Self::Trtllm(client) => Ok(ServerInfo::Trtllm(client.get_server_info().await?)),
            Self::Mlx(client) => Ok(ServerInfo::Mlx(client.get_server_info().await?)),
            Self::TokenSpeed(client) => Ok(ServerInfo::TokenSpeed(Box::new(
                client.get_server_info().await?,
            ))),
        }
    }

    /// Fetch tokenizer bundle from backend runtime and validate integrity/safety.
    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        let bundle = match self {
            Self::Sglang(client) => client.get_tokenizer().await,
            Self::Vllm(client) => client.get_tokenizer().await,
            Self::Trtllm(client) => client.get_tokenizer().await,
            Self::Mlx(client) => client.get_tokenizer().await,
            Self::TokenSpeed(_) => {
                return Err("GetTokenizer RPC not supported for TokenSpeed backend".into());
            }
        }?;

        tokenizer_bundle::validate_bundle_sha256(&bundle).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Tokenizer bundle SHA256 validation failed: {e}"),
            )
        })?;

        Ok(bundle)
    }

    /// Generate streaming response from request
    ///
    /// Dispatches to the appropriate backend client and wraps the result in ProtoStream.
    /// Returns `tonic::Status` on error so callers can inspect the gRPC status code directly.
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoGenerateRequest::Sglang(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Sglang(stream))
            }
            (Self::Vllm(client), ProtoGenerateRequest::Vllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Vllm(stream))
            }
            (Self::Trtllm(client), ProtoGenerateRequest::Trtllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Trtllm(stream))
            }
            (Self::Mlx(client), ProtoGenerateRequest::Mlx(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Mlx(stream))
            }
            (Self::TokenSpeed(client), ProtoGenerateRequest::TokenSpeed(boxed_req)) => {
                let shm_handles = collect_tokenspeed_generate_request_shm_handles(&boxed_req);
                match client.generate(*boxed_req).await {
                    Ok(stream) => Ok(ProtoStream::TokenSpeed(stream)),
                    Err(error) => {
                        cleanup_tokenspeed_shm_handles(&shm_handles);
                        Err(error)
                    }
                }
            }
            #[expect(
                clippy::panic,
                reason = "client and request types are always matched by construction in the pipeline"
            )]
            _ => panic!("Mismatched client and request types"),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoEmbedRequest::Sglang(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedComplete::Sglang(resp))
            }
            (Self::Vllm(client), ProtoEmbedRequest::Vllm(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedComplete::Vllm(resp))
            }
            (Self::Mlx(_), _) => Err(tonic::Status::unimplemented(
                "MLX backend does not support embedding",
            )),
            #[expect(
                clippy::panic,
                reason = "client and request types are always matched by construction in the pipeline"
            )]
            _ => panic!("Mismatched client and request types or unsupported embedding backend"),
        }
    }

    #[expect(
        clippy::unreachable,
        reason = "assembly stage guarantees matching MultimodalData variant for each backend"
    )]
    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let sglang_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Sglang(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    SglangGenerateRequestOptions {
                        multimodal_inputs: sglang_mm,
                        tool_call_constraint: options.tool_constraints,
                        require_reasoning: options.require_reasoning,
                    },
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let vllm_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Vllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    vllm_mm,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let trtllm_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Trtllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    trtllm_mm,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            // MLX: caller stage rejects multimodal before reaching this path.
            Self::Mlx(client) => {
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let tokenspeed_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::TokenSpeed(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                finish_tokenspeed_request(tokenspeed_mm, |mm| {
                    client.build_generate_request_from_chat(
                        request_id,
                        body,
                        processed_text,
                        token_ids,
                        mm,
                        options.tool_constraints,
                    )
                })
            }
        }
    }

    #[expect(
        clippy::unreachable,
        reason = "assembly stage guarantees matching MultimodalData variant for each backend"
    )]
    pub fn build_messages_request(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let sglang_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Sglang(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    SglangGenerateRequestOptions {
                        multimodal_inputs: sglang_mm,
                        tool_call_constraint: options.tool_constraints,
                        require_reasoning: options.require_reasoning,
                    },
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let vllm_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Vllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    vllm_mm,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let trtllm_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Trtllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    trtllm_mm,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            // MLX: caller stage rejects multimodal before reaching this path.
            Self::Mlx(client) => {
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    options.tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let tokenspeed_mm = options.multimodal_inputs.map(|mm| match mm {
                    MultimodalData::TokenSpeed(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                finish_tokenspeed_request(tokenspeed_mm, |mm| {
                    client.build_generate_request_from_messages(
                        request_id,
                        body,
                        processed_text,
                        token_ids,
                        mm,
                        options.tool_constraints,
                    )
                })
            }
        }
    }

    pub fn build_completion_request(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            Self::Mlx(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
            }
        }
    }

    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<MultimodalData>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let sglang_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Sglang(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                    sglang_mm,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let vllm_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Vllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                    vllm_mm,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let trtllm_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Trtllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                    trtllm_mm,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            Self::Mlx(client) => {
                if multimodal_inputs.is_some() {
                    return Err("MLX backend does not support multimodal inputs".to_string());
                }
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let tokenspeed_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::TokenSpeed(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                finish_tokenspeed_request(tokenspeed_mm, |mm| {
                    client.build_plain_generate_request(
                        request_id,
                        body,
                        original_text,
                        token_ids,
                        mm,
                    )
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata wrappers
// ---------------------------------------------------------------------------

pub enum ModelInfo {
    Sglang(Box<smg_grpc_client::sglang_proto::GetModelInfoResponse>),
    Vllm(smg_grpc_client::vllm_proto::GetModelInfoResponse),
    Trtllm(smg_grpc_client::trtllm_proto::GetModelInfoResponse),
    Mlx(smg_grpc_client::mlx_proto::GetModelInfoResponse),
    TokenSpeed(Box<smg_grpc_client::tokenspeed_proto::GetModelInfoResponse>),
}

pub enum ServerInfo {
    Sglang(Box<smg_grpc_client::sglang_proto::GetServerInfoResponse>),
    Vllm(smg_grpc_client::vllm_proto::GetServerInfoResponse),
    Trtllm(smg_grpc_client::trtllm_proto::GetServerInfoResponse),
    Mlx(smg_grpc_client::mlx_proto::GetServerInfoResponse),
    TokenSpeed(Box<smg_grpc_client::tokenspeed_proto::GetServerInfoResponse>),
}

impl ModelInfo {
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ModelInfo::Sglang(info) => flat_labels(info),
            ModelInfo::Vllm(info) => flat_labels(info),
            ModelInfo::Trtllm(info) => flat_labels(info),
            ModelInfo::Mlx(info) => flat_labels(info),
            ModelInfo::TokenSpeed(info) => flat_labels(info),
        }
    }
}

impl ServerInfo {
    /// Convert to labels. SGLang needs special handling because its `server_args`
    /// is a `prost_types::Struct` (not Serialize). vLLM/TRT-LLM are plain structs.
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ServerInfo::Sglang(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, SGLANG_GRPC_KEYS);
                }
                if !info.sglang_version.is_empty() {
                    labels.insert("version".to_string(), info.sglang_version.clone());
                }
                labels
            }
            ServerInfo::Vllm(info) => flat_labels(info),
            ServerInfo::Trtllm(info) => flat_labels(info),
            ServerInfo::Mlx(info) => flat_labels(info),
            ServerInfo::TokenSpeed(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, TOKENSPEED_GRPC_KEYS);
                }
                if !info.tokenspeed_version.is_empty() {
                    labels.insert("version".to_string(), info.tokenspeed_version.clone());
                }
                // Carry the worker's /dev/shm namespace identity (advertised in
                // scheduler_info). The router compares it to its own to decide the
                // SHM tensor transport by *verifying* a shared /dev/shm rather than
                // inferring it from the worker URL. See `worker_shares_dev_shm`.
                if let Some(ref sched) = info.scheduler_info {
                    pick_prost_fields(&mut labels, sched, &["shm_namespace_id"]);
                }
                labels
            }
        }
    }
}

/// Keys worth extracting from SGLang gRPC `server_args` (which contains the full config).
const SGLANG_GRPC_KEYS: &[&str] = &[
    "model_path",
    "served_model_name",
    "tokenizer_path",
    "tp_size",
    "dp_size",
    "pp_size",
    "context_length",
    "max_total_tokens",
    "max_running_requests",
    "load_balance_method",
    "disaggregation_mode",
    "is_embedding",
    "vocab_size",
    "weight_version",
];

const TOKENSPEED_GRPC_KEYS: &[&str] = &[
    "model_path",
    "served_model_name",
    "tokenizer_path",
    "tp_size",
    "dp_size",
    "pp_size",
    "context_length",
    "max_total_tokens",
    "max_running_requests",
    "is_embedding",
    "vocab_size",
    "weight_version",
];

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

/// Serialize to flat label map, skipping nulls/zeros/empty.
///
/// Booleans are emitted as `"true"` / `"false"` so downstream consumers
/// (e.g. `is_generation == "false"` for embedding detection) work correctly.
pub(crate) fn flat_labels<T: serde::Serialize>(value: &T) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(value) {
        for (key, val) in obj {
            match val {
                serde_json::Value::String(s) if !s.is_empty() && s != "null" => {
                    labels.insert(key, s);
                }
                serde_json::Value::Number(n) if n.as_f64().is_some_and(|v| v != 0.0) => {
                    // Format integers without decimal point
                    let formatted = n
                        .as_i64()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| n.to_string());
                    labels.insert(key, formatted);
                }
                serde_json::Value::Bool(b) => {
                    labels.insert(key, b.to_string());
                }
                serde_json::Value::Array(arr) if !arr.is_empty() => {
                    if let Ok(s) = serde_json::to_string(&arr) {
                        labels.insert(key, s);
                    }
                }
                _ => {}
            }
        }
    }
    labels
}

/// Pick specific keys from a `prost_types::Struct`.
fn pick_prost_fields(labels: &mut HashMap<String, String>, s: &prost_types::Struct, keys: &[&str]) {
    for key in keys {
        if let Some(val) = s.fields.get(*key) {
            if let Some(ref kind) = val.kind {
                match kind {
                    prost_types::value::Kind::StringValue(s) if !s.is_empty() && s != "null" => {
                        labels.insert((*key).to_string(), s.clone());
                    }
                    prost_types::value::Kind::NumberValue(n) if *n != 0.0 => {
                        let formatted = if *n == (*n as i64) as f64 {
                            (*n as i64).to_string()
                        } else {
                            n.to_string()
                        };
                        labels.insert((*key).to_string(), formatted);
                    }
                    prost_types::value::Kind::BoolValue(b) => {
                        labels.insert((*key).to_string(), b.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
}
