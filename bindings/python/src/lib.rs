use std::collections::HashMap;

use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use smg::*;
use smg_auth as auth;

// Define the enums with PyO3 bindings
#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    Passthrough,
    CacheAware,
    CacheAwareV1,
    LeastLoad,
    PowerOfTwo,
    Bucket,
    Manual,
    ConsistentHashing,
    PrefixHash,
    RequestNumBalance,
    ThroughputOptimal,
    ThroughputOptimalWithBudget,
}
#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum BackendType {
    Sglang,
    Openai,
    Anthropic,
}

#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum HistoryBackendType {
    Memory,
    None,
    Oracle,
    Postgres,
    Redis,
}

#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug, Default)]
pub enum PyRole {
    Admin,
    #[default]
    User,
}

impl PyRole {
    pub fn to_auth_role(&self) -> auth::Role {
        match self {
            PyRole::Admin => auth::Role::Admin,
            PyRole::User => auth::Role::User,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyApiKeyEntry {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub key: String,
    #[pyo3(get, set)]
    pub role: PyRole,
}

#[pymethods]
impl PyApiKeyEntry {
    #[new]
    #[pyo3(signature = (id, name, key, role = PyRole::User))]
    fn new(id: String, name: String, key: String, role: PyRole) -> Self {
        PyApiKeyEntry {
            id,
            name,
            key,
            role,
        }
    }
}

impl PyApiKeyEntry {
    pub fn to_auth_api_key_entry(&self) -> auth::ApiKeyEntry {
        auth::ApiKeyEntry::new(&self.id, &self.name, &self.key, self.role.to_auth_role())
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyJwtConfig {
    #[pyo3(get, set)]
    pub issuer: String,
    #[pyo3(get, set)]
    pub audience: String,
    #[pyo3(get, set)]
    pub jwks_uri: Option<String>,
    #[pyo3(get, set)]
    pub role_mapping: HashMap<String, String>,
}

#[pymethods]
impl PyJwtConfig {
    #[new]
    #[pyo3(signature = (
        issuer,
        audience,
        jwks_uri = None,
        role_mapping = HashMap::new(),
    ))]
    fn new(
        issuer: String,
        audience: String,
        jwks_uri: Option<String>,
        role_mapping: HashMap<String, String>,
    ) -> Self {
        PyJwtConfig {
            issuer,
            audience,
            jwks_uri,
            role_mapping,
        }
    }
}

impl PyJwtConfig {
    pub fn to_auth_jwt_config(&self) -> auth::JwtConfig {
        let mut config = auth::JwtConfig::new(&self.issuer, &self.audience);

        // Conditionally set JWKS URI
        if let Some(ref uri) = self.jwks_uri {
            config = config.with_jwks_uri(uri);
        }

        // Add role mappings
        for (idp_role, gateway_role) in &self.role_mapping {
            let role = match gateway_role.to_lowercase().as_str() {
                "admin" => auth::Role::Admin,
                _ => auth::Role::User,
            };
            config = config.with_role_mapping(idp_role, role);
        }

        config
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyControlPlaneAuthConfig {
    #[pyo3(get, set)]
    pub jwt: Option<PyJwtConfig>,
    #[pyo3(get, set)]
    pub api_keys: Vec<PyApiKeyEntry>,
    #[pyo3(get, set)]
    pub audit_enabled: bool,
}

#[pymethods]
impl PyControlPlaneAuthConfig {
    #[new]
    #[pyo3(signature = (
        jwt = None,
        api_keys = vec![],
        audit_enabled = true,
    ))]
    fn new(jwt: Option<PyJwtConfig>, api_keys: Vec<PyApiKeyEntry>, audit_enabled: bool) -> Self {
        PyControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled,
        }
    }
}

impl PyControlPlaneAuthConfig {
    pub fn to_auth_control_plane_config(&self) -> auth::ControlPlaneAuthConfig {
        auth::ControlPlaneAuthConfig {
            jwt: self.jwt.as_ref().map(|j| j.to_auth_jwt_config()),
            api_keys: self
                .api_keys
                .iter()
                .map(|k| k.to_auth_api_key_entry())
                .collect(),
            audit_enabled: self.audit_enabled,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyOracleConfig {
    #[pyo3(get, set)]
    pub wallet_path: Option<String>,
    #[pyo3(get, set)]
    pub connect_descriptor: Option<String>,
    #[pyo3(get, set)]
    pub external_auth: bool,
    #[pyo3(get, set)]
    pub username: Option<String>,
    #[pyo3(get, set)]
    pub password: Option<String>,
    #[pyo3(get, set)]
    pub pool_min: usize,
    #[pyo3(get, set)]
    pub pool_max: usize,
    #[pyo3(get, set)]
    pub pool_timeout_secs: u64,
}

impl std::fmt::Debug for PyOracleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyOracleConfig")
            .field("wallet_path", &self.wallet_path)
            .field("connect_descriptor", &"<redacted>")
            .field("external_auth", &self.external_auth)
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .field("pool_min", &self.pool_min)
            .field("pool_max", &self.pool_max)
            .field("pool_timeout_secs", &self.pool_timeout_secs)
            .finish()
    }
}

#[pymethods]
impl PyOracleConfig {
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        password = None,
        username = None,
        connect_descriptor = None,
        wallet_path = None,
        external_auth = false,
        pool_min = 1,
        pool_max = 16,
        pool_timeout_secs = 30,
    ))]
    fn new(
        password: Option<String>,
        username: Option<String>,
        connect_descriptor: Option<String>,
        wallet_path: Option<String>,
        external_auth: bool,
        pool_min: usize,
        pool_max: usize,
        pool_timeout_secs: u64,
    ) -> PyResult<Self> {
        if pool_min == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pool_min must be at least 1",
            ));
        }
        if pool_max < pool_min {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pool_max must be >= pool_min",
            ));
        }

        Ok(PyOracleConfig {
            wallet_path,
            connect_descriptor,
            external_auth,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
        })
    }
}

impl PyOracleConfig {
    pub fn to_config_oracle(&self) -> config::OracleConfig {
        config::OracleConfig {
            wallet_path: self.wallet_path.clone(),
            connect_descriptor: self.connect_descriptor.clone().unwrap_or_default(),
            external_auth: self.external_auth,
            username: self.username.clone().unwrap_or_default(),
            password: self.password.clone().unwrap_or_default(),
            pool_min: self.pool_min,
            pool_max: self.pool_max,
            pool_timeout_secs: self.pool_timeout_secs,
            schema: None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyRedisConfig {
    #[pyo3(get, set)]
    pub url: String,
    #[pyo3(get, set)]
    pub pool_max: usize,
    #[pyo3(get, set)]
    pub retention_days: Option<u64>,
}

#[pymethods]
impl PyRedisConfig {
    #[new]
    #[pyo3(signature = (url, pool_max = 16, retention_days = Some(30)))]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "PyO3 #[new] method signature requires PyResult"
    )]
    fn new(url: String, pool_max: usize, retention_days: Option<u64>) -> PyResult<Self> {
        Ok(PyRedisConfig {
            url,
            pool_max,
            retention_days,
        })
    }
}

impl PyRedisConfig {
    pub fn to_config_redis(&self) -> config::RedisConfig {
        config::RedisConfig {
            url: self.url.clone(),
            pool_max: self.pool_max,
            retention_days: self.retention_days,
            schema: None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPostgresConfig {
    #[pyo3(get, set)]
    pub db_url: Option<String>,

    #[pyo3(get, set)]
    pub pool_max: usize,
}

#[pymethods]
impl PyPostgresConfig {
    #[new]
    #[pyo3(signature = (db_url = None,pool_max = 16,))]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "PyO3 #[new] method signature requires PyResult"
    )]
    fn new(db_url: Option<String>, pool_max: usize) -> PyResult<Self> {
        Ok(PyPostgresConfig { db_url, pool_max })
    }
}

impl PyPostgresConfig {
    pub fn to_config_postgres(&self) -> config::PostgresConfig {
        config::PostgresConfig {
            db_url: self.db_url.clone().unwrap_or_default(),
            pool_max: self.pool_max,
            schema: None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
struct Router {
    host: String,
    port: u16,
    health_check_port: Option<u16>,
    routing_key_override: bool,
    worker_urls: Vec<String>,
    policy: PolicyType,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval: u64,
    load_monitor_interval: u64,
    engine_stats_staleness_threshold_ms: u64,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    block_size: usize,
    gpu_overlap_weight: f64,
    lmcache_overlap_weight: f64,
    balance_token_usage_threshold: f32,
    overload_token_usage_threshold: f32,
    max_concurrent_seqs_per_instance: usize,
    delta_throughput_threshold: f64,
    max_prompt_length: usize,
    request_budget: usize,
    max_num_waiting_reqs_after_preemption: usize,
    cost_model_path: Option<String>,
    max_idle_secs: u64,
    assignment_mode: String,
    max_payload_size: usize,
    dp_aware: bool,
    dp_minimum_tokens_scheduler: bool,
    api_key: Option<String>,
    log_dir: Option<String>,
    log_level: Option<String>,
    log_json: bool,
    service_discovery: bool,
    selector: HashMap<String, String>,
    service_discovery_port: u16,
    service_discovery_namespace: Option<String>,
    prefill_selector: HashMap<String, String>,
    decode_selector: HashMap<String, String>,
    router_selector: HashMap<String, String>,
    bootstrap_port_annotation: String,
    model_id_from: Option<String>,
    prometheus_port: Option<u16>,
    prometheus_host: Option<String>,
    prometheus_duration_buckets: Option<Vec<f64>>,
    request_timeout_secs: u64,
    shutdown_grace_period_secs: u64,
    request_id_headers: Option<Vec<String>>,
    trust_tenant_header: bool,
    tenant_header_name: String,
    storage_context_headers: HashMap<String, String>,
    pd_disaggregation: bool,
    bucket_adjust_interval_secs: usize,
    prefill_urls: Option<Vec<(String, Option<u16>)>>,
    decode_urls: Option<Vec<String>>,
    prefill_policy: Option<PolicyType>,
    decode_policy: Option<PolicyType>,
    max_concurrent_requests: i32,
    cors_allowed_origins: Vec<String>,
    retry_max_retries: u32,
    retry_initial_backoff_ms: u64,
    retry_max_backoff_ms: u64,
    retry_backoff_multiplier: f32,
    retry_jitter_factor: f32,
    disable_retries: bool,
    cb_failure_threshold: u32,
    cb_success_threshold: u32,
    cb_timeout_duration_secs: u64,
    cb_window_duration_secs: u64,
    disable_circuit_breaker: bool,
    health_failure_threshold: u32,
    health_success_threshold: u32,
    health_check_timeout_secs: u64,
    health_check_interval_secs: u64,
    health_check_endpoint: String,
    disable_health_check: bool,
    remove_unhealthy_workers: bool,
    enable_igw: bool,
    queue_size: usize,
    queue_timeout_secs: u64,
    rate_limit_tokens_per_second: Option<i32>,
    enable_routing_loop: bool,
    routing_loop_check_interval_ms: u64,
    routing_loop_request_sort_key: String,
    routing_loop_multi_priority_queue: bool,
    routing_loop_receive_batch_size: usize,
    routing_loop_dispatch_batch_size: usize,
    routing_loop_max_running_dispatch_tasks: usize,
    connection_mode: worker::ConnectionMode,
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    chat_template: Option<String>,
    disable_tokenizer_autoload: bool,
    tokenizer_cache_enable_l0: bool,
    tokenizer_cache_l0_max_entries: usize,
    tokenizer_cache_enable_l1: bool,
    tokenizer_cache_l1_max_memory: usize,
    reasoning_parser: Option<String>,
    tool_call_parser: Option<String>,
    mcp_config_path: Option<String>,
    storage_hook_wasm_path: Option<String>,
    backend: BackendType,
    history_backend: HistoryBackendType,
    oracle_config: Option<PyOracleConfig>,
    postgres_config: Option<PyPostgresConfig>,
    redis_config: Option<PyRedisConfig>,
    client_cert_path: Option<String>,
    client_key_path: Option<String>,
    ca_cert_paths: Vec<String>,
    server_cert_path: Option<String>,
    server_key_path: Option<String>,
    enable_trace: bool,
    otlp_traces_endpoint: String,
    control_plane_auth: Option<PyControlPlaneAuthConfig>,
    schema_config: Option<String>,
    // Mesh server
    enable_mesh: bool,
    mesh_server_name: Option<String>,
    mesh_host: String,
    mesh_advertise_host: Option<String>,
    mesh_port: u16,
    mesh_peer_urls: Vec<String>,
    worker_selection_strategy: String,
    // Config for PSRL worker selection strategy
    psrl_ps_manager_addr: String,
    psrl_candidate_sort_key: String,
    psrl_enable_group_sticky: bool,
    // KV-cache transfer when a request is re-routed to a different instance
    psrl_kv_transfer_enable: bool,
    psrl_kv_transfer_mode: String,
    psrl_kv_transfer_timeout_ms: u64,
    // TITO
    enable_tito: bool,
    tito_debug: bool,
    tito_gc_threshold: Option<usize>,
    /// New parameters MUST be appended here (not inserted mid-list) to avoid
    /// breaking external Python callers that pass `_Router(...)` positionally.
    drain_settle_secs: u64,
    enable_wasm: bool,
    encode_selector: HashMap<String, String>,
    epd_disaggregation: bool,
    encode_urls: Option<Vec<(String, Option<u16>)>>,
    encode_policy: Option<PolicyType>,
    multimodal_tensor_transport: Option<String>,
    multimodal_shm_min_bytes: Option<usize>,
    trajectory_id_strategy: String,
}

impl Router {
    fn determine_connection_mode(worker_urls: &[String]) -> worker::ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return worker::ConnectionMode::Grpc;
            }
        }
        worker::ConnectionMode::Http
    }

    fn parse_mesh_socket_addr(
        host: &str,
        port: u16,
        field: &str,
    ) -> PyResult<std::net::SocketAddr> {
        let addr = format!("{host}:{port}");
        addr.parse::<std::net::SocketAddr>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid value for {field}='{host}': invalid mesh socket address '{addr}': {e}"
            ))
        })
    }

    fn parse_assignment_mode(&self) -> Result<config::ManualAssignmentMode, config::ConfigError> {
        match self.assignment_mode.as_str() {
            "random" => Ok(config::ManualAssignmentMode::Random),
            "min_load" => Ok(config::ManualAssignmentMode::MinLoad),
            "min_group" => Ok(config::ManualAssignmentMode::MinGroup),
            other => Err(config::ConfigError::InvalidValue {
                field: "assignment_mode".to_string(),
                value: other.to_string(),
                reason: "expected 'random', 'min_load', or 'min_group'".to_string(),
            }),
        }
    }

    pub fn to_router_config(&self) -> config::ConfigResult<config::RouterConfig> {
        use config::{
            DiscoveryConfig, MetricsConfig, PolicyConfig as ConfigPolicyConfig, RoutingMode,
        };

        // Validate the transport mode up front. The CLI (value_parser) and the
        // argparse path (choices) already reject bad values; this covers direct
        // programmatic `RouterArgs` use, matching the CLI/Rust parsing contract.
        let multimodal_tensor_transport = self
            .multimodal_tensor_transport
            .as_deref()
            .map(|value| {
                config::TransportMode::parse(value).ok_or_else(|| {
                    config::ConfigError::InvalidValue {
                        field: "multimodal_tensor_transport".to_string(),
                        value: value.to_string(),
                        reason: "expected 'inline', 'shm', 'auto', or 'rdma'".to_string(),
                    }
                })
            })
            .transpose()?;

        let convert_policy = |policy: &PolicyType| -> config::ConfigResult<ConfigPolicyConfig> {
            Ok(match policy {
                PolicyType::Random => ConfigPolicyConfig::Random,
                PolicyType::RoundRobin => ConfigPolicyConfig::RoundRobin,
                PolicyType::Passthrough => ConfigPolicyConfig::Passthrough,
                PolicyType::CacheAware => ConfigPolicyConfig::CacheAware {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                    block_size: self.block_size,
                    gpu_overlap_weight: self.gpu_overlap_weight,
                    lmcache_overlap_weight: self.lmcache_overlap_weight,
                    balance_token_usage_threshold: self.balance_token_usage_threshold,
                    overload_token_usage_threshold: self.overload_token_usage_threshold,
                },
                PolicyType::CacheAwareV1 => ConfigPolicyConfig::CacheAwareV1 {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                    block_size: self.block_size,
                    gpu_overlap_weight: self.gpu_overlap_weight,
                    lmcache_overlap_weight: self.lmcache_overlap_weight,
                    balance_token_usage_threshold: self.balance_token_usage_threshold,
                    overload_token_usage_threshold: self.overload_token_usage_threshold,
                },
                PolicyType::LeastLoad => ConfigPolicyConfig::LeastLoad {
                    kv_pressure_weight: 0.5,
                    mean_prefill_tokens: 1024,
                    default_throughput: 100.0,
                    load_check_interval_secs: 5,
                },
                PolicyType::PowerOfTwo => ConfigPolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5,
                },
                PolicyType::Bucket => ConfigPolicyConfig::Bucket {
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    bucket_adjust_interval_secs: self.bucket_adjust_interval_secs,
                },
                PolicyType::Manual => ConfigPolicyConfig::Manual {
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_idle_secs: self.max_idle_secs,
                    assignment_mode: self.parse_assignment_mode()?,
                },
                PolicyType::ConsistentHashing => ConfigPolicyConfig::ConsistentHashing,
                PolicyType::PrefixHash => ConfigPolicyConfig::PrefixHash {
                    prefix_token_count: 256,
                    load_factor: 1.25,
                },
                PolicyType::RequestNumBalance => ConfigPolicyConfig::RequestNumBalance,
                PolicyType::ThroughputOptimal => ConfigPolicyConfig::ThroughputOptimal {
                    cost_model_path: self.cost_model_path.clone().unwrap_or_default(),
                    max_concurrent_seqs_per_instance: self.max_concurrent_seqs_per_instance,
                    delta_throughput_threshold: self.delta_throughput_threshold,
                    max_prompt_length: self.max_prompt_length,
                    request_budget: self.request_budget,
                    max_num_waiting_reqs_after_preemption: self
                        .max_num_waiting_reqs_after_preemption,
                },
                PolicyType::ThroughputOptimalWithBudget => {
                    ConfigPolicyConfig::ThroughputOptimalWithBudget {
                        cost_model_path: self.cost_model_path.clone().unwrap_or_default(),
                        max_concurrent_seqs_per_instance: self.max_concurrent_seqs_per_instance,
                        delta_throughput_threshold: self.delta_throughput_threshold,
                        max_prompt_length: self.max_prompt_length,
                        request_budget: self.request_budget,
                        max_num_waiting_reqs_after_preemption: self
                            .max_num_waiting_reqs_after_preemption,
                    }
                }
            })
        };

        let mode = if self.enable_igw {
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if matches!(self.backend, BackendType::Openai) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if matches!(self.backend, BackendType::Anthropic) {
            RoutingMode::Anthropic {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.epd_disaggregation {
            RoutingMode::EncodePrefillDecode {
                encode_urls: self.encode_urls.clone().unwrap_or_default(),
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                encode_policy: self
                    .encode_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
                prefill_policy: self
                    .prefill_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
                decode_policy: self
                    .decode_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
            }
        } else if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                prefill_policy: self
                    .prefill_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
                decode_policy: self
                    .decode_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let policy = convert_policy(&self.policy)?;

        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: self.selector.clone(),
                encode_selector: self.encode_selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
                router_selector: self.router_selector.clone(),
                router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                model_id_source: self.model_id_from.clone(),
            })
        } else {
            None
        };

        let metrics = match (self.prometheus_port, self.prometheus_host.as_ref()) {
            (Some(port), Some(host)) => Some(MetricsConfig {
                port,
                host: host.clone(),
            }),
            _ => None,
        };

        let trace_config = Some(config::TraceConfig {
            enable_trace: self.enable_trace,
            otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
        });

        let request_sort_key = match self.routing_loop_request_sort_key.as_str() {
            "short_length" => config::RequestSortKey::ShortLength,
            "long_length" => config::RequestSortKey::LongLength,
            "small_id" => config::RequestSortKey::SmallId,
            other => {
                return Err(config::ConfigError::InvalidValue {
                    field: "routing_loop_request_sort_key".to_string(),
                    value: other.to_string(),
                    reason: "expected 'short_length', 'long_length', or 'small_id'".to_string(),
                });
            }
        };

        let candidate_sort_key = match self.psrl_candidate_sort_key.as_str() {
            "reserve_capability" => config::CandidateSortKey::ReserveCapability,
            _ => config::CandidateSortKey::Version,
        };

        let kv_transfer_mode = match self.psrl_kv_transfer_mode.as_str() {
            "sync" => config::KvTransferMode::Sync,
            "pin_sync" => config::KvTransferMode::PinSync,
            _ => config::KvTransferMode::Async,
        };
        let kv_transfer_config = config::KvTransferConfig {
            enable: self.psrl_kv_transfer_enable,
            transfer_mode: kv_transfer_mode,
            transfer_timeout_ms: self.psrl_kv_transfer_timeout_ms,
            ..Default::default()
        };

        let worker_selection_strategy = match self.worker_selection_strategy.as_str() {
            "psrl" => config::WorkerSelectionStrategy::Psrl,
            _ => config::WorkerSelectionStrategy::Naive,
        };

        let history_backend = match self.history_backend {
            HistoryBackendType::Memory => config::HistoryBackend::Memory,
            HistoryBackendType::None => config::HistoryBackend::None,
            HistoryBackendType::Oracle => config::HistoryBackend::Oracle,
            HistoryBackendType::Postgres => config::HistoryBackend::Postgres,
            HistoryBackendType::Redis => config::HistoryBackend::Redis,
        };

        // Load schema config from YAML file if provided
        let schema = if let Some(ref path) = self.schema_config {
            let content = std::fs::read_to_string(path).map_err(|e| {
                config::ConfigError::ValidationFailed {
                    reason: format!("Failed to read schema config file '{path}': {e}"),
                }
            })?;
            let schema: config::SchemaConfig = serde_yaml::from_str(&content).map_err(|e| {
                config::ConfigError::ValidationFailed {
                    reason: format!("Failed to parse schema config file '{path}': {e}"),
                }
            })?;
            Some(schema)
        } else {
            None
        };

        let oracle = if matches!(self.history_backend, HistoryBackendType::Oracle) {
            self.oracle_config.as_ref().map(|cfg| {
                let mut c = cfg.to_config_oracle();
                c.schema.clone_from(&schema);
                c
            })
        } else {
            None
        };

        let postgres_config = if matches!(self.history_backend, HistoryBackendType::Postgres) {
            self.postgres_config.as_ref().map(|cfg| {
                let mut c = cfg.to_config_postgres();
                c.schema.clone_from(&schema);
                c
            })
        } else {
            None
        };

        let redis_config = if matches!(self.history_backend, HistoryBackendType::Redis) {
            self.redis_config.as_ref().map(|cfg| {
                let mut c = cfg.to_config_redis();
                c.schema = schema;
                c
            })
        } else {
            None
        };

        config::RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .host(&self.host)
            .port(self.port)
            .connection_mode(self.connection_mode)
            .max_payload_size(self.max_payload_size)
            .request_timeout_secs(self.request_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .load_monitor_interval_secs(self.load_monitor_interval)
            .engine_stats_staleness_threshold_ms(self.engine_stats_staleness_threshold_ms)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .enable_routing_loop(self.enable_routing_loop)
            .routing_loop_check_interval_ms(self.routing_loop_check_interval_ms)
            .routing_loop_request_sort_key(request_sort_key)
            .routing_loop_multi_priority_queue(self.routing_loop_multi_priority_queue)
            .routing_loop_receive_batch_size(self.routing_loop_receive_batch_size)
            .routing_loop_dispatch_batch_size(self.routing_loop_dispatch_batch_size)
            .routing_loop_max_running_dispatch_tasks(self.routing_loop_max_running_dispatch_tasks)
            .cors_allowed_origins(self.cors_allowed_origins.clone())
            .retry_config(config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(config::HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
                disable_health_check: self.disable_health_check,
                remove_unhealthy_workers: self.remove_unhealthy_workers,
                drain_settle_secs: 5,
            })
            .tokenizer_cache(config::TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .disable_tokenizer_autoload(self.disable_tokenizer_autoload)
            .history_backend(history_backend)
            .maybe_api_key(self.api_key.as_ref())
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_log_level(self.log_level.as_ref())
            .maybe_request_id_headers(self.request_id_headers.clone())
            .trust_tenant_header(self.trust_tenant_header)
            .tenant_header_name(&self.tenant_header_name)
            .maybe_storage_context_headers(
                (!self.storage_context_headers.is_empty())
                    .then(|| self.storage_context_headers.clone()),
            )
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .maybe_oracle(oracle)
            .maybe_postgres(postgres_config)
            .maybe_redis(redis_config)
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.mcp_config_path.as_ref())
            .maybe_storage_hook_wasm_path(self.storage_hook_wasm_path.as_deref())
            .dp_aware(self.dp_aware)
            .multimodal_tensor_transport(multimodal_tensor_transport)
            .multimodal_shm_min_bytes(self.multimodal_shm_min_bytes)
            .routing_key_override(config::RoutingKeyOverrideConfig {
                enabled: self.routing_key_override,
                eviction_interval_secs: self.eviction_interval_secs,
                max_idle_secs: self.max_idle_secs,
                assignment_mode: self.parse_assignment_mode()?,
            })
            .retries(!self.disable_retries)
            .circuit_breaker(!self.disable_circuit_breaker)
            .igw(self.enable_igw)
            .maybe_client_cert_and_key(
                self.client_cert_path.as_ref(),
                self.client_key_path.as_ref(),
            )
            .add_ca_certificates(self.ca_cert_paths.clone())
            .maybe_server_cert_and_key(
                self.server_cert_path.as_ref(),
                self.server_key_path.as_ref(),
            )
            .dp_minimum_tokens_scheduler(self.dp_minimum_tokens_scheduler)
            .worker_selection_strategy(worker_selection_strategy)
            .psrl_ps_manager_addr(&self.psrl_ps_manager_addr)
            .psrl_candidate_sort_key(candidate_sort_key)
            .psrl_enable_group_sticky(self.psrl_enable_group_sticky)
            .psrl_kv_transfer(kv_transfer_config)
            .build()
    }
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("0.0.0.0"),
        port = 3001,
        worker_startup_timeout_secs = 600,
        worker_startup_check_interval = 30,
        load_monitor_interval = 10,
        engine_stats_staleness_threshold_ms = 0,
        cache_threshold = 0.3,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        eviction_interval_secs = 120,
        max_tree_size = 2usize.pow(26),
        block_size = 16,
        gpu_overlap_weight = 1.0,
        lmcache_overlap_weight = 0.5,
        balance_token_usage_threshold = 1.0,
        overload_token_usage_threshold = 1.0,
        max_concurrent_seqs_per_instance = 100,
        delta_throughput_threshold = 0.5,
        max_prompt_length = 8192,
        request_budget = 1024,
        max_num_waiting_reqs_after_preemption = 1000,
        cost_model_path = None,
        max_idle_secs = 14400,
        assignment_mode = String::from("random"),
        max_payload_size = 512 * 1024 * 1024,
        dp_aware = false,
        dp_minimum_tokens_scheduler = false,
        api_key = None,
        log_dir = None,
        log_level = None,
        log_json = false,
        service_discovery = false,
        selector = HashMap::new(),
        service_discovery_port = 80,
        service_discovery_namespace = None,
        prefill_selector = HashMap::new(),
        decode_selector = HashMap::new(),
        router_selector = HashMap::new(),
        bootstrap_port_annotation = String::from("sglang.ai/bootstrap-port"),
        model_id_from = None,
        prometheus_port = None,
        prometheus_host = None,
        prometheus_duration_buckets = None,
        request_timeout_secs = 1800,
        shutdown_grace_period_secs = 180,
        request_id_headers = None,
        trust_tenant_header = false,
        tenant_header_name = String::from("x-smg-tenant-id"),
        storage_context_headers = HashMap::new(),
        pd_disaggregation = false,
        bucket_adjust_interval_secs = 5,
        prefill_urls = None,
        decode_urls = None,
        prefill_policy = None,
        decode_policy = None,
        max_concurrent_requests = -1,
        cors_allowed_origins = vec![],
        retry_max_retries = 5,
        retry_initial_backoff_ms = 50,
        retry_max_backoff_ms = 30_000,
        retry_backoff_multiplier = 1.5,
        retry_jitter_factor = 0.2,
        disable_retries = false,
        cb_failure_threshold = 10,
        cb_success_threshold = 3,
        cb_timeout_duration_secs = 60,
        cb_window_duration_secs = 120,
        disable_circuit_breaker = false,
        health_failure_threshold = 3,
        health_success_threshold = 2,
        health_check_timeout_secs = 5,
        health_check_interval_secs = 60,
        health_check_endpoint = String::from("/health"),
        disable_health_check = false,
        remove_unhealthy_workers = false,
        enable_igw = false,
        queue_size = 100,
        queue_timeout_secs = 60,
        rate_limit_tokens_per_second = None,
        enable_routing_loop = false,
        routing_loop_check_interval_ms = 10,
        routing_loop_request_sort_key = String::from("short_length"),
        routing_loop_multi_priority_queue = false,
        routing_loop_receive_batch_size = 1024,
        routing_loop_dispatch_batch_size = 1024,
        routing_loop_max_running_dispatch_tasks = 4096,
        connection_mode = None,
        model_path = None,
        tokenizer_path = None,
        chat_template = None,
        tokenizer_cache_enable_l0 = false,
        tokenizer_cache_l0_max_entries = 10000,
        tokenizer_cache_enable_l1 = false,
        tokenizer_cache_l1_max_memory = 52428800,
        reasoning_parser = None,
        tool_call_parser = None,
        mcp_config_path = None,
        storage_hook_wasm_path = None,
        backend = BackendType::Sglang,
        history_backend = HistoryBackendType::Memory,
        oracle_config = None,
        postgres_config = None,
        redis_config = None,
        client_cert_path = None,
        client_key_path = None,
        ca_cert_paths = vec![],
        server_cert_path = None,
        server_key_path = None,
        enable_trace = false,
        otlp_traces_endpoint = String::from("localhost:4317"),
        control_plane_auth = None,
        schema_config = None,
        disable_tokenizer_autoload = false,
        enable_mesh = false,
        mesh_server_name = None,
        mesh_host = String::from("0.0.0.0"),
        mesh_port = 39527u16,
        mesh_peer_urls = vec![],
        mesh_advertise_host = None,
        worker_selection_strategy = String::from("naive"),
        psrl_ps_manager_addr = String::new(),
        psrl_candidate_sort_key = String::from("version"),
        psrl_enable_group_sticky = false,
        psrl_kv_transfer_enable = false,
        psrl_kv_transfer_mode = String::from("async"),
        psrl_kv_transfer_timeout_ms = 30_000,
        enable_tito = false,
        tito_debug = false,
        tito_gc_threshold = None,
        drain_settle_secs = 5,
        enable_wasm = false,
        // Appended last (not inserted mid-list) so every pre-existing
        // positional argument keeps its index for callers that construct
        // `_Router(...)` positionally. See the struct-field note above.
        health_check_port = None,
        routing_key_override = false,
        encode_selector = HashMap::new(),
        epd_disaggregation = false,
        encode_urls = None,
        encode_policy = None,
        multimodal_tensor_transport = None,
        multimodal_shm_min_bytes = None,
        trajectory_id_strategy = String::from("manual"),
    ))]
    #[expect(clippy::too_many_arguments)]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "PyO3 #[new] method signature requires PyResult"
    )]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        load_monitor_interval: u64,
        engine_stats_staleness_threshold_ms: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        block_size: usize,
        gpu_overlap_weight: f64,
        lmcache_overlap_weight: f64,
        balance_token_usage_threshold: f32,
        overload_token_usage_threshold: f32,
        max_concurrent_seqs_per_instance: usize,
        delta_throughput_threshold: f64,
        max_prompt_length: usize,
        request_budget: usize,
        max_num_waiting_reqs_after_preemption: usize,
        cost_model_path: Option<String>,
        max_idle_secs: u64,
        assignment_mode: String,
        max_payload_size: usize,
        dp_aware: bool,
        dp_minimum_tokens_scheduler: bool,
        api_key: Option<String>,
        log_dir: Option<String>,
        log_level: Option<String>,
        log_json: bool,
        service_discovery: bool,
        selector: HashMap<String, String>,
        service_discovery_port: u16,
        service_discovery_namespace: Option<String>,
        prefill_selector: HashMap<String, String>,
        decode_selector: HashMap<String, String>,
        router_selector: HashMap<String, String>,
        bootstrap_port_annotation: String,
        model_id_from: Option<String>,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
        prometheus_duration_buckets: Option<Vec<f64>>,
        request_timeout_secs: u64,
        shutdown_grace_period_secs: u64,
        request_id_headers: Option<Vec<String>>,
        trust_tenant_header: bool,
        tenant_header_name: String,
        storage_context_headers: HashMap<String, String>,
        pd_disaggregation: bool,
        bucket_adjust_interval_secs: usize,
        prefill_urls: Option<Vec<(String, Option<u16>)>>,
        decode_urls: Option<Vec<String>>,
        prefill_policy: Option<PolicyType>,
        decode_policy: Option<PolicyType>,
        max_concurrent_requests: i32,
        cors_allowed_origins: Vec<String>,
        retry_max_retries: u32,
        retry_initial_backoff_ms: u64,
        retry_max_backoff_ms: u64,
        retry_backoff_multiplier: f32,
        retry_jitter_factor: f32,
        disable_retries: bool,
        cb_failure_threshold: u32,
        cb_success_threshold: u32,
        cb_timeout_duration_secs: u64,
        cb_window_duration_secs: u64,
        disable_circuit_breaker: bool,
        health_failure_threshold: u32,
        health_success_threshold: u32,
        health_check_timeout_secs: u64,
        health_check_interval_secs: u64,
        health_check_endpoint: String,
        disable_health_check: bool,
        remove_unhealthy_workers: bool,
        enable_igw: bool,
        queue_size: usize,
        queue_timeout_secs: u64,
        rate_limit_tokens_per_second: Option<i32>,
        enable_routing_loop: bool,
        routing_loop_check_interval_ms: u64,
        routing_loop_request_sort_key: String,
        routing_loop_multi_priority_queue: bool,
        routing_loop_receive_batch_size: usize,
        routing_loop_dispatch_batch_size: usize,
        routing_loop_max_running_dispatch_tasks: usize,
        connection_mode: Option<String>,
        model_path: Option<String>,
        tokenizer_path: Option<String>,
        chat_template: Option<String>,
        tokenizer_cache_enable_l0: bool,
        tokenizer_cache_l0_max_entries: usize,
        tokenizer_cache_enable_l1: bool,
        tokenizer_cache_l1_max_memory: usize,
        reasoning_parser: Option<String>,
        tool_call_parser: Option<String>,
        mcp_config_path: Option<String>,
        storage_hook_wasm_path: Option<String>,
        backend: BackendType,
        history_backend: HistoryBackendType,
        oracle_config: Option<PyOracleConfig>,
        postgres_config: Option<PyPostgresConfig>,
        redis_config: Option<PyRedisConfig>,
        client_cert_path: Option<String>,
        client_key_path: Option<String>,
        ca_cert_paths: Vec<String>,
        server_cert_path: Option<String>,
        server_key_path: Option<String>,
        enable_trace: bool,
        otlp_traces_endpoint: String,
        control_plane_auth: Option<PyControlPlaneAuthConfig>,
        schema_config: Option<String>,
        disable_tokenizer_autoload: bool,
        enable_mesh: bool,
        mesh_server_name: Option<String>,
        mesh_host: String,
        mesh_port: u16,
        mesh_peer_urls: Vec<String>,
        mesh_advertise_host: Option<String>,
        worker_selection_strategy: String,
        psrl_ps_manager_addr: String,
        psrl_candidate_sort_key: String,
        psrl_enable_group_sticky: bool,
        psrl_kv_transfer_enable: bool,
        psrl_kv_transfer_mode: String,
        psrl_kv_transfer_timeout_ms: u64,
        enable_tito: bool,
        tito_debug: bool,
        tito_gc_threshold: Option<usize>,
        drain_settle_secs: u64,
        enable_wasm: bool,
        // Appended last to match the `#[pyo3(signature)]` order above and
        // preserve positional-argument compatibility.
        health_check_port: Option<u16>,
        routing_key_override: bool,
        encode_selector: HashMap<String, String>,
        epd_disaggregation: bool,
        encode_urls: Option<Vec<(String, Option<u16>)>>,
        encode_policy: Option<PolicyType>,
        multimodal_tensor_transport: Option<String>,
        multimodal_shm_min_bytes: Option<usize>,
        trajectory_id_strategy: String,
    ) -> PyResult<Self> {
        let mut all_urls = worker_urls.clone();

        if let Some(ref encode_urls) = encode_urls {
            for (url, _) in encode_urls {
                all_urls.push(url.clone());
            }
        }

        if let Some(ref prefill_urls) = prefill_urls {
            for (url, _) in prefill_urls {
                all_urls.push(url.clone());
            }
        }

        if let Some(ref decode_urls) = decode_urls {
            all_urls.extend(decode_urls.clone());
        }

        let connection_mode = match connection_mode.as_deref() {
            Some("http") => worker::ConnectionMode::Http,
            Some("grpc") => worker::ConnectionMode::Grpc,
            _ => Self::determine_connection_mode(&all_urls),
        };

        if !matches!(trajectory_id_strategy.as_str(), "manual" | "auto") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid trajectory_id_strategy '{trajectory_id_strategy}'; expected manual or auto"
            )));
        }

        Ok(Router {
            host,
            port,
            health_check_port,
            routing_key_override,
            worker_urls,
            policy,
            worker_startup_timeout_secs,
            worker_startup_check_interval,
            load_monitor_interval,
            engine_stats_staleness_threshold_ms,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            block_size,
            gpu_overlap_weight,
            lmcache_overlap_weight,
            balance_token_usage_threshold,
            overload_token_usage_threshold,
            max_concurrent_seqs_per_instance,
            delta_throughput_threshold,
            max_prompt_length,
            request_budget,
            max_num_waiting_reqs_after_preemption,
            cost_model_path,
            max_idle_secs,
            assignment_mode,
            max_payload_size,
            dp_aware,
            dp_minimum_tokens_scheduler,
            api_key,
            log_dir,
            log_level,
            log_json,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prefill_selector,
            decode_selector,
            router_selector,
            bootstrap_port_annotation,
            model_id_from,
            prometheus_port,
            prometheus_host,
            prometheus_duration_buckets,
            request_timeout_secs,
            shutdown_grace_period_secs,
            request_id_headers,
            trust_tenant_header,
            tenant_header_name,
            storage_context_headers,
            pd_disaggregation,
            bucket_adjust_interval_secs,
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
            max_concurrent_requests,
            cors_allowed_origins,
            retry_max_retries,
            retry_initial_backoff_ms,
            retry_max_backoff_ms,
            retry_backoff_multiplier,
            retry_jitter_factor,
            disable_retries,
            cb_failure_threshold,
            cb_success_threshold,
            cb_timeout_duration_secs,
            cb_window_duration_secs,
            disable_circuit_breaker,
            health_failure_threshold,
            health_success_threshold,
            health_check_timeout_secs,
            health_check_interval_secs,
            health_check_endpoint,
            disable_health_check,
            remove_unhealthy_workers,
            enable_igw,
            queue_size,
            queue_timeout_secs,
            rate_limit_tokens_per_second,
            enable_routing_loop,
            routing_loop_check_interval_ms,
            routing_loop_request_sort_key,
            routing_loop_multi_priority_queue,
            routing_loop_receive_batch_size,
            routing_loop_dispatch_batch_size,
            routing_loop_max_running_dispatch_tasks,
            connection_mode,
            model_path,
            tokenizer_path,
            chat_template,
            disable_tokenizer_autoload,
            tokenizer_cache_enable_l0,
            tokenizer_cache_l0_max_entries,
            tokenizer_cache_enable_l1,
            tokenizer_cache_l1_max_memory,
            reasoning_parser,
            tool_call_parser,
            mcp_config_path,
            storage_hook_wasm_path,
            backend,
            history_backend,
            oracle_config,
            postgres_config,
            redis_config,
            client_cert_path,
            client_key_path,
            ca_cert_paths,
            server_cert_path,
            server_key_path,
            enable_trace,
            otlp_traces_endpoint,
            control_plane_auth,
            schema_config,
            enable_mesh,
            mesh_server_name,
            mesh_host,
            mesh_advertise_host,
            mesh_port,
            mesh_peer_urls,
            worker_selection_strategy,
            psrl_ps_manager_addr,
            psrl_candidate_sort_key,
            psrl_enable_group_sticky,
            psrl_kv_transfer_enable,
            psrl_kv_transfer_mode,
            psrl_kv_transfer_timeout_ms,
            enable_tito,
            tito_debug,
            tito_gc_threshold,
            drain_settle_secs,
            enable_wasm,
            encode_selector,
            epd_disaggregation,
            encode_urls,
            encode_policy,
            multimodal_tensor_transport,
            multimodal_shm_min_bytes,
            trajectory_id_strategy,
        })
    }

    fn start(&self) -> PyResult<()> {
        use observability::metrics::PrometheusConfig;

        let router_config = self.to_router_config().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {e}"))
        })?;

        router_config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration validation failed: {e}"))
        })?;

        let model_id_source = self
            .model_id_from
            .as_deref()
            .map(|s| {
                service_discovery::ModelIdSource::parse(s).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid --model-id-from value '{s}': {e}"
                    ))
                })
            })
            .transpose()?;

        let service_discovery_config = if self.service_discovery {
            Some(service_discovery::ServiceDiscoveryConfig {
                enabled: true,
                selector: self.selector.clone(),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                disaggregated_mode: self.pd_disaggregation || self.epd_disaggregation,
                encode_selector: self.encode_selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
                router_selector: self.router_selector.clone(),
                router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                model_id_source,
            })
        } else {
            None
        };

        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self
                .prometheus_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
            duration_buckets: self.prometheus_duration_buckets.clone(),
        });

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        runtime.block_on(async move {
            Box::pin(server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                router_config,
                max_payload_size: self.max_payload_size,
                log_dir: self.log_dir.clone(),
                log_level: self.log_level.clone(),
                log_json: self.log_json,
                service_discovery_config,
                prometheus_config,
                request_timeout_secs: self.request_timeout_secs,
                request_id_headers: self.request_id_headers.clone(),
                shutdown_grace_period_secs: self.shutdown_grace_period_secs,
                control_plane_auth: self
                    .control_plane_auth
                    .as_ref()
                    .map(|c| c.to_auth_control_plane_config()),
                mesh_server_config: if self.enable_mesh {
                    let self_name = self.mesh_server_name.clone().unwrap_or_else(|| {
                        use rand::{distr::Alphanumeric, RngExt};
                        let random_string: String = (0..4)
                            .map(|_| rand::rng().sample(Alphanumeric) as char)
                            .collect();
                        format!("Mesh_{random_string}")
                    });
                    let peer = self
                        .mesh_peer_urls
                        .first()
                        .map(|url| {
                            url.parse::<std::net::SocketAddr>().map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Invalid mesh peer URL '{url}': {e}"
                                ))
                            })
                        })
                        .transpose()?;
                    let bind_addr =
                        Self::parse_mesh_socket_addr(&self.mesh_host, self.mesh_port, "mesh_host")?;
                    let (advertise_host, advertise_field) =
                        if let Some(host) = self.mesh_advertise_host.as_deref() {
                            (host, "mesh_advertise_host")
                        } else {
                            (self.mesh_host.as_str(), "mesh_host")
                        };
                    let advertise_addr = Self::parse_mesh_socket_addr(
                        advertise_host,
                        self.mesh_port,
                        advertise_field,
                    )?;
                    if advertise_addr.ip().is_unspecified() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid value for {advertise_field}='{advertise_host}': mesh advertise address cannot be unspecified; set mesh_advertise_host to a routable node IP"
                        )));
                    }
                    Some(smg_mesh::MeshServerConfig {
                        self_name,
                        bind_addr,
                        advertise_addr,
                        init_peer: peer,
                        mtls_config: None,
                    })
                } else {
                    None
                },
                enable_tito: self.enable_tito,
                tito_debug: self.tito_debug,
                tito_gc_threshold: self.tito_gc_threshold,
                trajectory_id_strategy: self.trajectory_id_strategy.parse().map_err(|reason| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid trajectory_id_strategy: {reason}"
                    ))
                })?,
                webrtc_bind_addr: None,
                webrtc_stun_server: None,
                health_check_port: None,
                runtime_worker_threads: None,
            }))
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

/// Get simple version string (default for --version)
#[pyfunction]
fn get_version_string() -> String {
    version::get_version_string()
}

/// Get verbose version information string with full build details (for --version-verbose)
#[pyfunction]
fn get_verbose_version_string() -> String {
    version::get_verbose_version_string()
}

/// Print the startup banner with braille art and key configuration info.
#[pyfunction]
fn print_banner(host: &str, port: u16, mode: &str) {
    version::print_banner(host, port, mode);
}

/// Get the list of available tool call parsers from the Rust factory.
#[pyfunction]
fn get_available_tool_call_parsers() -> Vec<String> {
    static PARSERS: OnceCell<Vec<String>> = OnceCell::new();
    PARSERS
        .get_or_init(|| {
            let factory = tool_parser::ParserFactory::new();
            factory.list_parsers()
        })
        .clone()
}

/// Get the list of available reasoning parsers from the Rust factory.
#[pyfunction]
fn get_available_reasoning_parsers() -> Vec<String> {
    static PARSERS: OnceCell<Vec<String>> = OnceCell::new();
    PARSERS
        .get_or_init(|| {
            let factory = reasoning_parser::ParserFactory::new();
            factory.list_parsers()
        })
        .clone()
}

#[pymodule]
fn smg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<BackendType>()?;
    m.add_class::<HistoryBackendType>()?;
    m.add_class::<PyRole>()?;
    m.add_class::<PyApiKeyEntry>()?;
    m.add_class::<PyJwtConfig>()?;
    m.add_class::<PyControlPlaneAuthConfig>()?;
    m.add_class::<PyOracleConfig>()?;
    m.add_class::<PyPostgresConfig>()?;
    m.add_class::<PyRedisConfig>()?;
    m.add_class::<Router>()?;
    m.add_function(wrap_pyfunction!(get_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(get_verbose_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(print_banner, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_tool_call_parsers, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_reasoning_parsers, m)?)?;
    Ok(())
}
