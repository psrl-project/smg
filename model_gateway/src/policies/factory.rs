//! Factory for creating load balancing policies

use std::sync::Arc;

use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, CacheAwareV1Policy,
    ConsistentHashingPolicy, LeastLoadPolicy, LoadBalancingPolicy, ManualConfig, ManualPolicy,
    PassthroughPolicy, PowerOfTwoPolicy, PrefixHashConfig, PrefixHashPolicy, RandomPolicy,
    RequestNumBalancePolicy, RoundRobinPolicy, ThroughputOptimalConfig, ThroughputOptimalPolicy,
    ThroughputOptimalWithBudgetPolicy,
};
use crate::config::{ConfigError, ConfigResult, PolicyConfig};

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration
    pub fn create_from_config(config: &PolicyConfig) -> ConfigResult<Arc<dyn LoadBalancingPolicy>> {
        match config {
            PolicyConfig::Random => Ok(Arc::new(RandomPolicy::new())),
            PolicyConfig::RoundRobin => Ok(Arc::new(RoundRobinPolicy::new())),
            PolicyConfig::Passthrough => Ok(Arc::new(PassthroughPolicy::new())),
            PolicyConfig::PowerOfTwo { .. } => Ok(Arc::new(PowerOfTwoPolicy::new())),
            PolicyConfig::LeastLoad {
                kv_pressure_weight,
                mean_prefill_tokens,
                default_throughput,
                ..
            } => Ok(Arc::new(LeastLoadPolicy::with_params(
                *kv_pressure_weight,
                *mean_prefill_tokens,
                *default_throughput,
            ))),
            PolicyConfig::CacheAware {
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
                enable_kv_admission_control,
                kv_capacity_threshold,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                    block_size: *block_size,
                    gpu_overlap_weight: *gpu_overlap_weight,
                    lmcache_overlap_weight: *lmcache_overlap_weight,
                    balance_token_usage_threshold: *balance_token_usage_threshold,
                    overload_token_usage_threshold: *overload_token_usage_threshold,
                    enable_kv_admission_control: *enable_kv_admission_control,
                    kv_capacity_threshold: *kv_capacity_threshold,
                };
                Ok(Arc::new(CacheAwarePolicy::with_config(config)))
            }
            PolicyConfig::CacheAwareV1 {
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
                enable_kv_admission_control,
                kv_capacity_threshold,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                    block_size: *block_size,
                    gpu_overlap_weight: *gpu_overlap_weight,
                    lmcache_overlap_weight: *lmcache_overlap_weight,
                    balance_token_usage_threshold: *balance_token_usage_threshold,
                    overload_token_usage_threshold: *overload_token_usage_threshold,
                    enable_kv_admission_control: *enable_kv_admission_control,
                    kv_capacity_threshold: *kv_capacity_threshold,
                };
                Ok(Arc::new(CacheAwareV1Policy::with_config(config)))
            }
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                let config = BucketConfig {
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    bucket_adjust_interval_secs: *bucket_adjust_interval_secs,
                };
                Ok(Arc::new(BucketPolicy::with_config(config)))
            }
            PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode,
            } => {
                let config = ManualConfig {
                    eviction_interval_secs: *eviction_interval_secs,
                    max_idle_secs: *max_idle_secs,
                    assignment_mode: *assignment_mode,
                };
                Ok(Arc::new(ManualPolicy::with_config(config)))
            }
            PolicyConfig::ConsistentHashing => Ok(Arc::new(ConsistentHashingPolicy::new())),
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                let config = PrefixHashConfig {
                    prefix_token_count: *prefix_token_count,
                    load_factor: *load_factor,
                };
                Ok(Arc::new(PrefixHashPolicy::new(config)))
            }
            PolicyConfig::RequestNumBalance => Ok(Arc::new(RequestNumBalancePolicy::new())),
            PolicyConfig::ThroughputOptimal {
                cost_model_path,
                max_concurrent_seqs_per_instance,
                delta_throughput_threshold,
                max_prompt_length,
                request_budget,
                max_num_waiting_reqs_after_preemption,
            } => {
                let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig {
                    cost_model_path: cost_model_path.clone(),
                    max_concurrent_seqs_per_instance: *max_concurrent_seqs_per_instance,
                    delta_throughput_threshold: *delta_throughput_threshold,
                    max_prompt_length: *max_prompt_length,
                    request_budget: *request_budget,
                    max_num_waiting_reqs_after_preemption: *max_num_waiting_reqs_after_preemption,
                })
                .map_err(|e| ConfigError::ValidationFailed { reason: e })?;
                Ok(Arc::new(policy))
            }
            PolicyConfig::ThroughputOptimalWithBudget {
                request_budget,
                cost_model_path,
                max_concurrent_seqs_per_instance,
                delta_throughput_threshold,
                max_prompt_length,
                max_num_waiting_reqs_after_preemption,
            } => {
                // `request_budget` is the KV-cache page granularity used by the WithBudget
                // variant; it maps to `request_budget` in the shared config.
                let policy =
                    ThroughputOptimalWithBudgetPolicy::with_config(ThroughputOptimalConfig {
                        request_budget: *request_budget,
                        cost_model_path: cost_model_path.clone(),
                        max_concurrent_seqs_per_instance: *max_concurrent_seqs_per_instance,
                        delta_throughput_threshold: *delta_throughput_threshold,
                        max_prompt_length: *max_prompt_length,
                        max_num_waiting_reqs_after_preemption:
                            *max_num_waiting_reqs_after_preemption,
                    })
                    .map_err(|e| ConfigError::ValidationFailed { reason: e })?;
                Ok(Arc::new(policy))
            }
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "passthrough" => Some(Arc::new(PassthroughPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "least_load" | "leastload" => Some(Arc::new(LeastLoadPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "cache_aware_v1" | "cacheawarev1" => Some(Arc::new(CacheAwareV1Policy::new())),
            "bucket" => Some(Arc::new(BucketPolicy::new())),
            "manual" => Some(Arc::new(ManualPolicy::new())),
            "consistent_hashing" | "consistenthashing" => {
                Some(Arc::new(ConsistentHashingPolicy::new()))
            }
            "prefix_hash" | "prefixhash" => Some(Arc::new(PrefixHashPolicy::with_defaults())),
            "request_num_balance" | "requestnumbalance" => {
                Some(Arc::new(RequestNumBalancePolicy::new()))
            }
            "throughput_optimal" | "throughputoptimal" => {
                tracing::error!(
                    "ThroughputOptimal requires a valid cost model path, use create_from_config instead."
                );
                None
            }
            "throughput_optimal_with_budget" | "throughputoptimalwithbudget" => {
                tracing::error!(
                    "ThroughputOptimalWithBudget requires a valid cost model path, use create_from_config instead."
                );
                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random).unwrap();
        assert_eq!(policy.name(), "random");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin).unwrap();
        assert_eq!(policy.name(), "round_robin");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Passthrough).unwrap();
        assert_eq!(policy.name(), "passthrough");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        })
        .unwrap();
        assert_eq!(policy.name(), "power_of_two");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
            block_size: 16,
            gpu_overlap_weight: 1.0,
            lmcache_overlap_weight: 0.5,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
        enable_kv_admission_control: false,
                kv_capacity_threshold: 1.0,
        })
        .unwrap();
        assert_eq!(policy.name(), "cache_aware");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Bucket {
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        })
        .unwrap();
        assert_eq!(policy.name(), "bucket");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Default::default(),
        })
        .unwrap();
        assert_eq!(policy.name(), "manual");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing).unwrap();
        assert_eq!(policy.name(), "consistent_hashing");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RequestNumBalance).unwrap();
        assert_eq!(policy.name(), "request_num_balance");

        let result = PolicyFactory::create_from_config(&PolicyConfig::ThroughputOptimal {
            cost_model_path: "/nonexistent/cost_model.json".to_string(),
            max_concurrent_seqs_per_instance: 100,
            delta_throughput_threshold: 0.5,
            max_prompt_length: 8192,
            request_budget: 1024,
            max_num_waiting_reqs_after_preemption: 1000,
        });
        assert!(result.is_err(), "should fail with invalid cost model path");

        let result =
            PolicyFactory::create_from_config(&PolicyConfig::ThroughputOptimalWithBudget {
                request_budget: 1024,
                cost_model_path: "/nonexistent/cost_model.json".to_string(),
                max_concurrent_seqs_per_instance: 100,
                delta_throughput_threshold: 0.5,
                max_prompt_length: 8192,
                max_num_waiting_reqs_after_preemption: 1000,
            });
        assert!(result.is_err(), "should fail with invalid cost model path");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PrefixHash {
            prefix_token_count: 100,
            load_factor: 0.8,
        })
        .unwrap();
        assert_eq!(policy.name(), "prefix_hash");
    }

    #[tokio::test]
    async fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert_eq!(
            PolicyFactory::create_by_name("passthrough").unwrap().name(),
            "passthrough"
        );
        assert!(PolicyFactory::create_by_name("PASSTHROUGH").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("bucket").is_some());
        assert!(PolicyFactory::create_by_name("Bucket").is_some());
        assert!(PolicyFactory::create_by_name("manual").is_some());
        assert!(PolicyFactory::create_by_name("Manual").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hashing").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHashing").is_some());
        assert!(PolicyFactory::create_by_name("request_num_balance").is_some());
        assert!(PolicyFactory::create_by_name("RequestNumBalance").is_some());
        assert!(PolicyFactory::create_by_name("throughput_optimal").is_none());
        assert!(PolicyFactory::create_by_name("ThroughputOptimal").is_none());
        assert!(PolicyFactory::create_by_name("throughput_optimal_with_budget").is_none());
        assert!(PolicyFactory::create_by_name("ThroughputOptimalWithBudget").is_none());
        assert!(PolicyFactory::create_by_name("prefix_hash").is_some());
        assert!(PolicyFactory::create_by_name("PrefixHash").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }
}
