//! Admission and selection by predicted inter-token latency.

use std::sync::Arc;

use super::{
    cost_model_utils::CostModel, get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::worker::Worker;

#[derive(Debug)]
pub struct ItlConfig {
    pub cost_model_path: String,
    pub max_concurrent_seqs_per_instance: usize,
    pub max_itl: Option<f64>,
}

#[derive(Debug)]
pub struct ItlPolicy {
    model: CostModel,
    cfg: ItlConfig,
}

impl ItlPolicy {
    pub fn with_config(cfg: ItlConfig) -> Result<Self, String> {
        if cfg.max_concurrent_seqs_per_instance == 0
            || cfg
                .max_itl
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(
                "ITL requires positive concurrency and a positive finite latency limit.".into(),
            );
        }
        let model =
            CostModel::load_from_file(&cfg.cost_model_path).map_err(|error| error.to_string())?;
        Ok(Self { model, cfg })
    }

    fn next_itl(&self, worker: &Arc<dyn Worker>, load: usize) -> Option<f64> {
        let metadata = worker.metadata();
        let tp = metadata.spec.labels.get("tp_size")?;
        let pp = metadata.spec.labels.get("pp_size")?;
        let entry = self.model.get(&format!("TP{tp}_PP{pp}"))?;
        // RM ITL models use request count with a zero token term.
        let itl = (entry
            .other_threshold
            .max(entry.other_latency_k * (load + 1) as f64)
            + entry.attn_latency_b)
            .max(1e-9);
        itl.is_finite().then_some(itl)
    }
}

impl LoadBalancingPolicy for ItlPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        get_healthy_worker_indices(workers)
            .into_iter()
            .filter_map(|index| {
                let load = workers[index].load();
                if load >= self.cfg.max_concurrent_seqs_per_instance {
                    return None;
                }
                let itl = self.next_itl(&workers[index], load)?;
                if self.cfg.max_itl.is_some_and(|limit| itl > limit) {
                    return None;
                }
                Some((index, itl, load))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)))
            .map(|candidate| candidate.0)
    }

    fn name(&self) -> &'static str {
        "itl"
    }

    fn needs_load_guard(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn policy(limit: usize, max_itl: Option<f64>) -> ItlPolicy {
        let model = CostModel::load_from_str(
            r#"{
            "TP1_PP1": {"other_threshold": 0.01, "other_latency_b": 1.0,
                "other_latency_k": 0.001, "attn_latency_b": 0.002, "attn_latency_k": 0.1},
            "TP2_PP1": {"other_threshold": 0.005, "other_latency_b": 1.0,
                "other_latency_k": 0.0005, "attn_latency_b": 0.001, "attn_latency_k": 0.1}
        }"#,
        )
        .unwrap();
        ItlPolicy {
            model,
            cfg: ItlConfig {
                cost_model_path: String::new(),
                max_concurrent_seqs_per_instance: limit,
                max_itl,
            },
        }
    }

    fn worker(tp: usize, load: usize) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://worker{tp}:8000"))
                .worker_type(WorkerType::Regular)
                .labels(HashMap::from([
                    ("tp_size".into(), tp.to_string()),
                    ("pp_size".into(), "1".into()),
                ]))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        for _ in 0..load {
            worker.increment_load();
        }
        worker
    }

    #[test]
    fn heterogeneous_worker_selected_by_next_itl() {
        let policy = policy(128, None);
        let slow = worker(1, 0);
        let fast = worker(2, 5);
        assert_eq!(policy.next_itl(&slow, 0), Some(0.012));
        assert_eq!(
            policy.select_worker(&[slow, fast], &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn threshold_ties_use_load_then_index() {
        let policy = policy(128, None);
        assert_eq!(
            policy.select_worker(&[worker(1, 3), worker(1, 1)], &SelectWorkerInfo::default()),
            Some(1)
        );
        assert_eq!(
            policy.select_worker(&[worker(1, 1), worker(1, 1)], &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn saturated_workers_wait_until_a_request_departs() {
        let policy = policy(2, None);
        let w = worker(1, 2);
        assert_eq!(
            policy.select_worker(&[w.clone()], &SelectWorkerInfo::default()),
            None
        );
        w.decrement_load();
        assert_eq!(
            policy.select_worker(&[w], &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn latency_limit_rejects_next_request() {
        assert_eq!(
            policy(128, Some(0.012)).select_worker(&[worker(1, 11)], &SelectWorkerInfo::default()),
            None
        );
        assert_eq!(
            policy(128, Some(0.012)).select_worker(&[worker(1, 0)], &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn unavailable_and_missing_model_workers_are_excluded() {
        let sleeping = worker(2, 0);
        sleeping.set_status(WorkerStatus::NotReady);
        assert_eq!(
            policy(128, None)
                .select_worker(&[sleeping, worker(4, 0)], &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn invalid_configuration_fails_before_routing() {
        assert!(ItlPolicy::with_config(ItlConfig {
            cost_model_path: String::new(),
            max_concurrent_seqs_per_instance: 0,
            max_itl: None,
        })
        .is_err());
        assert!(ItlPolicy::with_config(ItlConfig {
            cost_model_path: String::new(),
            max_concurrent_seqs_per_instance: 128,
            max_itl: None,
        })
        .is_err());
    }
}
