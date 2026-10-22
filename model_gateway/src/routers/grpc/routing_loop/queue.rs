//! Priority request queue for routing-loop dispatch.

#![expect(
    dead_code,
    reason = "routing loop integration lands after queue/metadata"
)]

use std::collections::{BTreeMap, HashMap};

use crate::config::RequestSortKey;

/// Trait implemented by routing-loop queue entries.
///
/// Lower priority tuples are dispatched first.  Validation requests therefore
/// map to priority bucket `0`, while normal traffic maps to `1`.
pub(crate) trait RequestPriority {
    fn version_tag(&self) -> i64 {
        -1
    }

    fn is_validation(&self) -> bool {
        false
    }

    fn input_len(&self) -> usize {
        0
    }

    fn request_id(&self) -> Option<i64> {
        None
    }

    fn partition_key(&self) -> i32 {
        partition_key_for_version_tag(self.version_tag())
    }

    fn priority(&self, request_sort_key: RequestSortKey) -> (i64, i64, i64) {
        let validation_priority = i64::from(!self.is_validation());
        let version_priority = priority_for_version_tag(self.version_tag());
        let final_priority = match request_sort_key {
            RequestSortKey::ShortLength => self.input_len().min(i32::MAX as usize) as i64,
            RequestSortKey::LongLength => -(self.input_len().min(i32::MAX as usize) as i64),
            RequestSortKey::SmallId => self.request_id().unwrap_or(i64::MAX),
        };
        (validation_priority, version_priority, final_priority)
    }
}

/// Map unversioned or invalid-negative requests to the lowest scheduling class.
pub(crate) fn priority_for_version_tag(version_tag: i64) -> i64 {
    if version_tag < 0 {
        i64::MAX
    } else {
        version_tag
    }
}

/// Partition key for versioned multi-queue mode.
pub(crate) fn partition_key_for_version_tag(version_tag: i64) -> i32 {
    if version_tag < 0 {
        i32::MAX
    } else if version_tag > i64::from(i32::MAX - 1) {
        i32::MAX - 1
    } else {
        version_tag as i32
    }
}

struct RequestQueueEntry<T> {
    priority: (i64, i64, i64),
    seq: u64,
    request: T,
}

impl<T> Ord for RequestQueueEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl<T> PartialOrd for RequestQueueEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> PartialEq for RequestQueueEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl<T> Eq for RequestQueueEntry<T> {}

/// Single unbounded priority queue with FIFO tie-breaking for equal priorities.
pub(crate) struct PriorityRequestQueue<T: RequestPriority> {
    heap: std::collections::BinaryHeap<RequestQueueEntry<T>>,
    next_seq: u64,
    request_sort_key: RequestSortKey,
}

impl<T: RequestPriority> PriorityRequestQueue<T> {
    pub(crate) fn new(request_sort_key: RequestSortKey) -> Self {
        Self {
            heap: std::collections::BinaryHeap::new(),
            next_seq: 0,
            request_sort_key,
        }
    }

    pub(crate) fn push(&mut self, request: T) {
        let entry = RequestQueueEntry {
            priority: request.priority(self.request_sort_key),
            seq: self.next_seq,
            request,
        };
        self.next_seq = self.next_seq.wrapping_add(1);
        self.heap.push(entry);
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|entry| entry.request)
    }

    pub(crate) fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|entry| &entry.request)
    }

    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub(crate) fn request_sort_key(&self) -> RequestSortKey {
        self.request_sort_key
    }

    pub(crate) fn iter_requests(&self) -> impl Iterator<Item = &T> {
        self.heap.iter().map(|entry| &entry.request)
    }

    fn ordered_requests(&self) -> Vec<&T> {
        let mut entries: Vec<_> = self.heap.iter().collect();
        entries.sort_by(|a, b| b.cmp(a));
        entries.into_iter().map(|entry| &entry.request).collect()
    }

    fn pop_matching(&mut self, predicate: &impl Fn(&T) -> bool) -> Option<T> {
        let mut held = Vec::new();
        let mut selected = None;
        while let Some(entry) = self.heap.pop() {
            if predicate(&entry.request) {
                selected = Some(entry.request);
                break;
            }
            held.push(entry);
        }
        self.heap.extend(held);
        selected
    }
}

/// Version-partitioned unbounded priority queue.
///
/// In multi-priority mode, partitions are scanned in sorted key order on every
/// pop. This intentionally allows lower version partitions to suppress higher
/// version partitions while low-version work is present.
pub(crate) struct MultiPriorityRequestQueue<T: RequestPriority> {
    queues: BTreeMap<i32, PriorityRequestQueue<T>>,
    request_sort_key: RequestSortKey,
    enable_multi_priority_queue: bool,
}

impl<T: RequestPriority> MultiPriorityRequestQueue<T> {
    pub(crate) fn new(request_sort_key: RequestSortKey, enable_multi_priority_queue: bool) -> Self {
        Self {
            queues: BTreeMap::new(),
            request_sort_key,
            enable_multi_priority_queue,
        }
    }

    pub(crate) fn push(&mut self, request: T) {
        let partition = self.partition_for(&request);
        self.push_back_to_partition(partition, request);
    }

    pub(crate) fn push_back_to_partition(&mut self, partition: i32, request: T) {
        self.queues
            .entry(partition)
            .or_insert_with(|| PriorityRequestQueue::new(self.request_sort_key))
            .push(request);
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        for queue in self.queues.values_mut() {
            if let Some(request) = queue.pop() {
                return Some(request);
            }
        }
        None
    }

    pub(crate) fn pop_from_partition(&mut self, partition: i32) -> Option<T> {
        self.queues.get_mut(&partition)?.pop()
    }

    pub(crate) fn peek(&self) -> Option<&T> {
        for queue in self.queues.values() {
            if let Some(request) = queue.peek() {
                return Some(request);
            }
        }
        None
    }

    pub(crate) fn len(&self) -> usize {
        self.queues.values().map(PriorityRequestQueue::len).sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.queues.values().all(PriorityRequestQueue::is_empty)
    }

    pub(crate) fn queue_size(&self, partition: i32) -> usize {
        self.queues
            .get(&partition)
            .map_or(0, PriorityRequestQueue::len)
    }

    pub(crate) fn queue_keys(&self) -> Vec<i32> {
        self.queues
            .iter()
            .filter_map(|(&key, queue)| (!queue.is_empty()).then_some(key))
            .collect()
    }

    pub(crate) fn per_partition_sizes(&self) -> HashMap<i32, usize> {
        self.queues
            .iter()
            .filter_map(|(&key, queue)| {
                let len = queue.len();
                (len > 0).then_some((key, len))
            })
            .collect()
    }

    pub(crate) fn remove_empty_partitions(&mut self) {
        self.queues.retain(|_, queue| !queue.is_empty());
    }

    pub(crate) fn iter_requests(&self) -> impl Iterator<Item = &T> {
        self.queues
            .iter()
            .flat_map(|(_, queue)| queue.iter_requests())
    }

    pub(crate) fn ordered_requests(&self) -> Vec<&T> {
        self.queues.values().flat_map(|queue| queue.ordered_requests()).collect()
    }

    pub(crate) fn pop_matching(&mut self, predicate: impl Fn(&T) -> bool) -> Option<T> {
        self.queues.values_mut().find_map(|queue| queue.pop_matching(&predicate))
    }

    fn partition_for(&self, request: &T) -> i32 {
        if self.enable_multi_priority_queue {
            if request.is_validation() {
                i32::MIN
            } else {
                request.partition_key()
            }
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestReq {
        id: i64,
        version: i64,
        len: usize,
        validate: bool,
    }

    impl TestReq {
        fn new(id: i64, version: i64, len: usize) -> Self {
            Self {
                id,
                version,
                len,
                validate: false,
            }
        }
    }

    impl RequestPriority for TestReq {
        fn version_tag(&self) -> i64 {
            self.version
        }

        fn is_validation(&self) -> bool {
            self.validate
        }

        fn input_len(&self) -> usize {
            self.len
        }

        fn request_id(&self) -> Option<i64> {
            Some(self.id)
        }
    }

    #[test]
    fn snapshot_follows_dispatch_order_without_consuming_requests() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::ShortLength, true);
        q.push(TestReq::new(1, 2, 1));
        q.push(TestReq::new(2, 1, 8));
        q.push(TestReq::new(3, 1, 2));
        q.push(TestReq::new(4, 1, 2));
        assert_eq!(q.ordered_requests().iter().map(|r| r.id).collect::<Vec<_>>(), vec![3, 4, 2, 1]);
        assert_eq!(q.len(), 4);
        let mut actual = Vec::new();
        while let Some(request) = q.pop() {
            actual.push(request.id);
        }
        assert_eq!(actual, vec![3, 4, 2, 1]);
    }

    #[test]
    fn validation_precedes_older_training_partitions() {
        for multi_queue in [false, true] {
            let mut queue = MultiPriorityRequestQueue::new(RequestSortKey::ShortLength, multi_queue);
            queue.push(TestReq::new(1, 1, 1));
            let mut validation = TestReq::new(2, 5, 100);
            validation.validate = true;
            queue.push(validation);
            assert_eq!(queue.ordered_requests()[0].id, 2);
            assert_eq!(queue.peek().map(|request| request.id), Some(2));
            assert_eq!(queue.pop().map(|request| request.id), Some(2));
            assert_eq!(queue.pop().map(|request| request.id), Some(1));
        }
    }

    #[test]
    fn exclusive_migration_pop_preserves_other_requests_and_fifo_ties() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::ShortLength, true);
        q.push(TestReq::new(1, 0, 2));
        q.push(TestReq::new(2, 0, 2));
        q.push(TestReq::new(3, 1, 1));
        assert_eq!(q.pop_matching(|r| r.id == 3).map(|r| r.id), Some(3));
        assert_eq!(q.pop_matching(|r| r.id == 99), None);
        assert_eq!(q.pop().map(|r| r.id), Some(1));
        assert_eq!(q.pop().map(|r| r.id), Some(2));
    }

    #[test]
    fn priority_queue_orders_by_short_length_with_fifo_tiebreak() {
        let mut q = PriorityRequestQueue::new(RequestSortKey::ShortLength);
        q.push(TestReq::new(1, 0, 10));
        q.push(TestReq::new(2, 0, 2));
        q.push(TestReq::new(3, 0, 2));

        assert_eq!(q.pop().map(|req| req.id), Some(2));
        assert_eq!(q.pop().map(|req| req.id), Some(3));
        assert_eq!(q.pop().map(|req| req.id), Some(1));
    }

    #[test]
    fn priority_queue_orders_validate_before_normal() {
        let mut q = PriorityRequestQueue::new(RequestSortKey::ShortLength);
        q.push(TestReq::new(1, 0, 1));
        let mut validate = TestReq::new(2, 0, 100);
        validate.validate = true;
        q.push(validate);

        assert_eq!(q.pop().map(|req| req.id), Some(2));
        assert_eq!(q.pop().map(|req| req.id), Some(1));
    }

    #[test]
    fn priority_queue_supports_long_length_and_small_id() {
        let mut long = PriorityRequestQueue::new(RequestSortKey::LongLength);
        long.push(TestReq::new(1, 0, 1));
        long.push(TestReq::new(2, 0, 5));
        assert_eq!(long.pop().map(|req| req.id), Some(2));

        let mut id = PriorityRequestQueue::new(RequestSortKey::SmallId);
        id.push(TestReq::new(20, 0, 1));
        id.push(TestReq::new(3, 0, 100));
        assert_eq!(id.pop().map(|req| req.id), Some(3));
    }

    #[test]
    fn multi_queue_partitions_by_version_and_maps_unversioned_last() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::SmallId, true);
        q.push(TestReq::new(1, 2, 1));
        q.push(TestReq::new(2, -1, 1));
        q.push(TestReq::new(3, 1, 1));

        assert_eq!(q.queue_keys(), vec![1, 2, i32::MAX]);
        assert_eq!(q.queue_size(1), 1);
        assert_eq!(q.queue_size(i32::MAX), 1);
    }

    #[test]
    fn multi_queue_strictly_drains_lower_partition_first() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::SmallId, true);
        q.push(TestReq::new(1, 1, 1));
        q.push(TestReq::new(2, 1, 1));
        q.push(TestReq::new(3, 2, 1));
        q.push(TestReq::new(4, 2, 1));

        let ids = [
            q.pop().map(|req| req.id),
            q.pop().map(|req| req.id),
            q.pop().map(|req| req.id),
            q.pop().map(|req| req.id),
        ];
        assert_eq!(ids, [Some(1), Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn multi_queue_single_queue_mode_ignores_version_partitions() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::SmallId, false);
        q.push(TestReq::new(2, 2, 1));
        q.push(TestReq::new(1, 1, 1));

        assert_eq!(q.queue_keys(), vec![0]);
        assert_eq!(q.pop().map(|req| req.id), Some(1));
        assert_eq!(q.pop().map(|req| req.id), Some(2));
    }

    #[test]
    fn multi_queue_can_push_back_to_original_partition() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::SmallId, true);
        q.push_back_to_partition(7, TestReq::new(1, 99, 1));

        assert_eq!(q.queue_keys(), vec![7]);
        assert_eq!(q.pop_from_partition(7).map(|req| req.id), Some(1));
        assert!(q.pop_from_partition(7).is_none());
    }

    #[test]
    fn multi_queue_len_and_per_partition_sizes_ignore_empty_partitions() {
        let mut q = MultiPriorityRequestQueue::new(RequestSortKey::SmallId, true);
        q.push(TestReq::new(1, 1, 1));
        q.push(TestReq::new(2, 2, 1));
        q.push(TestReq::new(3, 2, 1));

        assert_eq!(q.len(), 3);
        assert_eq!(q.pop_from_partition(1).map(|req| req.id), Some(1));
        assert_eq!(q.queue_keys(), vec![2]);

        let sizes = q.per_partition_sizes();
        assert_eq!(sizes.get(&1), None);
        assert_eq!(sizes.get(&2), Some(&2));
    }
}
