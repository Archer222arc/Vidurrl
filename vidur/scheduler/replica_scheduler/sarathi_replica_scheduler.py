from math import ceil

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
def dump_request(r):
    fields = [
            "id",
            "arrived_at",
            "num_prefill_tokens",
            "num_processed_tokens",
            "is_prefill_complete",
            "completed",
            # 如果你的类里真有，就会打印出来；没有就显示None
            "num_decode_tokens",
            "priority",
        ]
    kv = ", ".join(f"{k}={getattr(r, k, None)}" for k in fields)
    print("    " + kv)


class SarathiReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._num_running_batches = 0
        self._preempted_requests = []
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,
            self._config.chunk_size - num_batch_tokens,
        )

        next_num_tokens = max(0, next_num_tokens)

        return next_num_tokens


    def _get_next_batch(self) -> Batch:
        # 输出当前副本状态
        '''print(f"[Replica {self._replica_id}] 调度前状态：")
        print(f"  请求队列长度: {len(self._request_queue)}")
        print(f"  已分配blocks: {self._num_allocated_blocks}/{self._config.num_blocks}")
        print(f"  预占用请求数: {len(self._preempted_requests)}")
        print(f"  allocation_map: {self._allocation_map}")
        
        print(f"[Replica {self._replica_id}] 队列明细（len={len(self._request_queue)}）:")
        for req in self._request_queue:
            dump_request(req)
        # 预占用明细
        print(f"[Replica {self._replica_id}] _preempted_requests（len={len(self._preempted_requests)}）:")
        for r in self._preempted_requests:
            dump_request(r)
        # 调度前打印吞吐与延迟（确保已通过 set_runtime_context 注入了 _metric_store/_current_time）
        if getattr(self, "_metric_store", None) is not None:
            avg_latency = self._metric_store.get_average_latency()
            throughput  = self._metric_store.get_throughput(self._current_time)  # 若已在 MetricsStore 维护 current_time，也可不传参
            print(f"[Scheduler] 当前吞吐量: {throughput} req/s, 平均延迟: {avg_latency} s")
        else:
            print("[Scheduler] 无 metric_store 上下文（未注入）")'''

        requests = []
        num_tokens = []
        skipped_requests = []
        running_prefills = []
        contains_prefill = False
        num_batch_tokens = 0

        # preempted requests could contain multiple requests which have
        # partial prefills completed, so we need to be careful
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # re-add the skipped requests, but make sure that we add them to the
        # front of the queue so that they are scheduled first and we maintain FIFO ordering
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )
        skipped_requests = []

        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            if not self._can_allocate_request(self._request_queue[0]):
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)

            # all new requests will have a prefill
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
