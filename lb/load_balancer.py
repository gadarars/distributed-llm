# lb/load_balancer.py
# Distributes incoming requests across GPU workers using different strategies.
# Supports Round Robin, Least Connections, and Load-Aware routing.

import random
import threading
from enum import Enum
from common.models  import Request, Response, RequestStatus
from common.logger  import get_logger
from workers.gpu_worker import GPUWorker

log = get_logger("LoadBalancer")

MAX_RETRIES = 5


class Strategy(Enum):
    ROUND_ROBIN       = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE        = "load_aware"


class LoadBalancer:

    def __init__(self, workers: list[GPUWorker], strategy: Strategy = Strategy.ROUND_ROBIN):
        self.workers  = workers
        self.strategy = strategy
        self._rr_index = 0
        self._lock     = threading.Lock()

    # public API

    def get_next_worker(self) -> GPUWorker:
        # returns the next worker based on current strategy (used by scheduler/tests)
        alive = self._alive_workers()
        if not alive:
            raise RuntimeError("No alive workers available.")
        return self._select(alive)

    def dispatch(self, request: Request) -> Response:
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            alive = self._alive_workers()
            if not alive:
                log.critical("ALL workers are DEAD -- system unavailable.")
                return self._error_response(request, "All workers are dead.")

            worker = self._select(alive)
            log.info(
                "Strategy=%s -> Worker-%d (gpu=%.0f%%, conn=%d, attempt=%d/%d)",
                self.strategy.value, worker.worker_id,
                worker.gpu_utilization, worker.active_connections,
                attempt, MAX_RETRIES,
            )

            try:
                return worker.process(request)
            except RuntimeError as exc:
                last_exc = exc
                log.warning(
                    "Worker-%d failed on req=%s -- retrying on another node...",
                    worker.worker_id, request.request_id,
                )

        log.error("Request %s exhausted all %d retry attempts.",
                  request.request_id, MAX_RETRIES)
        return self._error_response(request, str(last_exc))

    def set_strategy(self, strategy: Strategy):
        self.strategy = strategy
        log.info("Load-balancing strategy switched to: %s", strategy.value)

    def alive_count(self) -> int:
        return len(self._alive_workers())

    def total_count(self) -> int:
        return len(self.workers)

    # strategy implementations

    def _select(self, alive: list[GPUWorker]) -> GPUWorker:
        if self.strategy == Strategy.ROUND_ROBIN:
            return self._round_robin(alive)
        if self.strategy == Strategy.LEAST_CONNECTIONS:
            return self._least_connections(alive)
        if self.strategy == Strategy.LOAD_AWARE:
            return self._load_aware(alive)
        return self._round_robin(alive)

    def _round_robin(self, alive: list[GPUWorker]) -> GPUWorker:
        with self._lock:
            self._rr_index %= len(alive)
            chosen = alive[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(alive)
        return chosen

    def _least_connections(self, alive: list[GPUWorker]) -> GPUWorker:
        # FIX: fair random tie-breaking instead of always picking the first minimum
        with self._lock:
            min_val = min(w.active_connections for w in alive)
            tied    = [w for w in alive if w.active_connections == min_val]
        return random.choice(tied)

    def _load_aware(self, alive: list[GPUWorker]) -> GPUWorker:
        # FIX: fair random tie-breaking instead of always picking the first minimum
        with self._lock:
            min_val = min(w.gpu_utilization for w in alive)
            tied    = [w for w in alive if w.gpu_utilization == min_val]
        return random.choice(tied)

    # helpers

    def _alive_workers(self) -> list[GPUWorker]:
        return [w for w in self.workers if w.is_alive]

    @staticmethod
    def _error_response(request: Request, msg: str) -> Response:
        return Response(
            request_id    = request.request_id,
            result        = "",
            status        = RequestStatus.FAILED,
            worker_id     = -1,
            latency_ms    = 0.0,
            error_message = msg,
        )
