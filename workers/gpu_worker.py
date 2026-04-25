# workers/gpu_worker.py
# Simulates a single GPU worker node. Handles LLM+RAG requests,
# tracks load/health, and supports failure simulation for fault tolerance testing.
#
# Multiprocessing additions (NEW):
#   worker_process_entry  -- entry-point run inside each OS process
#   ProcessWorkerHandle   -- drop-in replacement for GPUWorker that backs each
#                            logical worker with a real multiprocessing.Process.
#                            Exposes the same public interface (is_alive, status,
#                            active_connections, gpu_utilization, process, get_metrics,
#                            shutdown, revive) so LoadBalancer / HeartbeatMonitor /
#                            Scheduler work without any changes.

import time
import random
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue
from common.models import Request, Response, RequestStatus, WorkerStatus, WorkerMetrics
from common.logger  import get_logger
from rag.retriever  import retrieve_context
from llm.inference  import run_llm

log = get_logger("Worker.GPU")

# each active request adds 20% GPU utilization (capped at 100%)
_UTIL_PER_CONNECTION = 20.0


class GPUWorker:
    def __init__(self, worker_id: int, failure_rate: float = 0.0):
        # failure_rate: 0.0 = stable, 0.4 = crashes 40% of requests (for testing)
        self.worker_id    = worker_id
        self.failure_rate = failure_rate

        self._status             = WorkerStatus.HEALTHY
        self._active_connections = 0
        self._gpu_utilization    = 0.0      # %
        self._total_processed    = 0
        self._total_failed       = 0
        self._latency_history: list[float] = []   # ms

        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        return self._status != WorkerStatus.DEAD

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def active_connections(self) -> int:
        return self._active_connections

    @property
    def gpu_utilization(self) -> float:
        return self._gpu_utilization

    def process(self, request: Request) -> Response:
        # raises RuntimeError if this worker is dead or crashes during processing
        if not self.is_alive:
            raise RuntimeError(
                f"Worker-{self.worker_id} is {self._status.value}. "
                "Cannot accept new requests."
            )

        self._on_request_start()

        start_ns = time.perf_counter_ns()
        try:
            # simulated crash based on failure_rate
            if random.random() < self.failure_rate:
                self._trigger_failure(request.request_id)
                raise RuntimeError(
                    f"Worker-{self.worker_id} crashed while processing "
                    f"request {request.request_id}."
                )

            # RAG step
            context = retrieve_context(request.query)

            # LLM step
            result = run_llm(request.query, context)

            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            self._on_request_complete(latency_ms)

            log.info(
                "Worker-%d OK req=%-8s latency=%6.1f ms gpu=%.0f%% conn=%d",
                self.worker_id, request.request_id, latency_ms,
                self._gpu_utilization, self._active_connections,
            )

            return Response(
                request_id  = request.request_id,
                result      = result,
                status      = RequestStatus.COMPLETED,
                worker_id   = self.worker_id,
                latency_ms  = latency_ms,
                rag_context = context,
            )

        except RuntimeError:
            self._on_request_failed()
            raise

    def shutdown(self):
        # controlled shutdown - marks worker as dead for testing
        with self._lock:
            self._status = WorkerStatus.DEAD
        log.warning("Worker-%d: GRACEFUL SHUTDOWN initiated.", self.worker_id)

    def revive(self):
        # bring a dead worker back online (simulates node restart)
        with self._lock:
            self._status             = WorkerStatus.HEALTHY
            self._active_connections = 0
            self._gpu_utilization    = 0.0
        log.info("Worker-%d: back online.", self.worker_id)

    def get_metrics(self) -> WorkerMetrics:
        with self._lock:
            avg_lat = (
                sum(self._latency_history) / len(self._latency_history)
                if self._latency_history else 0.0
            )
            return WorkerMetrics(
                worker_id          = self.worker_id,
                status             = self._status,
                active_connections = self._active_connections,
                gpu_utilization    = self._gpu_utilization,
                total_processed    = self._total_processed,
                total_failed       = self._total_failed,
                avg_latency_ms     = round(avg_lat, 2),
            )

    def _on_request_start(self):
        with self._lock:
            self._active_connections += 1
            self._gpu_utilization = min(
                100.0, self._active_connections * _UTIL_PER_CONNECTION
            )
            if self._gpu_utilization >= 80.0:
                self._status = WorkerStatus.DEGRADED

    def _on_request_complete(self, latency_ms: float):
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)
            self._gpu_utilization = min(
                100.0, self._active_connections * _UTIL_PER_CONNECTION
            )
            if self._gpu_utilization < 80.0 and self._status == WorkerStatus.DEGRADED:
                self._status = WorkerStatus.HEALTHY
            self._total_processed += 1
            self._latency_history.append(latency_ms)
            # Keep history bounded
            if len(self._latency_history) > 500:
                self._latency_history = self._latency_history[-500:]

    def _on_request_failed(self):
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)
            self._total_failed += 1

    def _trigger_failure(self, request_id: str):
        with self._lock:
            self._status = WorkerStatus.DEAD
        log.error(
            "Worker-%d: CRASH detected on req=%s -- node marked DEAD.",
            self.worker_id, request_id,
        )

    def __repr__(self):
        return (
            f"GPUWorker(id={self.worker_id}, status={self._status.value}, "
            f"gpu={self._gpu_utilization:.0f}%, conn={self._active_connections})"
        )


# ── multiprocessing sentinel ─────────────────────────────────────────────────
_STOP_SENTINEL = "__STOP__"

# ── process entry-point ──────────────────────────────────────────────────────

def worker_process_entry(worker_id: int,
                         failure_rate: float,
                         task_q: Queue,
                         result_q: Queue,
                         metrics_q: Queue) -> None:
    """
    Runs inside an independent OS process (spawned by ProcessWorkerHandle).

    Loop:
      1. Pull a Request (or _STOP_SENTINEL) from task_q.
      2. Run RAG + LLM exactly as GPUWorker.process() does.
      3. Push a Response to result_q.
      4. Push a WorkerMetrics snapshot to metrics_q after every request
         so the parent process can keep its counters up-to-date.
    """
    log = get_logger(f"Worker.GPU-{worker_id}")
    log.info("Worker-%d process started  pid=%d", worker_id, mp.current_process().pid)

    _UTIL_PER_CONN = 20.0
    active   = 0
    gpu_util = 0.0
    total_ok = 0
    total_fail = 0
    latencies: list[float] = []

    def _snap(status: WorkerStatus) -> WorkerMetrics:
        avg = (sum(latencies) / len(latencies)) if latencies else 0.0
        return WorkerMetrics(
            worker_id          = worker_id,
            status             = status,
            active_connections = active,
            gpu_utilization    = gpu_util,
            total_processed    = total_ok,
            total_failed       = total_fail,
            avg_latency_ms     = round(avg, 2),
        )

    current_status = WorkerStatus.HEALTHY

    while True:
        item = task_q.get()
        if item == _STOP_SENTINEL:
            log.info("Worker-%d: received stop signal, exiting.", worker_id)
            metrics_q.put(_snap(WorkerStatus.DEAD))
            break

        request: Request = item

        # increment load
        active   += 1
        gpu_util  = min(100.0, active * _UTIL_PER_CONN)
        if gpu_util >= 80.0:
            current_status = WorkerStatus.DEGRADED

        start_ns = time.perf_counter_ns()
        try:
            # simulated crash
            if random.random() < failure_rate:
                active     = max(0, active - 1)
                gpu_util   = min(100.0, active * _UTIL_PER_CONN)
                total_fail += 1
                current_status = WorkerStatus.DEAD
                log.error("Worker-%d: CRASH on req=%s -- marking DEAD.",
                          worker_id, request.request_id)
                result_q.put(Response(
                    request_id    = request.request_id,
                    result        = "",
                    status        = RequestStatus.FAILED,
                    worker_id     = worker_id,
                    latency_ms    = 0.0,
                    error_message = f"Worker-{worker_id} crashed.",
                ))
                metrics_q.put(_snap(WorkerStatus.DEAD))
                # process exits – parent detects via process.is_alive()
                return

            # RAG step
            context = retrieve_context(request.query)
            # LLM step
            result = run_llm(request.query, context)

            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            active     = max(0, active - 1)
            gpu_util   = min(100.0, active * _UTIL_PER_CONN)
            if gpu_util < 80.0 and current_status == WorkerStatus.DEGRADED:
                current_status = WorkerStatus.HEALTHY
            total_ok += 1
            latencies.append(latency_ms)
            if len(latencies) > 500:
                latencies = latencies[-500:]

            log.info("Worker-%d OK req=%-8s latency=%6.1f ms gpu=%.0f%% conn=%d",
                     worker_id, request.request_id, latency_ms, gpu_util, active)

            result_q.put(Response(
                request_id  = request.request_id,
                result      = result,
                status      = RequestStatus.COMPLETED,
                worker_id   = worker_id,
                latency_ms  = latency_ms,
                rag_context = context,
            ))
            metrics_q.put(_snap(current_status))

        except Exception as exc:  # noqa: BLE001
            active     = max(0, active - 1)
            total_fail += 1
            log.exception("Worker-%d unexpected error on req=%s", worker_id, request.request_id)
            result_q.put(Response(
                request_id    = request.request_id,
                result        = "",
                status        = RequestStatus.FAILED,
                worker_id     = worker_id,
                latency_ms    = 0.0,
                error_message = str(exc),
            ))
            metrics_q.put(_snap(current_status))


# ── ProcessWorkerHandle ──────────────────────────────────────────────────────

class ProcessWorkerHandle:
    """
    Drop-in replacement for GPUWorker that runs each worker in its own OS
    process. The public interface is identical so LoadBalancer, HeartbeatMonitor,
    and Scheduler need no changes.

    Internally it owns:
      task_q    -- parent  → process  (send Request objects)
      result_q  -- process → parent   (receive Response objects)
      metrics_q -- process → parent   (latest WorkerMetrics snapshot)
    """

    def __init__(self, worker_id: int, failure_rate: float = 0.0):
        self.worker_id    = worker_id
        self.failure_rate = failure_rate

        self._task_q    = Queue()
        self._result_q  = Queue()
        self._metrics_q = Queue()
        self._lock      = threading.Lock()

        # cached metrics state (updated from metrics_q on each process() call)
        self._cached: WorkerMetrics = WorkerMetrics(
            worker_id          = worker_id,
            status             = WorkerStatus.HEALTHY,
            active_connections = 0,
            gpu_utilization    = 0.0,
            total_processed    = 0,
            total_failed       = 0,
            avg_latency_ms     = 0.0,
        )

        self._process: Process = Process(
            target = worker_process_entry,
            args   = (worker_id, failure_rate,
                      self._task_q, self._result_q, self._metrics_q),
            name   = f"GPUWorker-{worker_id}",
            daemon = True,
        )
        self._process.start()

    # ── properties that mirror GPUWorker ────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    @property
    def status(self) -> WorkerStatus:
        self._drain_metrics()
        return self._cached.status

    @property
    def active_connections(self) -> int:
        self._drain_metrics()
        return self._cached.active_connections

    @property
    def gpu_utilization(self) -> float:
        self._drain_metrics()
        return self._cached.gpu_utilization

    # ── public methods that mirror GPUWorker ─────────────────────────────────

    def process(self, request: Request) -> Response:
        """
        Send request to the worker process and block until the response arrives.
        Raises RuntimeError if the worker is dead or if it returns a FAILED response
        (so the LoadBalancer retry logic works unchanged).
        """
        if not self.is_alive:
            raise RuntimeError(
                f"Worker-{self.worker_id} process is dead. "
                "Cannot accept new requests."
            )
        self._task_q.put(request)
        response: Response = self._result_q.get()   # blocks
        self._drain_metrics()

        if response.status == RequestStatus.FAILED:
            raise RuntimeError(
                response.error_message
                or f"Worker-{self.worker_id} returned FAILED"
            )
        return response

    def shutdown(self) -> None:
        """Gracefully stop the worker process."""
        if self._process.is_alive():
            self._task_q.put(_STOP_SENTINEL)
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()

    def revive(self) -> None:
        """
        Restart a dead worker process (mirrors GPUWorker.revive()).
        Called by HeartbeatMonitor.
        """
        if self._process.is_alive():
            return  # nothing to do

        # drain stale queue data
        for q in (self._task_q, self._result_q, self._metrics_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:  # noqa: BLE001
                    break

        self._process = Process(
            target = worker_process_entry,
            args   = (self.worker_id, self.failure_rate,
                      self._task_q, self._result_q, self._metrics_q),
            name   = f"GPUWorker-{self.worker_id}",
            daemon = True,
        )
        self._process.start()
        self._cached = WorkerMetrics(
            worker_id          = self.worker_id,
            status             = WorkerStatus.HEALTHY,
            active_connections = 0,
            gpu_utilization    = 0.0,
            total_processed    = 0,
            total_failed       = 0,
            avg_latency_ms     = 0.0,
        )
        get_logger("Worker.GPU").info(
            "Worker-%d: process revived (pid=%d).",
            self.worker_id, self._process.pid,
        )

    def get_metrics(self) -> WorkerMetrics:
        self._drain_metrics()
        with self._lock:
            # reflect live process state
            if not self._process.is_alive():
                self._cached = WorkerMetrics(
                    worker_id          = self.worker_id,
                    status             = WorkerStatus.DEAD,
                    active_connections = self._cached.active_connections,
                    gpu_utilization    = self._cached.gpu_utilization,
                    total_processed    = self._cached.total_processed,
                    total_failed       = self._cached.total_failed,
                    avg_latency_ms     = self._cached.avg_latency_ms,
                )
            return self._cached

    # ── internal ─────────────────────────────────────────────────────────────

    def _drain_metrics(self) -> None:
        """Pull all pending WorkerMetrics snapshots from the process, keep latest."""
        with self._lock:
            latest = None
            while not self._metrics_q.empty():
                try:
                    latest = self._metrics_q.get_nowait()
                except Exception:  # noqa: BLE001
                    break
            if latest is not None:
                self._cached = latest

    def __repr__(self) -> str:
        m = self.get_metrics()
        return (
            f"ProcessWorkerHandle(id={self.worker_id}, "
            f"status={m.status.value}, "
            f"gpu={m.gpu_utilization:.0f}%, "
            f"conn={m.active_connections}, "
            f"pid={self._process.pid})"
        )
