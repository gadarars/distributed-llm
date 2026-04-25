# master/metrics.py
# Collects and reports performance metrics (latency, throughput, worker distribution).
# Thread-safe -- called from multiple worker threads simultaneously.

import threading
import time
from collections import defaultdict
from common.models import Response, RequestStatus
from common.logger import get_logger

log = get_logger("Master.Metrics")


class MetricsCollector:

    def __init__(self):
        self._lock            = threading.Lock()
        self._start_time      = time.perf_counter()
        self._total           = 0
        self._completed       = 0
        self._failed          = 0
        self._latencies: list[float]       = []     # ms
        self._worker_dist: dict[int, int]  = defaultdict(int)

    def record(self, response: Response):
        with self._lock:
            self._total += 1
            if response.status == RequestStatus.COMPLETED:
                self._completed += 1
                self._latencies.append(response.latency_ms)
                self._worker_dist[response.worker_id] += 1
            else:
                self._failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(time.perf_counter() - self._start_time, 1e-9)
            lats = self._latencies or [0.0]
            sorted_lats = sorted(lats)
            n = len(sorted_lats)

            return {
                "total_requests":     self._total,
                "completed":          self._completed,
                "failed":             self._failed,
                "success_rate_pct":   round(100 * self._completed / max(self._total, 1), 1),
                "elapsed_s":          round(elapsed, 2),
                "throughput_rps":     round(self._completed / elapsed, 2),
                "latency_avg_ms":     round(sum(lats) / len(lats), 1),
                "latency_min_ms":     round(min(lats), 1),
                "latency_max_ms":     round(max(lats), 1),
                "latency_p50_ms":     round(sorted_lats[int(n * 0.50)], 1),
                "latency_p95_ms":     round(sorted_lats[int(n * 0.95)], 1),
                "latency_p99_ms":     round(sorted_lats[min(int(n * 0.99), n - 1)], 1),
                "worker_distribution": dict(self._worker_dist),
            }

    def print_report(self):
        s = self.snapshot()
        sep = "-" * 54

        log.info("\n" + sep)
        log.info("  PERFORMANCE REPORT")
        log.info(sep)
        log.info("  Requests   total=%-6d  completed=%-6d  failed=%-6d",
                 s["total_requests"], s["completed"], s["failed"])
        log.info("  Success rate     : %s %%", s["success_rate_pct"])
        log.info("  Elapsed time     : %s s",  s["elapsed_s"])
        log.info("  Throughput       : %s req/s", s["throughput_rps"])
        log.info(sep)
        log.info("  Latency (ms)")
        log.info("    avg  = %s", s["latency_avg_ms"])
        log.info("    min  = %s", s["latency_min_ms"])
        log.info("    max  = %s", s["latency_max_ms"])
        log.info("    p50  = %s", s["latency_p50_ms"])
        log.info("    p95  = %s", s["latency_p95_ms"])
        log.info("    p99  = %s", s["latency_p99_ms"])
        log.info(sep)
        log.info("  Worker distribution")
        for wid, cnt in sorted(s["worker_distribution"].items()):
            log.info("    Worker-%-2d : %4d requests", wid, cnt)
        log.info(sep + "\n")
