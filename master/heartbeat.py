# master/heartbeat.py
# Runs a background thread that periodically checks worker health
# and logs the current state of the cluster.

import threading
import time
from common.models import WorkerStatus
from common.logger import get_logger

log = get_logger("Master.Heartbeat")

_STATUS_LABEL = {
    WorkerStatus.HEALTHY  : "OK",
    WorkerStatus.DEGRADED : "DEGRADED",
    WorkerStatus.DEAD     : "DEAD",
}


class HeartbeatMonitor:

    def __init__(self, workers, interval_s: float = 2.0):
        # workers: list of GPUWorker to monitor
        # interval_s: seconds between health checks
        self._workers  = workers
        self._interval = interval_s
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target = self._run,
            name   = "HeartbeatMonitor",
            daemon = True,
        )

    def start(self):
        self._thread.start()
        log.info("Heartbeat monitor started  (interval=%.1f s)", self._interval)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self._interval + 1)
        log.info("Heartbeat monitor stopped.")

    def _run(self):
        while not self._stop.wait(self._interval):
            self._sweep()

    def _sweep(self):
        alive = [w for w in self._workers if w.is_alive]
        dead  = [w for w in self._workers if not w.is_alive]

        # auto-revive dead workers after 2 intervals (simulates node restart)
        for w in dead:
            if not hasattr(w, '_dead_since'):
                w._dead_since = time.time()
            elif time.time() - w._dead_since >= self._interval * 2:
                w.revive()
                del w._dead_since
                log.info("Worker-%d revived by heartbeat monitor.", w.worker_id)

        lines = []
        for w in self._workers:
            m     = w.get_metrics()
            label = _STATUS_LABEL.get(m.status, "?")
            lines.append(
                f"  Worker-{w.worker_id} [{label}] "
                f"gpu={m.gpu_utilization:5.1f}%  "
                f"conn={m.active_connections:2d}  "
                f"done={m.total_processed:4d}  "
                f"failed={m.total_failed:3d}  "
                f"avg_lat={m.avg_latency_ms:6.1f} ms"
            )

        summary = (
            f"Cluster health -- alive={len(alive)}/{len(self._workers)}"
            + (f"  DEAD={[w.worker_id for w in dead]}" if dead else "")
        )
        log.info("%s\n%s", summary, "\n".join(lines))
