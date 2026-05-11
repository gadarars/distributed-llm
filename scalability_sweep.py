# scalability_sweep.py
# Runs the system across multiple user counts and worker counts to produce
# the scalability data required by the project report.
#
# Usage:
#   python scalability_sweep.py
#
# Outputs:
#   scalability_results.xlsx  – two sheets: UserScaling and WorkerScaling

import time
import multiprocessing as mp
import pandas as pd

from workers.gpu_worker    import ProcessWorkerHandle
from lb.load_balancer      import LoadBalancer, Strategy
from master.scheduler      import Scheduler
from master.metrics        import MetricsCollector
from client.load_generator import run_load_test
from common.logger         import get_logger

log = get_logger("Scalability")


def _run_once(num_workers: int, num_users: int) -> dict:
    """Spin up a fresh cluster, run the load test, return metrics snapshot."""
    workers   = [ProcessWorkerHandle(i) for i in range(num_workers)]
    metrics   = MetricsCollector()
    lb        = LoadBalancer(workers, strategy=Strategy.LEAST_CONNECTIONS)
    scheduler = Scheduler(lb, metrics)

    run_load_test(scheduler, num_users=num_users)

    snap = metrics.snapshot()

    for w in workers:
        w.shutdown()
    time.sleep(0.3)

    return snap


# ── User Scaling ──────────────────────────────────────────────────────────────
def user_scaling_sweep():
    """Fix 4 workers, vary user count: 100 → 250 → 500 → 750 → 1000."""
    user_counts  = [100, 250, 500, 750, 1000]
    num_workers  = 4
    rows         = []

    log.info("=" * 54)
    log.info("  USER SCALING SWEEP  (workers=%d)", num_workers)
    log.info("=" * 54)

    for n in user_counts:
        log.info("Running: %d users …", n)
        snap = _run_once(num_workers=num_workers, num_users=n)
        rows.append({
            "Users":             n,
            "Workers":           num_workers,
            "Success Rate (%)":  snap["success_rate_pct"],
            "Throughput (req/s)": snap["throughput_rps"],
            "Avg Latency (ms)":  snap["latency_avg_ms"],
            "P95 Latency (ms)":  snap["latency_p95_ms"],
        })
        log.info(
            "  users=%-4d  thr=%.2f req/s  avg_lat=%.1f ms  p95=%.1f ms",
            n, snap["throughput_rps"], snap["latency_avg_ms"], snap["latency_p95_ms"]
        )

    return rows


# ── Worker Scaling ────────────────────────────────────────────────────────────
def worker_scaling_sweep():
    """Fix 500 users, vary worker count: 1 → 2 → 4 → 8."""
    worker_counts = [1, 2, 4, 8]
    num_users     = 500
    rows          = []

    log.info("=" * 54)
    log.info("  WORKER SCALING SWEEP  (users=%d)", num_users)
    log.info("=" * 54)

    for n in worker_counts:
        log.info("Running: %d workers …", n)
        snap = _run_once(num_workers=n, num_users=num_users)
        rows.append({
            "Workers":           n,
            "Users":             num_users,
            "Success Rate (%)":  snap["success_rate_pct"],
            "Throughput (req/s)": snap["throughput_rps"],
            "Avg Latency (ms)":  snap["latency_avg_ms"],
            "P95 Latency (ms)":  snap["latency_p95_ms"],
        })
        log.info(
            "  workers=%-2d  thr=%.2f req/s  avg_lat=%.1f ms  p95=%.1f ms",
            n, snap["throughput_rps"], snap["latency_avg_ms"], snap["latency_p95_ms"]
        )

    return rows


def main():
    user_rows   = user_scaling_sweep()
    worker_rows = worker_scaling_sweep()

    filename = "scalability_results.xlsx"
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(user_rows).to_excel(writer,   sheet_name="UserScaling",   index=False)
        pd.DataFrame(worker_rows).to_excel(writer, sheet_name="WorkerScaling", index=False)

    log.info("Scalability results saved to %s", filename)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
