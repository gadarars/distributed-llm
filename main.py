# main.py
# Entry point for the distributed LLM inference simulation.
# Runs 4 scenarios: Round Robin, Least Connections, Load-Aware, and Fault Tolerance.
#
# To run: python main.py (from the project root)
# Adjust NUM_WORKERS or NUM_USERS below to change the simulation scale.
#
# Multiprocessing change: GPUWorker replaced with ProcessWorkerHandle so each
# logical worker runs in its own OS process. All other logic is unchanged.

import time
import multiprocessing as mp
from workers.gpu_worker     import ProcessWorkerHandle          # NEW: real processes
from lb.load_balancer       import LoadBalancer, Strategy
from master.scheduler       import Scheduler
from master.metrics         import MetricsCollector
from master.heartbeat       import HeartbeatMonitor
from client.load_generator  import run_load_test
from common.logger          import get_logger

log = get_logger("Main")


NUM_WORKERS        = 4
NUM_USERS          = 1000
FAULT_USERS        = 1000     
HEARTBEAT_INTERVAL = 2.0   # seconds between health checks


def run_scenario(
    title:        str,
    workers:      list[ProcessWorkerHandle],
    strategy:     Strategy,
    num_users:    int,
    use_heartbeat: bool = False,
):
    log.info("")
    log.info("=" * 54)
    log.info("  SCENARIO : %s", title)
    log.info("  Strategy : %s", strategy.value)
    log.info("  Workers  : %d   Users: %d", len(workers), num_users)
    log.info("=" * 54)

    metrics   = MetricsCollector()
    lb        = LoadBalancer(workers, strategy=strategy)
    scheduler = Scheduler(lb, metrics)

    monitor = None
    if use_heartbeat:
        monitor = HeartbeatMonitor(workers, interval_s=HEARTBEAT_INTERVAL)
        monitor.start()

    run_load_test(scheduler, num_users=num_users)

    if monitor:
        monitor.stop()

    metrics.print_report()
    metrics.export_to_excel(title)

    # clean up worker processes for this scenario before starting the next
    for w in workers:
        w.shutdown()

    time.sleep(0.3)   # brief pause between scenarios


def main():
    # 1. Round Robin
    run_scenario(
        title     = "Round Robin - stable cluster",
        workers   = [ProcessWorkerHandle(i) for i in range(NUM_WORKERS)],
        strategy  = Strategy.ROUND_ROBIN,
        num_users = NUM_USERS,
    )

    # 2. Least Connections
    run_scenario(
        title     = "Least Connections - stable cluster",
        workers   = [ProcessWorkerHandle(i) for i in range(NUM_WORKERS)],
        strategy  = Strategy.LEAST_CONNECTIONS,
        num_users = NUM_USERS,
    )

    # 3. Load-Aware Routing
    run_scenario(
        title     = "Load-Aware Routing - stable cluster",
        workers   = [ProcessWorkerHandle(i) for i in range(NUM_WORKERS)],
        strategy  = Strategy.LOAD_AWARE,
        num_users = NUM_USERS,
    )

    # 4. Fault Tolerance -- workers 1 and 2 crash randomly, heartbeat active
    fault_workers = [
        ProcessWorkerHandle(0, failure_rate=0.0),   # stable
        ProcessWorkerHandle(1, failure_rate=0.4),   # unstable
        ProcessWorkerHandle(2, failure_rate=0.4),   # unstable
        ProcessWorkerHandle(3, failure_rate=0.0),   # stable
    ]
    run_scenario(
        title         = "Fault Tolerance - workers crash and reassign",
        workers       = fault_workers,
        strategy      = Strategy.LEAST_CONNECTIONS,
        num_users     = FAULT_USERS,
        use_heartbeat = True,
    )

    log.info("All scenarios completed.")


if __name__ == "__main__":
    # "spawn" is safe on Windows, macOS, and Linux.
    # Must be called before any Process is created.
    mp.set_start_method("spawn", force=True)
    main()
