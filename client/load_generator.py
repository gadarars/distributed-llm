# client/load_generator.py
# Simulates concurrent users sending queries to the system.

import random
import threading
from common.models import Request
from common.logger import get_logger

log = get_logger("Client.LoadGen")

# sample queries for testing
_QUERY_POOL = [
    "What is distributed computing and why is it important?",
    "Explain load balancing strategies in GPU clusters.",
    "How does fault tolerance work in distributed LLM inference?",
    "What is Retrieval-Augmented Generation (RAG) and how does it help?",
    "Compare round-robin vs least-connections load balancing.",
    "How do GPU worker nodes handle parallel LLM requests?",
    "What happens when a worker node fails during inference?",
    "Explain the role of a master scheduler in a distributed system.",
    "How does load-aware routing reduce GPU hot-spots?",
    "What metrics are used to evaluate distributed system performance?",
    "Describe the CAP theorem and its relevance to this system.",
    "How does heartbeat monitoring detect node failures?",
    "What is continuous batching in LLM serving?",
    "Explain KV-cache and its impact on inference latency.",
    "How should task reassignment be handled after a worker crash?",
]


def _user_thread(scheduler, user_id: int, results: list, lock: threading.Lock):
    query = random.choice(_QUERY_POOL)
    request = Request(query=query, user_id=user_id)
    response = scheduler.handle_request(request)

    log.info(
        "User-%-4d  req=%-8s  worker=%-3s  latency=%6.1f ms  status=%s",
        user_id,
        response.request_id,
        str(response.worker_id) if response.worker_id >= 0 else "ERR",
        response.latency_ms,
        response.status.value,
    )

    with lock:
        results.append(response)


def run_load_test(scheduler, num_users: int = 100, batch_size: int = 50) -> list:
    # launch users in batches to avoid overwhelming the OS thread limit
    log.info("=" * 54)
    log.info("  Starting load test  --  users=%d", num_users)
    log.info("=" * 54)

    results = []
    lock = threading.Lock()

    for batch_start in range(0, num_users, batch_size):
        batch_end = min(batch_start + batch_size, num_users)
        threads = []
        for uid in range(batch_start, batch_end):
            t = threading.Thread(
                target=_user_thread,
                args=(scheduler, uid, results, lock),
                daemon=True,
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    log.info("Load test complete  --  %d/%d responses received.",
             len(results), num_users)
    return results
