# Distributed LLM Inference System
CSE354 - Distributed Computing | Ain Shams University

## How to Run

Make sure you have Python 3.9 or higher installed.

```bash
# from the project root folder:
python main.py

# if you get import errors, set the Python path first:
PYTHONPATH=. python main.py
```

The program runs 4 scenarios automatically and prints a performance report after each one.

## How to Change Number of Users

Open `main.py` and change the `NUM_USERS` variable near the top:

```python
NUM_USERS = 1000  # change this to any number
```

You can also change `NUM_WORKERS` to simulate more or fewer GPU nodes.

## Project Structure

| File | Description |
|------|-------------|
| `common/models.py` | Shared data classes: Request, Response, WorkerMetrics |
| `common/logger.py` | Centralized logging setup used by all modules |
| `lb/load_balancer.py` | Load balancer with Round Robin, Least Connections, and Load-Aware strategies |
| `master/scheduler.py` | Receives incoming requests and forwards them to the load balancer |
| `master/heartbeat.py` | Background thread that monitors worker health and auto-revives dead nodes |
| `master/metrics.py` | Tracks latency, throughput, and per-worker request counts |
| `workers/gpu_worker.py` | Simulates a GPU worker node that runs RAG + LLM inference |
| `rag/retriever.py` | Simulates RAG retrieval using a keyword-matched knowledge base |
| `llm/inference.py` | Simulates LLM inference with realistic latency (no real model loaded) |
| `client/load_generator.py` | Spawns concurrent user threads to stress-test the system |
| `main.py` | Entry point - runs all 4 test scenarios |

## Scenarios

1. **Round Robin** - requests distributed in order across workers
2. **Least Connections** - each request goes to the least busy worker
3. **Load-Aware** - routes based on GPU utilization
4. **Fault Tolerance** - two workers crash randomly; heartbeat detects failures and revives them automatically
