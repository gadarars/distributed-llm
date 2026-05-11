# Distributed LLM Inference System

**CSE354 – Distributed Computing | Ain Shams University | 2nd Semester 2025/2026**

> A complete distributed system for handling **1,000+ concurrent LLM inference requests** across a simulated GPU cluster. Implements three load-balancing strategies, real semantic search (RAG via FAISS), automatic fault detection and node revival, and an optional FastAPI HTTP layer for external clients.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Technology Stack](#4-technology-stack)
5. [Installation](#5-installation)
6. [How to Run](#6-how-to-run)
7. [Test Scenarios & Results](#7-test-scenarios--results)
8. [Limitations](#8-limitations)
9. [Team](#9-team)
10. [References](#10-references)

---

## 1. Project Overview

Modern AI services must serve millions of users in parallel. Each request requires two expensive operations:

1. **RAG** — semantic search through a knowledge base (FAISS + sentence transformers).
2. **LLM Inference** — a neural network forward pass to generate the response (simulated).

This project builds a complete distributed system that handles both operations at scale, demonstrating the core principles of distributed computing: parallel processing, load balancing, fault tolerance, and performance measurement.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│               Layer 1: Client Layer                  │
│     1000 concurrent users  /  FastAPI HTTP client    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│             Layer 2: Load Balancer                   │
│   Round Robin  │  Least Connections  │  Load-Aware   │
│         Retry logic (MAX_RETRIES = 5)                │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│        Layer 3: Master Scheduler + Heartbeat         │
│  Scheduler dispatches; Heartbeat monitors health     │
│  MetricsCollector records latency / throughput       │
└──────┬─────────┬─────────┬─────────┬────────────────┘
       │         │         │         │
  ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐ ┌▼───────┐
  │Worker-0│ │Worker-1│ │Worker-2│ │Worker-3│  Layer 4: GPU Workers
  │Process │ │Process │ │Process │ │Process │  (real OS processes)
  └────┬───┘ └───┬────┘ └──┬─────┘ └┬───────┘
       └─────────┴──────────┴────────┘
                      │
       ┌──────────────▼──────────────────┐
       │      Layer 5: Inference Layer   │
       │  RAG (FAISS + SentenceTransf.)  │
       │  LLM Inference (simulated)      │
       └─────────────────────────────────┘
```

---

## 3. Project Structure

```
distributed-llm/
├── main.py                  # Entry point — runs all 4 test scenarios
├── server.py                # FastAPI HTTP server (optional, for external clients)
├── client_test.py           # External HTTP load test client (20 requests)
├── scalability_sweep.py     # Automated scalability benchmarking
├── sample_output.txt        # Representative terminal output from a full run
├── handover.md              # Internal team handover notes
├── .gitignore
│
├── common/
│   ├── models.py            # Request, Response, WorkerMetrics dataclasses
│   └── logger.py            # Centralised logging used by all modules
│
├── lb/
│   └── load_balancer.py     # 3 strategies + MAX_RETRIES=5 retry logic
│
├── master/
│   ├── scheduler.py         # Receives requests and dispatches via load balancer
│   ├── heartbeat.py         # Background health monitor + auto-revival
│   └── metrics.py           # Thread-safe metrics collector + Excel export
│
├── workers/
│   └── gpu_worker.py        # GPUWorker + ProcessWorkerHandle (IPC wrapper)
│
├── llm/
│   └── inference.py         # Simulated LLM with realistic latency model
│
├── rag/
│   ├── ingest.py            # One-time: build FAISS index from knowledge documents
│   ├── retriever.py         # Query embedding → FAISS search → context retrieval
│   ├── knowledge_base.index # Pre-built FAISS flat L2 index (binary, committed)
│   └── documents.pkl        # Serialised embedded document store (committed)
│
├── client/
│   └── load_generator.py    # Spawns N concurrent user threads in batches of 50
│
├── test_rag.py              # RAG smoke test (single semantic query)
├── test_ragv2.py            # Extended RAG tests (3 query types)
│
└── Results_*.xlsx           # Auto-generated Excel reports (one per scenario)
```

---

## 4. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core implementation |
| Concurrency | `multiprocessing`, `threading` | Parallel workers and monitoring |
| HTTP Server | FastAPI + Uvicorn | Optional `/generate` REST endpoint |
| Vector Search | FAISS (`faiss-cpu`) | Semantic similarity search |
| Embeddings | `sentence-transformers` | Text → 384-dim vector encoding |
| LLM Simulation | `time.sleep` + templates | Realistic inference latency |
| Metrics Export | `pandas` + `openpyxl` | Excel report generation |
| HTTP Client | `requests` | External load test client |

---

## 5. Installation

**Requirements**: Python 3.9 or higher.

```bash
# 1. Clone the repository
git clone https://github.com/gadarars/distributed-llm.git
cd distributed-llm

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell

# 3. Install dependencies
pip install faiss-cpu sentence-transformers pandas openpyxl fastapi uvicorn requests
```

> `rag/knowledge_base.index` and `rag/documents.pkl` are already committed. Only run `python rag/ingest.py` if they are missing or you modify the knowledge base.

---

## 6. How to Run

### 6.1 Main simulation (all 4 scenarios)

```bash
python main.py

# If you get import errors, set PYTHONPATH first:
PYTHONPATH=. python main.py           # Linux / macOS
$env:PYTHONPATH="."; python main.py   # Windows PowerShell
```

Runs Round Robin, Least Connections, Load-Aware, and Fault Tolerance — 1,000 users each.  
Exports one Excel report per scenario. **Expected runtime: ~10–15 minutes.**

To change scale, edit in `main.py`:

```python
NUM_WORKERS = 4      # simulated GPU nodes
NUM_USERS   = 1000   # concurrent users per scenario
```

### 6.2 Optional: HTTP API server + external client

```bash
# Terminal 1 — start the server
uvicorn server:app --host 127.0.0.1 --port 8000

# Terminal 2 — run the external load test
python client_test.py
```

### 6.3 Scalability sweep

```bash
python scalability_sweep.py
```

Tests user counts 100 → 1000 (4 workers) and worker counts 1 → 8 (500 users).  
Saves results to `scalability_results.xlsx`. **Expected runtime: ~20–30 minutes.**

### 6.4 RAG pipeline tests

```bash
python test_rag.py      # single semantic query
python test_ragv2.py    # 3 query types
```

---

## 7. Test Scenarios & Results

All four scenarios were run with 1,000 concurrent users.

### Performance Summary

| Metric | Round Robin | Least Conn. | Load-Aware | Fault Tol. |
|---|---|---|---|---|
| Completed / Total | 1000/1000 | 1000/1000 | 1000/1000 | 1000/1000 |
| Success Rate | **100%** | **100%** | **100%** | **100%** |
| Throughput (req/s) | **15.22** | 13.62 | 13.53 | 2.02 |
| Avg Latency (ms) | 200.8 | 195.7 | **195.5** | 405.8 |
| P95 Latency (ms) | 291.7 | **245.8** | 277.6 | 343.2 |

### Worker Distribution

| Worker | Round Robin | Least Conn. | Load-Aware | Fault Tol. |
|---|---|---|---|---|
| Worker-0 | 250 | 254 | 260 | **463** |
| Worker-1 | 250 | 244 | 252 | 38 |
| Worker-2 | 250 | 245 | 247 | 42 |
| Worker-3 | 250 | 257 | 241 | **457** |



---

## 8. Limitations

1. **Simulated GPU**: Workers use `time.sleep()` to mimic inference delay. No real LLM model is loaded.
2. **Single machine**: All processes run on one physical host. A production system distributes workers over a real network.
3. **No persistent queue**: If all workers fail simultaneously and retries are exhausted, requests are marked failed. Production systems use Kafka or RabbitMQ.
4. **Fixed worker pool**: Workers are allocated at startup; dynamic auto-scaling is not implemented.
5. **Small knowledge base**: The RAG index contains only 5 documents.

---

## 9. Team

| Name | Student ID |
|---|---|
| Nour Eldeen Ahmed Ali Rehab | 22P0040 |
| Mohamed Yasser Khallaf | 22P0245 |
| Roaa Sherif Gadara | 22P0188 |
| Mostafa Fouad Ahmed | 22P0077 |
| Ibrahim Shaker Hammad | 22P0056 |

**Supervisor**: Dr. Ayman & Eng. Alaa Hamdy  
**Course**: CSE354 – Distributed Computing, 2nd Semester 2025/2026  
**Institution**: Faculty of Engineering, Ain Shams University  
**Demo Video**: https://youtu.be/sSoPts4GpLM  
**GitHub**: https://github.com/gadarars/distributed-llm

---

## 10. References

1. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data.*
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019.*
3. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.*
4. Van Steen, M., & Tanenbaum, A. S. (2017). *Distributed Systems* (3rd ed.).
5. Python Software Foundation. multiprocessing – Process-based parallelism. *Python 3.12 Docs.* https://docs.python.org/3/library/multiprocessing.html
6. McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly Media.