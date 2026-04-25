# rag/retriever.py
# Simulates a RAG pipeline using a keyword-matched knowledge base.
# In production this would use a vector DB (e.g. FAISS or Pinecone) with embeddings.
# TODO: replace with real vector store integration when GPU resources are available

import time
import random
from common.logger import get_logger

log = get_logger("RAG.Retriever")

# simple knowledge base -- maps topic keywords to context snippets
_KNOWLEDGE_BASE: dict[str, list[str]] = {
    "distributed": [
        "Distributed systems coordinate multiple autonomous nodes over a network to achieve "
        "a common goal, trading consistency for availability as described by the CAP theorem.",
        "Consensus algorithms such as Raft and Paxos ensure that distributed nodes agree on "
        "a single value even in the presence of partial failures.",
    ],
    "load balancing": [
        "Round-robin distributes requests cyclically and works well when all nodes have "
        "similar capacity. Least-connections improves fairness under variable request durations.",
        "Load-aware routing reads real-time CPU/GPU utilisation metrics before forwarding a "
        "request, minimising hot-spots in heterogeneous clusters.",
    ],
    "gpu": [
        "Modern GPU accelerators expose thousands of CUDA cores that can execute tensor "
        "operations in parallel, making them ideal for LLM inference workloads.",
        "GPU memory bandwidth is often the bottleneck during inference; batching requests "
        "amortises that cost and significantly increases throughput.",
    ],
    "llm": [
        "Large Language Models are transformer-based architectures trained on vast corpora. "
        "At inference time they perform autoregressive token generation.",
        "Techniques such as KV-caching and continuous batching reduce per-token latency "
        "in production LLM serving systems like vLLM and TensorRT-LLM.",
    ],
    "fault tolerance": [
        "Fault tolerance in distributed systems is achieved through redundancy, heartbeat "
        "monitoring, and automatic task reassignment upon node failure.",
        "Checkpointing and idempotent task design allow failed jobs to be safely retried "
        "on a different node without producing duplicate side-effects.",
    ],
    "rag": [
        "Retrieval-Augmented Generation grounds LLM responses in external knowledge, "
        "reducing hallucinations and enabling up-to-date answers without retraining.",
        "Dense passage retrieval using bi-encoder models like DPR or Contriever produces "
        "semantically relevant chunks faster than sparse BM25 retrieval.",
    ],
}

_DEFAULT_CONTEXT = (
    "General knowledge: distributed computing involves coordinating multiple machines "
    "to process workloads reliably and efficiently."
)


def retrieve_context(query: str) -> str:
    # simulate retrieval latency (5-30ms like a real vector DB lookup)
    time.sleep(random.uniform(0.005, 0.030))

    query_lower = query.lower()
    for keyword, snippets in _KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            chosen = random.choice(snippets)
            log.debug("Cache hit for keyword '%s'", keyword)
            return chosen

    # Fall back to default context
    return _DEFAULT_CONTEXT
