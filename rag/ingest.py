# rag/ingest.py
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

def build_vector_store():
    # Automatically get the absolute path to the 'rag' folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "knowledge_base.index")
    docs_path = os.path.join(current_dir, "documents.pkl")

    # 1. Load your documents
    documents = [
    # Document 1: Distributed Systems & Autonomy
    "Topic Tags: Distributed Systems, Autonomous Clusters, Node Coordination. "
    "Distributed computing involves coordinating multiple autonomous machine clusters "
    "over a network. These systems manage CAP theorem trade-offs between consistency, "
    "availability, and partition tolerance. Common architectures include master-worker "
    "setups and decentralized peer-to-peer coordination. "
    "Key Concepts: Autonomous Systems, Distributed Coordination, CAP Theorem.",

    # Document 2: Consensus & Reliability
    "Topic Tags: Consensus Algorithms, Fault Tolerance, System Integrity. "
    "Consensus algorithms like Raft and Paxos allow a cluster of machines to agree "
    "on a single state even during partial failures. This ensures fault tolerance, "
    "reliability, and data integrity in high-availability environments. "
    "Key Concepts: Raft, Paxos, Data Integrity, Cluster Reliability.",

    # Document 3: Load Balancing Basics
    "Topic Tags: Load Balancing, Round-Robin, Least Connections, Traffic Routing. "
    "Load balancing strategies like Round-robin and Least Connections distribute "
    "incoming user traffic across a worker pool. Round-robin is cyclic, while "
    "Least Connections targets nodes with the lowest workload. "
    "Key Concepts: Request Distribution, Traffic Management, Load Balancer.",

    # Document 4: Load-Aware Routing & Telemetry
    "Topic Tags: Load-Aware Routing, Real-time Metrics, GPU Utilization, Telemetry. "
    "Load-aware routing uses real-time telemetry and metrics, such as CPU/GPU "
    "utilization and active connections, to intelligently forward requests. This "
    "minimizes hot-spots in heterogeneous clusters and improves resource efficiency. "
    "Key Concepts: System Metrics, Real-time Telemetry, Resource Optimization.",

    # Document 5: GPU Acceleration & LLM Inference
    "Topic Tags: GPU Acceleration, CUDA Cores, LLM Inference, Parallel Computing. "
    "Modern GPU accelerators, powered by NVIDIA CUDA cores, execute thousands of "
    "tensor operations in parallel. This hardware is optimized for Large Language "
    "Model (LLM) inference and heavy mathematical computations. "
    "Key Concepts: Hardware Acceleration, Parallel Processing, GPU Workers."
]

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Embedding {len(documents)} documents...")
    # 2. Convert text into vector arrays
    embeddings = model.encode(documents)

    # 3. Create the FAISS Index
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)

    # 4. Save using the absolute paths
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
        
    print(f"Vector store created successfully!\nSaved to: {current_dir}")

if __name__ == "__main__":
    build_vector_store()