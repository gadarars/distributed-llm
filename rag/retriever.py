# rag/retriever.py
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from common.logger import get_logger

log = get_logger("RAG.Retriever")

# Global variables to hold the model and index in memory so we don't 
# reload them from the hard drive on every single user request.
_model = None
_index = None
_documents = None

def _init_rag():
    global _model, _index, _documents
    # Only load the model and database if they haven't been loaded yet
    if _model is None:
        log.info("Loading SentenceTransformer model and FAISS index...")
        try:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            _index = faiss.read_index("rag/knowledge_base.index")
            with open("rag/documents.pkl", "rb") as f:
                _documents = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError("FAISS index not found. Please run 'python rag/ingest.py' first.")

def retrieve_context(query: str, top_k: int = 1) -> str:
    """Embeds the query and searches the FAISS index for the closest match."""
    _init_rag() 

    start_time = time.time()
    
    # 1. Turn the user's string query into a mathematical vector
    query_vector = _model.encode([query])
    
    # 2. Search the FAISS index for the closest matching vector
    distances, indices = _index.search(query_vector, top_k)
    
    # 3. Grab the index number of the best match
    match_idx = indices[0][0]
    
    # 4. Retrieve the actual text using that index number
    if match_idx == -1: 
        return "No relevant context found in the database."
        
    chosen_context = _documents[match_idx]
    
    latency = time.time() - start_time
    log.debug("Real RAG lookup took %.3fs (Matched Index: %d)", latency, match_idx)
    
    return chosen_context