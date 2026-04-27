from rag.retriever import retrieve_context

# 1. Semantic Test (Should match the GPU snippet)
print(f"Result 1: {retrieve_context('How do hardware accelerators help with AI?')}")

# 2. Concept Test (Should match the CAP theorem/Distributed snippet)
print(f"Result 2: {retrieve_context('What are the trade-offs in autonomous machine clusters?')}")

# 3. Specific Technology Test (Should match the Load-aware snippet)
print(f"Result 3: {retrieve_context('Tell me about monitoring telemetry for routing')}")