from rag.retriever import retrieve_context

# Test with a query that DOES NOT use exact keywords from the snippets
query = "Tell me about node coordination in a network"
result = retrieve_context(query)

print(f"Query: {query}")
print(f"Matched Context: {result}")