from utils.pdf_loader import load_rhp
from utils.vector_store import build_vector_store, get_retriever

chunks = load_rhp("data/rhp.pdf")
vector_store = build_vector_store(chunks)
retriever = get_retriever(vector_store)

results = retriever.invoke("What are the main risk factors?")
print(f"\nFound {len(results)} relevant chunks")
print(f"\nSample result:\n{results[0].page_content[:500]}")