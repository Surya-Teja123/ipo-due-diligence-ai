from utils.pdf_loader import load_rhp

chunks = load_rhp("data/rhp.pdf")
print(f"\nSample chunk:\n{chunks[0].page_content[:500]}")