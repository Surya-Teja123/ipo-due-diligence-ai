import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_rhp(pdf_path: str):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks