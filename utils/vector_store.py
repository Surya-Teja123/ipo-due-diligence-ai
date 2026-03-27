import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(chunks):
    print("Building vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store built successfully!")
    return vector_store

def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(search_kwargs={"k": k})