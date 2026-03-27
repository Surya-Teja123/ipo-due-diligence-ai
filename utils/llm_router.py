import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Track which key to use next
_current_key = [1]

def get_llm(key_index=1):
    key = os.getenv(f"GROQ_API_KEY_{key_index}")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=key,
        temperature=0.1
    )

def get_fresh_llm():
    """Tries keys in order until one works"""
    for i in range(1, 6):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=key,
                temperature=0.1
            )