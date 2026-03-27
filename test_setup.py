import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {api_key[:10]}...")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

response = llm.invoke("Say hello in one sentence")
print("Groq response:", response.content)