import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    print("ERROR: API key not found. .env file is not being read.")
else:
    print(f"API Key loaded successfully: {api_key[:10]}...")
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    response = llm.invoke("Say hello in one sentence")
    print("Gemini response:", response.content)