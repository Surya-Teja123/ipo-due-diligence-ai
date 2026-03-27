import os
from dotenv import load_dotenv

load_dotenv()

for i in range(1, 6):
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if key:
        print(f"Key {i}: {key[:15]}...")
    else:
        print(f"Key {i}: NOT FOUND")