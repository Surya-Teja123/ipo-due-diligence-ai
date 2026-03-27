# 📊 IPO Due Diligence AI

An AI-powered multi-agent system that automatically analyzes IPO documents (RHP) and generates a professional investment memo in under 5 minutes.

## 🎯 Problem Statement

IPO documents (Red Herring Prospectus) are 400-800 pages of dense legal and financial language. Human analysts spend days reading them manually, leading to missed red flags, analyst bias, and inconsistent evaluation. This system automates the entire process.

## 🤖 How It Works

7 specialized AI agents work in a sequential pipeline:

| Agent | Role |
|-------|------|
| 🔍 Risk Extraction | Finds and categorizes all risk factors |
| 💰 Financial Analysis | Detects revenue trends, debt levels, red flags |
| ⚖️ Legal & Litigation | Scans for lawsuits, regulatory notices, liabilities |
| 🏛️ Governance & Promoter | Evaluates promoter background and shareholding |
| 🌐 Industry & Market | Assesses competition and macro risks |
| 📊 Risk Scoring | Assigns weighted risk score from 0 to 100 |
| 📝 Investment Memo | Compiles everything into investment recommendation |

## 🛠️ Tech Stack

- **LangChain** — Agent orchestration and RAG pipeline
- **Groq LLaMA 3.3 70B** — Large language model
- **FAISS** — Vector database for intelligent document search
- **HuggingFace Embeddings** — Text to vector conversion
- **PyMuPDF** — PDF parsing and text extraction
- **Streamlit** — Web application interface

## 🧠 Key Concepts

**RAG (Retrieval Augmented Generation)** — Instead of feeding the entire 500 page document to the LLM at once, the system splits it into 2000+ chunks, stores them in FAISS, and retrieves only the most relevant sections per query.

**Multi-Agent Pipeline** — Each agent is a specialized LangChain chain with a domain-specific prompt. Agents 1-5 query the vector store independently. Agent 6 scores their outputs. Agent 7 generates the final memo.

**Citation System** — Every AI claim is linked back to the exact page in the original document for verification.

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Surya-Teja123/ipo-due-diligence-ai.git
cd ipo-due-diligence-ai
```

### 2. Install dependencies
```bash
pip install langchain langchain-groq langchain-community
pip install langchain-huggingface langchain-text-splitters langchain-core
pip install faiss-cpu pymupdf sentence-transformers python-dotenv streamlit
```

### 3. Set up API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY_1=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

### 4. Add your RHP PDF
Place your RHP PDF file in the `data/` folder.

### 5. Run the app
```bash
streamlit run app.py
```

## 📊 Risk Scoring Formula

| Category | Weight |
|----------|--------|
| Financial Risk | 30% |
| Legal Risk | 25% |
| Governance Risk | 20% |
| Business Risk | 15% |
| Industry Risk | 10% |

| Score | Risk Level |
|-------|-----------|
| 0-30 | 🟢 Low |
| 31-60 | 🟡 Medium |
| 61-100 | 🔴 High |

## 🚀 Features

- Upload any RHP PDF and get full analysis in under 5 minutes
- 7 specialized agents each focused on a specific domain
- Weighted risk scoring with detailed justification
- Side by side PDF viewer with source page citations
- Interactive chat to ask anything about the IPO
- Individual section buttons to view each agent's output

## ⚠️ Limitations

- Free Groq API has 100k token daily limit per account
- Vector store is rebuilt every session
- Financial tables in PDFs sometimes misread
- Analysis quality depends on RHP document quality

## 📝 License

This project is for educational purposes only. Not financial advice.
