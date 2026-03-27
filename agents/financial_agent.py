import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.llm_router import get_llm

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_financial_agent(retriever):
    print("\nRunning Financial Analysis Agent...")

    try:
        llm = get_llm(4)

        prompt_template = PromptTemplate.from_template("""
        You are an expert financial analyst specializing in IPO analysis.
        Using the context from the RHP document, extract and analyze key financial information.

        Look for:
        - Revenue trends (growth or decline)
        - Profitability (net profit/loss margins)
        - Debt levels and debt-to-equity ratio
        - Cash flow situation
        - Revenue concentration risks
        - Any financial red flags

        Context: {context}

        Question: {question}

        Provide output in this exact format:

        FINANCIAL METRIC 1:
        Metric: [metric name]
        Value: [actual value or trend]
        Observation: [what this means for investors]
        Red Flag: [Yes/No]

        OVERALL FINANCIAL HEALTH: [Strong/Moderate/Weak]
        SUMMARY: [2-3 sentence overall financial assessment]
        """)
        docs = retriever.invoke("Extract financial data revenue profit debt from this IPO document")
        pages = sorted(list(set([doc.metadata.get('page') for doc in docs if doc.metadata.get('page')])))
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        result = chain.invoke("Extract financial data including revenue, profit, losses, EBITDA, debt, cash flow from this IPO document.")

        if pages:
            page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
            result += f"\n\n**Sources from RHP:** {page_refs}"

        print("Financial Analysis Agent completed!")
        return result

    except Exception as e:
        print(f"ERROR in financial agent: {e}")
        return None