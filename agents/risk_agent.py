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

def run_risk_agent(retriever):
    print("\nRunning Risk Extraction Agent...")

    llm = get_llm(3)

    prompt_template = PromptTemplate.from_template("""
    You are an expert IPO risk analyst. Using the context provided from the RHP document,
    extract and categorize all risk factors.

    For each risk found, provide:
    - Risk Category (Business/Financial/Legal/Governance/Industry)
    - Risk Title (short name)
    - Risk Description (2-3 sentences)
    - Severity (High/Medium/Low)

    Context: {context}

    Question: {question}

    Provide output in this exact format:

    RISK 1:
    Category: [category]
    Title: [title]
    Description: [description]
    Severity: [severity]

    RISK 2:
    ...and so on
    """)
    docs = retriever.invoke("Extract and categorize all major risk factors from this IPO document")
    context = format_docs(docs)

    pages = sorted(list(set([
        doc.metadata.get('page')
        for doc in docs
        if doc.metadata.get('page')
    ])))
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    result = chain.invoke("Extract and categorize all major risk factors from this IPO document")

    if pages:
        page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
        result += f"\n\n**Sources from RHP:** {page_refs}"

    print("Risk Extraction Agent completed!")
    return result