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

def run_legal_agent(retriever):
    print("\nRunning Legal & Litigation Agent...")

    try:
        llm = get_llm(5)

        prompt_template = PromptTemplate.from_template("""
        You are an expert legal analyst specializing in IPO due diligence.
        Using the context from the RHP document, extract all legal and litigation information.

        Look for:
        - Ongoing lawsuits or legal cases
        - Regulatory penalties or notices
        - Contingent liabilities
        - Compliance issues
        - Criminal or civil proceedings against promoters or company

        Context: {context}

        Question: {question}

        Provide output in this exact format:

        LEGAL ISSUE 1:
        Type: [Lawsuit/Regulatory/Compliance/Criminal/Civil]
        Description: [what the issue is]
        Amount Involved: [monetary amount if mentioned, else N/A]
        Status: [Ongoing/Resolved/Pending]
        Severity: [High/Medium/Low]

        LEGAL ISSUE 2:
        ...and so on

        OVERALL LEGAL RISK: [High/Medium/Low]
        SUMMARY: [2-3 sentence overall legal risk assessment]
        """)
        docs = retriever.invoke("Extract legal cases litigation regulatory notices from this IPO document")
        pages = sorted(list(set([doc.metadata.get('page') for doc in docs if doc.metadata.get('page')])))

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        result = chain.invoke("Extract all legal cases, litigation, regulatory notices, contingent liabilities and compliance issues from this IPO document.")

        if pages:
            page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
            result += f"\n\n**Sources from RHP:** {page_refs}"

        print("Legal Agent completed!")
        return result

    except Exception as e:
        print(f"ERROR in legal agent: {e}")
        return None