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

def run_governance_agent(retriever):
    print("\nRunning Governance & Promoter Agent...")

    try:
        llm = get_llm(3)

        prompt_template = PromptTemplate.from_template("""
        You are an expert corporate governance analyst specializing in IPO due diligence.
        Using the context from the RHP document, extract all governance and promoter related information.

        Look for:
        - Promoter background and experience
        - Promoter shareholding percentage before and after IPO
        - Promoter share pledging
        - Related party transactions
        - Board composition and independence
        - Management conflicts of interest
        - Any governance red flags

        Context: {context}

        Question: {question}

        Provide output in this exact format:

        GOVERNANCE ISSUE 1:
        Type: [Promoter/Board/Related Party/Shareholding/Pledge]
        Description: [what the issue is]
        Impact: [how this affects investors]
        Red Flag: [Yes/No]

        GOVERNANCE ISSUE 2:
        ...and so on

        OVERALL GOVERNANCE RISK: [High/Medium/Low]
        SUMMARY: [2-3 sentence overall governance assessment]
        """)
        docs = retriever.invoke("Extract promoter shareholding related party transactions governance from this IPO document")
        pages = sorted(list(set([doc.metadata.get('page') for doc in docs if doc.metadata.get('page')])))

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        result = chain.invoke("Extract promoter details, shareholding pattern, related party transactions, board composition and governance red flags from this IPO document.")

        if pages:
            page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
            result += f"\n\n**Sources from RHP:** {page_refs}"

        print("Governance Agent completed!")
        return result

    except Exception as e:
        print(f"ERROR in governance agent: {e}")
        return None