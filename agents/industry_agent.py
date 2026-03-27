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

def run_industry_agent(retriever):
    print("\nRunning Industry & Market Risk Agent...")

    try:
        llm = get_llm(4)

        prompt_template = PromptTemplate.from_template("""
        You are an expert industry analyst specializing in IPO due diligence.
        Using the context from the RHP document, extract all industry and market risk information.

        Look for:
        - Industry overview and growth prospects
        - Market competition and competitive position
        - Regulatory environment and dependencies
        - Cyclicality of the industry
        - Foreign revenue exposure
        - Macro economic risks affecting the industry
        - Technological disruption risks

        Context: {context}

        Question: {question}

        Provide output in this exact format:

        INDUSTRY RISK 1:
        Type: [Competition/Regulatory/Cyclicality/Macro/Technology/Foreign Exposure]
        Description: [what the risk is]
        Impact on IPO: [how this affects the investment]
        Severity: [High/Medium/Low]

        INDUSTRY RISK 2:
        ...and so on

        OVERALL INDUSTRY RISK: [High/Medium/Low]
        SUMMARY: [2-3 sentence overall industry risk assessment]
        """)

        docs = retriever.invoke("Extract industry overview market competition regulatory environment from this IPO document")
        pages = sorted(list(set([doc.metadata.get('page') for doc in docs if doc.metadata.get('page')])))
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        result = chain.invoke("Extract industry overview, market competition, regulatory environment, cyclicality and macro risks from this IPO document.")

        if pages:
            page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
            result += f"\n\n**Sources from RHP:** {page_refs}"

        print("Industry Agent completed!")
        return result

    except Exception as e:
        print(f"ERROR in industry agent: {e}")
        return None