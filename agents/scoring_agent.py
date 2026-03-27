import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm_router import get_llm
load_dotenv()

def run_scoring_agent(risk_output, financial_output, legal_output, governance_output, industry_output):
    print("\nRunning Risk Scoring Agent...")

    try:
        llm = get_llm(5)

        prompt_template = PromptTemplate.from_template("""
        You are an expert IPO risk scoring analyst. Based on the outputs from 5 specialized 
        analysis agents, assign a comprehensive risk score to this IPO.

        Use these weights for scoring:
        - Financial Risk: 30%
        - Legal Risk: 25%
        - Governance Risk: 20%
        - Business/General Risk: 15%
        - Industry Risk: 10%

        For each category score from 0 to 100 where:
        0-30 = Low Risk
        31-60 = Medium Risk
        61-100 = High Risk

        Here are the agent outputs:

        RISK AGENT OUTPUT:
        {risk_output}

        FINANCIAL AGENT OUTPUT:
        {financial_output}

        LEGAL AGENT OUTPUT:
        {legal_output}

        GOVERNANCE AGENT OUTPUT:
        {governance_output}

        INDUSTRY AGENT OUTPUT:
        {industry_output}

        Provide output in this exact format:

        CATEGORY SCORES:
        Financial Risk Score: [0-100]
        Legal Risk Score: [0-100]
        Governance Risk Score: [0-100]
        Business Risk Score: [0-100]
        Industry Risk Score: [0-100]

        WEIGHTED FINAL SCORE: [0-100]
        RISK LEVEL: [Low/Medium/High]
        
        CRITICAL RISK FLAGS:
        - [list any critical risks that investors must know]

        SCORING RATIONALE: [3-4 sentences explaining the score]

        DETAILED SCORING CRITERIA:

        For each category explain:
        - What specific evidence from the document drove this score
        - What would make this score better or worse
        - The 2-3 most important factors considered

        FINANCIAL SCORE JUSTIFICATION:
        [specific reasons]

        LEGAL SCORE JUSTIFICATION:
        [specific reasons]

        GOVERNANCE SCORE JUSTIFICATION:
        [specific reasons]

        BUSINESS RISK SCORE JUSTIFICATION:
        [specific reasons]

        INDUSTRY SCORE JUSTIFICATION:
        [specific reasons]                                             
        """)

        chain = prompt_template | llm | StrOutputParser()

        result = chain.invoke({
            "risk_output": risk_output,
            "financial_output": financial_output,
            "legal_output": legal_output,
            "governance_output": governance_output,
            "industry_output": industry_output
        })

        print("Scoring Agent completed!")
        return result

    except Exception as e:
        print(f"ERROR in scoring agent: {e}")
        return None