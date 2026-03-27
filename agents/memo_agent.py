import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm_router import get_llm
load_dotenv()

def run_memo_agent(risk_output, financial_output, legal_output, governance_output, industry_output, scoring_output):
    print("\nRunning Investment Memo Agent...")

    try:
        llm = get_llm(3)

        prompt_template = PromptTemplate.from_template("""
        You are a senior investment analyst at a top tier investment bank.
        Based on the outputs from all analysis agents, write a professional 
        investment memo that a fund manager would read before making an IPO decision.

        RISK ANALYSIS:
        {risk_output}

        FINANCIAL ANALYSIS:
        {financial_output}

        LEGAL ANALYSIS:
        {legal_output}

        GOVERNANCE ANALYSIS:
        {governance_output}

        INDUSTRY ANALYSIS:
        {industry_output}

        RISK SCORING:
        {scoring_output}

        Write a professional investment memo in this exact format:

        ================================================
        INVESTMENT MEMO — IPO DUE DILIGENCE REPORT
        ================================================

        EXECUTIVE SUMMARY:
        [3-4 sentences summarizing the IPO and overall assessment]

        COMPANY OVERVIEW:
        [2-3 sentences about the company and its business]

        KEY STRENGTHS:
        1. [strength]
        2. [strength]
        3. [strength]

        KEY RISKS:
        1. [risk]
        2. [risk]
        3. [risk]

        FINANCIAL HIGHLIGHTS:
        [3-4 sentences on financial health]

        LEGAL & GOVERNANCE ASSESSMENT:
        [3-4 sentences on legal and governance risks]

        INDUSTRY OUTLOOK:
        [2-3 sentences on industry prospects]

        RISK SCORE SUMMARY:
        [Include the scores from scoring agent]

        INVESTMENT RECOMMENDATION:
        [Subscribe/Avoid/Neutral with reasoning in 3-4 sentences]

        AI CONFIDENCE LEVEL: [High/Medium/Low]

        ---
        *Disclaimer: This memo is AI-generated for educational purposes only.*
        ================================================
        """)

        chain = prompt_template | llm | StrOutputParser()

        result = chain.invoke({
            "risk_output": risk_output,
            "financial_output": financial_output,
            "legal_output": legal_output,
            "governance_output": governance_output,
            "industry_output": industry_output,
            "scoring_output": scoring_output
        })

        print("Memo Agent completed!")
        return result

    except Exception as e:
        print(f"ERROR in memo agent: {e}")
        return None