from utils.pdf_loader import load_rhp
from utils.vector_store import build_vector_store, get_retriever
from agents.risk_agent import run_risk_agent
from agents.financial_agent import run_financial_agent
from agents.legal_agent import run_legal_agent
from agents.governance_agent import run_governance_agent
from agents.industry_agent import run_industry_agent
from agents.scoring_agent import run_scoring_agent
from agents.memo_agent import run_memo_agent
import time

chunks = load_rhp("data/rhp.pdf")
vector_store = build_vector_store(chunks)
retriever = get_retriever(vector_store)

risk_output = run_risk_agent(retriever)
print("\n===== RISK AGENT OUTPUT =====")
print(risk_output)

time.sleep(4)

retriever_financial = get_retriever(vector_store, k=15)
financial_output = run_financial_agent(retriever_financial)
print("\n===== FINANCIAL AGENT OUTPUT =====")
print(financial_output)

time.sleep(4)

retriever_legal = get_retriever(vector_store, k=15)
legal_output = run_legal_agent(retriever_legal)
print("\n===== LEGAL AGENT OUTPUT =====")
print(legal_output)

time.sleep(4)

retriever_governance = get_retriever(vector_store, k=15)
governance_output = run_governance_agent(retriever_governance)
print("\n===== GOVERNANCE AGENT OUTPUT =====")
print(governance_output)


time.sleep(4)

retriever_industry = get_retriever(vector_store, k=15)
industry_output = run_industry_agent(retriever_industry)
print("\n===== INDUSTRY AGENT OUTPUT =====")
print(industry_output)


time.sleep(4)

scoring_output = run_scoring_agent(
    risk_output, financial_output, 
    legal_output, governance_output, industry_output
)
print("\n===== SCORING AGENT OUTPUT =====")
print(scoring_output)


time.sleep(4)
memo_output = run_memo_agent(
    risk_output, financial_output,
    legal_output, governance_output,
    industry_output, scoring_output
)
print("\n===== INVESTMENT MEMO =====")
print(memo_output)