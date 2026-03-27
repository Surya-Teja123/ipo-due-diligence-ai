import streamlit as st
import time
import os
import shutil
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.pdf_loader import load_rhp
from utils.vector_store import build_vector_store, get_retriever
from utils.llm_router import get_llm
from agents.risk_agent import run_risk_agent
from agents.financial_agent import run_financial_agent
from agents.legal_agent import run_legal_agent
from agents.governance_agent import run_governance_agent
from agents.industry_agent import run_industry_agent
from agents.scoring_agent import run_scoring_agent
from agents.memo_agent import run_memo_agent

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO Due Diligence AI",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #1a1a2e;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 300;
        margin-top: 0;
    }

    div[data-testid="stSidebar"] {
        background: #1a1a2e !important;
    }

    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] h4 {
        color: #e5e7eb !important;
    }

    div[data-testid="stSidebar"] .stButton > button {
        background: #6366f1;
        color: white !important;
        border: none;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        padding: 10px;
        margin-bottom: 4px;
    }

    div[data-testid="stSidebar"] .stButton > button:hover {
        background: #4f46e5;
    }

    .stChatMessage { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "analysis_done": False,
    "memo": None,
    "risk_output": None,
    "financial_output": None,
    "legal_output": None,
    "governance_output": None,
    "industry_output": None,
    "scoring_output": None,
    "vector_store": None,
    "pdf_path": None,
    "current_page": 1,
    "view_mode": "chat"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# handle page jump from URL params
params = st.query_params
if "page" in params:
    st.session_state.current_page = int(params["page"])

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Upload RHP Document")
    uploaded_file = st.file_uploader("Upload RHP PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        temp_path = "data/temp_rhp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.pdf_path = temp_path
        st.success(f"✅ {uploaded_file.name}")

        if st.button("🚀 Run Full Analysis", type="primary"):
            st.session_state.messages = []
            st.session_state.analysis_done = False
            st.session_state.view_mode = "chat"

            st.session_state.messages.append({
                "role": "user",
                "content": f"Analyze this IPO document: **{uploaded_file.name}**"
            })

            with st.spinner("Loading PDF and building vector store..."):
                chunks = load_rhp(temp_path)
                vector_store = build_vector_store(chunks)
                st.session_state.vector_store = vector_store
                retriever = get_retriever(vector_store, k=15)

            progress = st.progress(0)
            status = st.empty()

            agents = [
                ("🔍 Risk Extraction Agent", run_risk_agent, "risk_output", retriever),
                ("💰 Financial Analysis Agent", run_financial_agent, "financial_output", retriever),
                ("⚖️ Legal & Litigation Agent", run_legal_agent, "legal_output", retriever),
                ("🏛️ Governance & Promoter Agent", run_governance_agent, "governance_output", retriever),
                ("🌐 Industry & Market Agent", run_industry_agent, "industry_output", retriever),
            ]

            for i, (name, func, key, ret) in enumerate(agents):
                status.markdown(f"**Running {name}...**")
                result = func(ret)
                st.session_state[key] = result
                progress.progress((i + 1) / 7)
                time.sleep(3)

            status.markdown("**Running Risk Scoring Agent...**")
            scoring = run_scoring_agent(
                st.session_state.risk_output,
                st.session_state.financial_output,
                st.session_state.legal_output,
                st.session_state.governance_output,
                st.session_state.industry_output
            )
            st.session_state.scoring_output = scoring
            progress.progress(6 / 7)
            time.sleep(3)

            status.markdown("**Running Investment Memo Agent...**")
            memo = run_memo_agent(
                st.session_state.risk_output,
                st.session_state.financial_output,
                st.session_state.legal_output,
                st.session_state.governance_output,
                st.session_state.industry_output,
                st.session_state.scoring_output
            )
            st.session_state.memo = memo
            progress.progress(1.0)
            st.session_state.analysis_done = True

            full_response = """✅ **Risk Extraction** — Complete
✅ **Financial Analysis** — Complete
✅ **Legal & Litigation** — Complete
✅ **Governance & Promoter** — Complete
✅ **Industry & Market** — Complete
✅ **Risk Scoring** — Complete
✅ **Investment Memo** — Complete

---

🎉 **Analysis complete!** Here is your Investment Memo:

"""
            full_response += memo
            full_response += "\n\n---\n*Disclaimer: This memo is AI-generated for educational purposes only.*"

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

            status.empty()
            st.rerun()

    st.divider()

    if st.session_state.analysis_done:
        st.markdown("### 📋 View Sections")

        sections = [
            ("📄 Full Memo", "memo", "Show me the full investment memo"),
            ("🔍 Risk Analysis", "risk_output", "Show me the detailed risk analysis"),
            ("💰 Financial Analysis", "financial_output", "Show me the financial analysis"),
            ("⚖️ Legal Analysis", "legal_output", "Show me the legal analysis"),
            ("🏛️ Governance Analysis", "governance_output", "Show me the governance analysis"),
            ("🌐 Industry Analysis", "industry_output", "Show me the industry analysis"),
            ("📊 Risk Score", "scoring_output", "Show me the risk scores"),
        ]

        for label, key, user_msg in sections:
            if st.button(label):
                content = st.session_state.memo if key == "memo" else st.session_state[key]
                st.session_state.messages.append({"role": "user", "content": user_msg})
                st.session_state.messages.append({"role": "assistant", "content": content})
                st.rerun()

        st.divider()

        if st.session_state.pdf_path:
            if st.button("📑 Toggle PDF Viewer"):
                st.session_state.view_mode = "split" if st.session_state.view_mode == "chat" else "chat"
                st.rerun()

    st.divider()
    st.markdown("**Built with**")
    st.markdown("🦙 Groq LLaMA 3.3 70B")
    st.markdown("🔗 LangChain")
    st.markdown("📚 FAISS Vector Store")
    st.markdown("🖥️ Streamlit")

# ── Main Area ──────────────────────────────────────────────────────────────────
if st.session_state.view_mode == "split" and st.session_state.pdf_path:

    # Inject CSS to kill outer scroll
    st.markdown("""
    <style>
    section.main > div.block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        overflow: hidden !important;
        max-height: 100vh !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_pdf, col_chat = st.columns([1, 1])

    with col_pdf:
        st.markdown("### 📄 RHP Document")

        os.makedirs("static", exist_ok=True)
        shutil.copy(st.session_state.pdf_path, "static/rhp.pdf")

        pdf_url = f"http://localhost:8501/app/static/rhp.pdf#page={st.session_state.current_page}&t={int(time.time())}"

        st.markdown(f'''
            <iframe
                src="{pdf_url}"
                width="100%"
                height="600px"
                style="border:1px solid #374151; border-radius:8px; display:block;">
            </iframe>
        ''', unsafe_allow_html=True)

    with col_chat:
        st.markdown("### 💬 Analysis & Chat")

        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask anything about the IPO..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            retriever = get_retriever(st.session_state.vector_store, k=10)

            def format_docs(docs):
                return "\n\n".join(
                    f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
                    for doc in docs
                )

            llm = get_llm(3)

            answer_prompt = PromptTemplate.from_template("""
You are an expert IPO analyst assistant. Using the investment memo and RHP excerpts below, answer the user's question accurately. Mention page numbers when referencing specific information.

INVESTMENT MEMO:
{memo}

RELEVANT RHP EXCERPTS:
{context}

USER QUESTION:
{question}

Provide a clear, detailed answer:
""")
            docs = retriever.invoke(prompt)
            context = format_docs(docs)
            pages = sorted(list(set([
                doc.metadata.get('page')
                for doc in docs
                if doc.metadata.get('page')
            ])))

            chain = answer_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "memo": st.session_state.memo,
                "context": context,
                "question": prompt
            })

            if pages:
                page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
                response += f"\n\n**Sources from RHP:** {page_refs}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

else:
    # ── Full Chat View ─────────────────────────────────────────────────────────
    st.markdown('<p class="main-title">📊 IPO Due Diligence AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-agent AI system for professional IPO analysis • Powered by LLaMA 3.3 70B + LangChain</p>', unsafe_allow_html=True)
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about the IPO... (e.g. What are the top 3 risks?)"):
        if not st.session_state.analysis_done:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Please upload an RHP PDF and click **Run Full Analysis** in the sidebar first."
            })
            st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            retriever = get_retriever(st.session_state.vector_store, k=10)

            def format_docs(docs):
                return "\n\n".join(
                    f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
                    for doc in docs
                )

            llm = get_llm(3)

            answer_prompt = PromptTemplate.from_template("""
You are an expert IPO analyst assistant. Using the investment memo and RHP excerpts below, answer the user's question accurately. Mention page numbers when referencing specific information.

INVESTMENT MEMO:
{memo}

RELEVANT RHP EXCERPTS:
{context}

USER QUESTION:
{question}

Provide a clear, detailed answer:
""")
            docs = retriever.invoke(prompt)
            context = format_docs(docs)
            pages = sorted(list(set([
                doc.metadata.get('page')
                for doc in docs
                if doc.metadata.get('page')
            ])))

            chain = answer_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "memo": st.session_state.memo,
                "context": context,
                "question": prompt
            })

            if pages:
                page_refs = " ".join([f"`📄 p.{p}`" for p in pages[:5]])
                response += f"\n\n**Sources from RHP:** {page_refs}"
                response += f"\n\n*💡 Tip: Click **📑 Toggle PDF Viewer** in the sidebar to view source pages side by side.*"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()