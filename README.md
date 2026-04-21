# 🤖 AI Study Buddy — Agentic AI Capstone Project

**Domain:** Agentic AI Course | **User:** B.Tech students | **Deadline:** April 21, 2026

---
# AI Study Buddy 📚

## Overview
This project is an AI-powered Study Assistant using LangGraph and RAG.

## Features
- Knowledge-based answers
- No hallucination
- Tool usage (date/time)
- Memory-based responses
- Streamlit UI

## Tech Stack
- Python
- LangGraph
- ChromaDB
- SentenceTransformers
- Groq API
- Streamlit

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Add API key in `.env`:
GROQ_API_KEY=your_key

3. Run:
streamlit run capstone_streamlit.py

## Note
API keys are not included for security reasons.

## Project Structure

```
study_buddy/
├── state.py           # CapstoneState TypedDict — design this FIRST
├── knowledge_base.py  # 12 KB documents + ChromaDB + retrieval test
├── nodes.py           # All 8 node functions + routing decision functions
├── graph.py           # LangGraph assembly + ask() helper
├── app.py             # Streamlit UI with @st.cache_resource
├── tests.py           # 20 tests: domain, tool, memory, red-team
├── ragas_eval.py      # RAGAS baseline evaluation (Part 6)
├── VIVA_AND_NOTES.md  # Viva Q&A + error guide + MCQ answers
├── requirements.txt   # All dependencies
└── .env.example       # API key template (copy to .env)
```

---

## Setup (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# Edit .env and add your Groq API key from https://console.groq.com

# 4. Test the KB retrieval (ALWAYS do this first)
python knowledge_base.py

# 5. Smoke test the graph
python graph.py

# 6. Run full test suite
python tests.py

# 7. Launch Streamlit UI
streamlit run app.py
```

---

## Agent Capabilities

| Capability | Implementation |
|-----------|----------------|
| RAG (Knowledge Base) | ChromaDB + SentenceTransformer (all-MiniLM-L6-v2) |
| Memory | MemorySaver + thread_id (sliding window: 6 messages) |
| Routing | LLM-based router (retrieve / tool / memory_only) |
| Tools | Date/time tool + safe AST calculator |
| Self-evaluation | Faithfulness scoring (0.0–1.0) with retry logic |
| UI | Streamlit with @st.cache_resource |

---

## Six Mandatory Capabilities (Verified)

- [x] LangGraph StateGraph (8 nodes)
- [x] ChromaDB RAG (12 documents)
- [x] MemorySaver + thread_id
- [x] Self-reflection eval node (faithfulness threshold 0.7)
- [x] Tool use (date + calculator)
- [x] Streamlit deployment

---

## Submission Files

1. `day13_capstone.ipynb` — Completed notebook (Kernel → Restart & Run All)
2. `app.py` — Streamlit application
3. `graph.py` + supporting modules — Agent code

---

## Tech Stack

- LangGraph 0.2+ | LangChain 0.3+
- ChromaDB 0.4+ | SentenceTransformers 3.0+
- Groq API (llama-3.3-70b-versatile)
- Streamlit 1.40+
- Python 3.10+
