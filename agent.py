
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from datetime import datetime
import re

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import chromadb
from sentence_transformers import SentenceTransformer

# 🔐 Load API Key
load_dotenv()
llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# 🧠 Embeddings + DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("study_bot")

# 📚 Documents (keep yours — shortened here)
DOCUMENTS = [
    {"id": "doc_001", "topic": "LangGraph Basics",
     "text": "LangGraph is a Python library for building stateful AI agents using graphs."},
]

def load_documents():
    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    emb = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=emb, ids=ids)

load_documents()

# 📃 STATE
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

# 🧠 MEMORY
def memory_node(state):
    msgs = state.get("messages", [])
    msgs.append({"role": "user", "content": state["question"]})
    return {"messages": msgs[-6:], "eval_retries": 0}

# 🧠 ROUTER
def router_node(state):
    q = state["question"].lower()
    if "date" in q or "time" in q:
        return {"route": "tool"}
    return {"route": "retrieve"}

# 🔍 RETRIEVE
def retrieval_node(state):
    emb = embedder.encode([state["question"]]).tolist()
    res = collection.query(query_embeddings=emb, n_results=2)
    context = "\n".join(res["documents"][0])
    return {"retrieved": context}

# 🛠️ TOOL
def tool_node(state):
    now = datetime.now()
    return {"tool_result": now.strftime("%A, %d %B %Y %I:%M %p")}

# 🤖 ANSWER
def answer_node(state):
    prompt = f'''
Answer ONLY from context.
If not found, say "I don't know".

Context:
{state.get("retrieved","")}

Tool:
{state.get("tool_result","")}

Question:
{state["question"]}
'''
    res = llm.invoke(prompt)
    return {"answer": res.content}

# ✅ EVAL (simplified)
def eval_node(state):
    return {"faithfulness": 1.0, "eval_retries": 1}

# 🔗 GRAPH
graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")

def route_decision(state):
    return state["route"]

graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool"
})

graph.add_edge("retrieve", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_edge("eval", END)

app = graph.compile(checkpointer=MemorySaver())
