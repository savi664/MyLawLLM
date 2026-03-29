import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from rank_bm25 import BM25Okapi
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


# ── Config ────────────────────────────────────────────────────
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # for GitHub Models / Azure inference endpoint

EMBED_MODEL = "all-MiniLM-L6-v2"
GPT_MODEL = "gpt-4o"

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_docs")

DENSE_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 5


def require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


QDRANT_API_KEY = require_env("QDRANT_API_KEY", QDRANT_API_KEY)
QDRANT_URL = require_env("QDRANT_URL", QDRANT_URL)
GITHUB_TOKEN = require_env("GITHUB_TOKEN", GITHUB_TOKEN)


# ── Startup resources ─────────────────────────────────────────
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

print("Connecting to Qdrant Cloud...")
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

db = Qdrant(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embeddings=embeddings,
)

print("Fetching documents for BM25 index...")
try:
    store = db.get(include=["documents", "metadatas"])
except TypeError:
    store = db.get(include=["documents"])

all_texts = store.get("documents", []) or []
all_metadata = store.get("metadatas", []) or [{"source": "Unknown"}] * len(all_texts)

if not all_texts:
    raise RuntimeError(
        f"No documents found in Qdrant collection '{QDRANT_COLLECTION}'."
    )

if len(all_metadata) != len(all_texts):
    all_metadata = [{"source": "Unknown"}] * len(all_texts)

print("Building BM25 index...")
bm25 = BM25Okapi([text.lower().split() for text in all_texts])

print(f"Ready - {len(all_texts):,} chunks loaded")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)


# ── Hybrid search ─────────────────────────────────────────────
def search(query: str) -> List[dict[str, Any]]:
    dense = db.similarity_search_with_score(query, k=DENSE_TOP_K)

    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:BM25_TOP_K]

    pool: dict[str, dict[str, Any]] = {}

    for doc, score in dense:
        content = doc.page_content
        pool[content] = {
            "content": content,
            "source": doc.metadata.get("source", "Unknown"),
            "dense_score": float(score),
            "bm25_score": 0.0,
        }

    for i in sparse_idx:
        content = all_texts[i]
        metadata = all_metadata[i] if i < len(all_metadata) else {"source": "Unknown"}

        if content in pool:
            pool[content]["bm25_score"] = float(bm25_scores[i])
        else:
            pool[content] = {
                "content": content,
                "source": metadata.get("source", "Unknown")
                if isinstance(metadata, dict)
                else "Unknown",
                "dense_score": 0.0,
                "bm25_score": float(bm25_scores[i]),
            }

    chunks = list(pool.values())
    if not chunks:
        return []

    max_dense = max((c["dense_score"] for c in chunks), default=1.0) or 1.0
    max_bm25 = max((c["bm25_score"] for c in chunks), default=1.0) or 1.0

    for c in chunks:
        dense_component = 1 - (c["dense_score"] / max_dense) if max_dense else 0.0
        bm25_component = c["bm25_score"] / max_bm25 if max_bm25 else 0.0
        c["score"] = 0.5 * dense_component + 0.5 * bm25_component

    return sorted(chunks, key=lambda c: c["score"], reverse=True)[:FINAL_TOP_K]


# ── System prompt ─────────────────────────────────────────────
SYSTEM = """You are MyLawLLM — a legal assistant that knows Sri Lankan law inside out.

Keep it human. Talk like a knowledgeable friend who happens to be a lawyer, not a textbook.
Be direct, clear, and concise. No waffle.

IMPORTANT: Structure your response EXACTLY like this:

**Plain-English Answer:**
[Give a clear, concise answer in normal language that anyone can understand. Address the user's question directly.]

**Legal Basis:**
[List the specific Acts, Ordinances, or Laws and their sections that apply. Use bold for Act names like **Consumer Protection Act** or section 23 of the relevant law. Be precise about which section supports each point.]

Rules:
- Only use the legal text given to you. Don't make things up.
- Always bold the Act name and section number where applicable.
- If the context doesn't cover it, just say so honestly.
"""


# ── API ───────────────────────────────────────────────────────
app = FastAPI(title="MyLawLLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class Query(BaseModel):
    question: str
    history: List[Message] = Field(default_factory=list)


@app.post("/ask")
def ask(req: Query):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = search(question)
    if not chunks:
        return {
            "answer": "I couldn't find relevant legal material in the current knowledge base for that question.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[{i + 1}. {c['source']}]\n{c['content']}" for i, c in enumerate(chunks)
    )

    safe_history = [{"role": m.role, "content": m.content} for m in req.history[-6:]]

    messages = [{"role": "system", "content": SYSTEM}] + safe_history + [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    ]

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.2,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")

    answer = response.choices[0].message.content or "No response generated."

    return {
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "excerpt": c["content"][:300],
            }
            for c in chunks
        ],
    }


# ── Frontend ──────────────────────────────────────────────────
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/")
def root():
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "MyLawLLM API is running"}