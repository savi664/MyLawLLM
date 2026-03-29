MyLawLLM ⚖️

A practical legal assistant for Sri Lankan law that combines dense retrieval, BM25 search, and LLM reasoning to deliver clear, grounded answers with proper legal references.

What this is

MyLawLLM is not just another chatbot. It is built to actually reason over real legal text and explain it in a way that makes sense to normal people while still pointing back to the law.

You ask a question.
It finds the most relevant legal sections.
Then it answers in plain English and backs it up with the actual Acts and sections.

Core Features
Hybrid Retrieval (Dense + BM25)
Grounded answers using real legal documents
Clear dual-output format:
Plain-English explanation
Legal basis with Acts and sections
FastAPI backend with simple frontend
Cloud-based vector DB (Qdrant)
Optimized for real-world legal queries in Sri Lanka
How it works
Your query goes through hybrid search:
Dense retrieval (embeddings via MiniLM)
Sparse retrieval (BM25)
Results are merged and ranked
Top chunks are sent as context to the LLM
The LLM generates a structured response using only that context
Tech Stack
FastAPI
Uvicorn
Qdrant (Vector Database)
HuggingFace Embeddings (MiniLM)
BM25 (rank-bm25)
OpenAI-compatible API (GitHub Models / Azure endpoint)
Vanilla JS frontend
