**# Production RAG Platform

Production-ready Retrieval-Augmented Generation (RAG) platform built with FastAPI, Qdrant, Hybrid Search (Vector + BM25), Redis Conversation Memory, Groq LLM, and AI Reranking.

---

# Features

* FastAPI-based AI backend
* Qdrant Vector Database integration
* Semantic Vector Search
* BM25 Keyword Search
* Hybrid Search Architecture
* Redis Conversation Memory
* Multi-tenant Metadata Filtering
* Payload Indexing
* AI Reranking using BGE Reranker
* Groq LLM Integration
* Async API Processing
* Production-style Retrieval Pipeline
* Advanced RAG Techniques

---

# Tech Stack

| Technology            | Purpose                     |
| --------------------- | --------------------------- |
| Python                | Backend Development         |
| FastAPI               | API Framework               |
| Qdrant                | Vector Database             |
| Redis                 | Cache & Conversation Memory |
| Groq                  | LLM Inference               |
| Sentence Transformers | Embeddings                  |
| BM25                  | Keyword Ranking             |
| BGE Reranker          | AI Reranking                |
| uv                    | Python Package Manager      |

---

# Architecture

```text id="jlwm6n"
User Query
   ↓
Embedding Generation
   ↓
Qdrant Vector Search
   ↓
BM25 Keyword Ranking
   ↓
Hybrid Score Calculation
   ↓
BGE Reranker
   ↓
Top Relevant Chunks
   ↓
Groq LLM
   ↓
Final AI Response
```

---

# Project Structure

```text id="0jlwmf"
project/
│
├── app/
│   ├── crews/
│   ├── rag/
│   ├── routes/
│   ├── services/
│   ├── utils/
│   ├── config.py
│   └── main.py
│
├── uploads/
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash id="4jlwm3"
git clone https://github.com/Pankajthapa4/production-ai-playground.git
cd production-rag-platform
```

---

# Create Virtual Environment

```bash id="8jlwmf"
uv venv
```

Activate environment:

### Windows

```bash id="3jlwm2"
.venv\Scripts\activate
```

### Linux / Mac

```bash id="7jlwm0"
source .venv/bin/activate
```

---

# Install Dependencies

```bash id="5jlwm4"
uv sync
```

Or:

```bash id="7jlwm5"
uv pip install -r requirements.txt
```

---

# Environment Variables

Create `.env`

Example:

```env id="9jlwm1"
GROQ_API_KEY=your_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile

REDIS_HOST=localhost
REDIS_PORT=6379

QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

# Run Redis

```bash id="1jlwmn"
docker run -d -p 6379:6379 redis
```

---

# Run Qdrant

```bash id="2jlwmn"
docker run -p 6333:6333 qdrant/qdrant
```

---

# Run Application

```bash id="4jlwmz"
uvicorn app.main:app --reload
```

API available at:

```text id="9jlwm7"
http://127.0.0.1:8000
```

---

# API Endpoints

## Health Check

```http id="3jlwm7"
GET /
```

---

## Redis Test

```http id="2jlwm2"
GET /redis-test
```

---

## Upload Documents

```http id="6jlwm0"
POST /upload-doc
```

---

## Search Documents

```http id="0jlwm8"
GET /search-docs?query=your_query
```

---

## RAG Chat

```http id="9jlwmq"
GET /rag-chat?query=your_query
```

---

## Get Conversation Memory

```http id="4jlwm0"
GET /memory/{session_id}
```

---

# Hybrid Search

This project implements Hybrid Search using:

* Semantic Vector Search
* BM25 Keyword Search

Hybrid score calculation:

```python id="3jlwmu"
hybrid_score = (
    vector_score * 0.7
) + (
    bm25_score * 0.3
)
```

This improves retrieval quality for:

* Emails
* Phone numbers
* IDs
* Exact keywords
* Technical terms

---

# Reranking

Uses:

```text id="1jlwmf"
BAAI/bge-reranker-base
```

Architecture:

```text id="5jlwm2"
Retrieve Many
   ↓
Rerank Intelligently
   ↓
Return Best Chunks
```

This significantly improves final retrieval precision.

---

# Production Concepts Implemented

* Vector Search
* Hybrid Search
* BM25 Ranking
* AI Reranking
* Conversation Memory
* Metadata Filtering
* Payload Indexing
* Multi-Tenant Retrieval
* Async Processing
* LLM Integration

---

# Future Improvements

* JWT Authentication
* Role-Based Access Control (RBAC)
* Streaming Responses
* Background Workers
* Docker Compose
* Kubernetes Deployment
* Observability & Monitoring
* Multi-Agent Workflows
* OCR Pipelines
* Multi-modal RAG

---

# Learning Goals

This project was built to explore real-world production AI architecture patterns including:

* Enterprise RAG Systems
* Hybrid Retrieval
* Retrieval Optimization
* AI Search Systems
* Multi-tenant AI Platforms
* Production AI Engineering

---

# Advance Rag Technique
1. Recursive Chunking

Bad chunking destroys RAG quality.

The project was upgraded from:

Fixed Chunking

to:

Recursive Character Chunking

using:

RecursiveCharacterTextSplitter

Chunking directly impacts:

retrieval quality
reranking quality
hallucinations
final LLM answers

Recursive chunking preserves:

paragraphs
sentence boundaries
semantic continuity


2. Parent-Child Retrieval (Enterprise RAG)
 Traditional RAG systems usually retrieve only small chunks for semantic similarity search.

While small chunks improve retrieval precision, they often lose surrounding context required for high-quality LLM reasoning.

This project now supports Parent-Child Retrieval Architecture.

Architecture:

User Query
   ↓
Semantic Search On Child Chunks
   ↓
Matched Child Chunk
   ↓
Load Parent Chunk
   ↓
Send Rich Context To LLM
Child Chunks

Small chunks are optimized for:

semantic retrieval
hybrid search
reranking precision
accurate matching
Parent Chunks

Large contextual chunks are optimized for:

richer LLM context
contextual reasoning
complete answers
hallucination reduction
Production Insight
retrieve small
expand large

This is a widely used enterprise RAG optimization strategy.


# License

MIT License

---

# Author
Pankaj Thapa

LinkedIn:
[LinkedIn Profile][https://in.linkedin.com/in/pankaj-thapa-0ba85055]
**
