# 🌿 Eco Lifestyle Agent

A RAG-powered eco-friendly lifestyle assistant built with **IBM watsonx.ai**, **FastAPI**, and **React + Tailwind CSS**.

Ask questions about sustainable living, zero-waste practices, green home tips, ethical shopping, and more. Answers are retrieved directly from your IBM watsonx.ai VectorStore knowledge base using the LLaMA 3.3 70B Instruct model.

---

## Architecture

```
frontend/  (React + Tailwind)
    └─ src/
        ├─ App.jsx               # Main chat UI
        ├─ components/
        │   ├─ ChatMessage.jsx   # User/assistant bubble with Markdown + source pills
        │   ├─ TypingIndicator.jsx
        │   └─ SuggestedQuestions.jsx
        └─ index.css             # Tailwind + custom nature-inspired styles

backend/
    └─ main.py                   # FastAPI app
        ├─ IAMTokenManager       # IBM Cloud IAM bearer-token with auto-refresh
        ├─ retrieve_chunks()     # IBM watsonx.ai VectorStore search (top 4)
        ├─ chat_completion()     # LLaMA 3.3 70B via /ml/v1/text/chat
        ├─ POST /chat            # → { answer, sources, session_id }
        └─ GET  /health          # → { status, model, vector_index }
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- IBM watsonx.ai account with:
  - API key
  - Project ID
  - Vector Index ID (pre-populated with eco-lifestyle documents)

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your real credentials
```

### 2. Start the backend

```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

Health check: http://localhost:8000/health

### 3. Start the frontend

```bash
cd frontend
npm install
npm start
```

App opens at: http://localhost:3000

---

## Environment Variables

| Variable          | Description                                    |
|-------------------|------------------------------------------------|
| `WATSONX_API_KEY` | IBM Cloud API key for IAM authentication       |
| `PROJECT_ID`      | watsonx.ai Project ID                          |
| `WATSONX_URL`     | watsonx.ai regional endpoint (eu-gb by default)|
| `VECTOR_INDEX_ID` | ID of your watsonx.ai Vector Index             |
| `MODEL_ID`        | LLM model ID (default: llama-3-3-70b-instruct) |

---

## API Reference

### `POST /chat`

**Request:**
```json
{ "message": "How do I reduce plastic waste?", "session_id": "optional-uuid" }
```

**Response:**
```json
{
  "answer": "## Reducing Plastic Waste\n...",
  "sources": [
    { "title": "eco-living-guide.pdf", "score": 0.87 }
  ],
  "session_id": "abc-123"
}
```

### `GET /health`

```json
{ "status": "ok", "model": "meta-llama/llama-3-3-70b-instruct", "vector_index": "..." }
```

---

## Features

- **Real RAG** — top-4 chunk retrieval from IBM watsonx.ai VectorStore
- **IAM token refresh** — automatic bearer-token renewal 5 min before expiry
- **Session history** — multi-turn conversation memory per `session_id`
- **Fallback message** — "I cannot answer that question based on the provided document." when context is missing
- **Nature-inspired UI** — soft greens, warm neutrals, rounded cards
- **Markdown rendering** — headings, bold, italic, tables, code blocks, blockquotes
- **Source pills** — clickable document attribution badges
- **Suggested questions** — one-click conversation starters
- **Typing indicator** — animated three-dot bounce

---

## Project Structure

```
.
├── .env                  # Secrets (never commit)
├── .env.example          # Template for new developers
├── .gitignore
├── requirements.txt
├── README.md
├── backend/
│   ├── main.py
│   └── requirements.txt
└── frontend/
    ├── package.json
    ├── tailwind.config.js
    └── src/
```
