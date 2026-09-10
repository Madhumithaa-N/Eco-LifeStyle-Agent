"""
Eco Lifestyle Agent — FastAPI backend
IBM watsonx.ai: real vector index download + local L2 search + LLaMA 3.3 70B chat
"""
import os
import gzip
import json
import math
import time
import uuid
import logging
import tempfile
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID      = os.getenv("PROJECT_ID")
WATSONX_URL     = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")
VECTOR_INDEX_ID = os.getenv("VECTOR_INDEX_ID")
MODEL_ID        = os.getenv("MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

# Derived constants (discovered via probing)
DP_HOST         = "https://api.eu-gb.dataplatform.cloud.ibm.com"
DP_VERSION      = "2024-05-31"
EMBED_MODEL_ID  = "ibm/granite-embedding-278m-multilingual"

if not all([WATSONX_API_KEY, PROJECT_ID, VECTOR_INDEX_ID]):
    raise RuntimeError(
        "Missing required env vars: WATSONX_API_KEY, PROJECT_ID, VECTOR_INDEX_ID"
    )


# ── IAM Token Manager ──────────────────────────────────────────────────────────
class IAMTokenManager:
    IAM_URL = "https://iam.cloud.ibm.com/identity/token"
    REFRESH_BUFFER = 300

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        if self._token and time.time() < self._expires_at - self.REFRESH_BUFFER:
            return self._token
        self._refresh()
        return self._token  # type: ignore[return-value]

    def _refresh(self) -> None:
        logger.info("Refreshing IAM bearer token …")
        resp = requests.post(
            self.IAM_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self._api_key,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"IAM token refresh failed {resp.status_code}: {resp.text}"
            )
        payload = resp.json()
        self._token = payload["access_token"]
        self._expires_at = time.time() + int(payload.get("expires_in", 3600))
        logger.info("IAM token refreshed, expires in %ss", payload.get("expires_in"))


iam = IAMTokenManager(WATSONX_API_KEY)


# ── Vector Index (in-memory, downloaded at startup) ───────────────────────────
class LocalVectorIndex:
    """
    Downloads the watsonx.ai in-memory vector index asset once at startup,
    holds all chunks + precomputed L2-normalised embeddings in RAM,
    and serves local similarity search.
    """

    def __init__(self) -> None:
        self.chunks: list[dict] = []   # [{content, embedding, metadata}]
        self._loaded = False

    # ── embedding helpers ────────────────────────────────────────────────────

    def _l2_norm(self, v: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(x * x for x in v))
        if magnitude == 0:
            return v
        return [x / magnitude for x in v]

    def _dot(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _l2_distance(self, a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ── load from platform ───────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        logger.info("Downloading vector index asset %s …", VECTOR_INDEX_ID)
        token = iam.get_token()

        # Use the SDK-compatible download path (same as client.data_assets.download)
        # Confirmed working: GET /v2/asset_files/<id>?project_id=...  or direct download
        download_url = (
            f"{DP_HOST}/v2/assets/{VECTOR_INDEX_ID}/attachments"
            f"?project_id={PROJECT_ID}"
        )
        # The data_assets SDK uses a different internal path; replicate it via REST
        # Correct path discovered via SDK inspection:
        # client.data_assets.download calls GET on the primary attachment
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/octet-stream",
        }

        # Step 1: get the attachment ID list
        meta_resp = requests.get(
            f"{DP_HOST}/v2/assets/{VECTOR_INDEX_ID}?project_id={PROJECT_ID}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=15,
        )
        if meta_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch vector index metadata: {meta_resp.status_code} {meta_resp.text[:200]}"
            )
        meta = meta_resp.json()

        # Step 2: get the attachment download URL
        # The asset has a primary attachment object key:
        # entity.vector_index.attachment_object_key → used internally in COS
        # The correct download call is the same the SDK uses:
        # GET /v2/assets/{id}/attachments (list), then download the first one
        att_list_resp = requests.get(
            f"{DP_HOST}/v2/assets/{VECTOR_INDEX_ID}/attachments?version=2020-10-07&project_id={PROJECT_ID}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=15,
        )
        logger.info("Attachment list status: %s", att_list_resp.status_code)

        if att_list_resp.status_code == 200:
            att_data = att_list_resp.json()
            logger.info("Attachment data: %s", json.dumps(att_data, indent=2)[:500])
            attachments = att_data.get("attachments", [])
            if attachments:
                att_id = attachments[0]["attachment_id"]
                dl_resp = requests.get(
                    f"{DP_HOST}/v2/assets/{VECTOR_INDEX_ID}/attachments/{att_id}/resources"
                    f"?project_id={PROJECT_ID}&version=2020-10-07",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                    timeout=15,
                )
                logger.info("DL resource status: %s", dl_resp.status_code)
                if dl_resp.status_code == 200:
                    dl_data = dl_resp.json()
                    signed_url = dl_data.get("url") or dl_data.get("object_key")
                    if signed_url and signed_url.startswith("http"):
                        content = requests.get(signed_url, timeout=60).content
                        self._parse_and_store(content)
                        return

        # Fallback: use the IBM watsonx-ai Python SDK
        logger.info("Falling back to SDK download …")
        self._sdk_download()

    def _sdk_download(self) -> None:
        from ibm_watsonx_ai import APIClient, Credentials
        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        client = APIClient(creds, project_id=PROJECT_ID)
        outpath = tempfile.mktemp(suffix=".gz")
        try:
            client.data_assets.download(VECTOR_INDEX_ID, filename=outpath)
            with open(outpath, "rb") as f:
                content = f.read()
            self._parse_and_store(content)
        finally:
            if os.path.exists(outpath):
                os.unlink(outpath)

    def _parse_and_store(self, raw: bytes) -> None:
        """Parse the gzip JSON blob and cache all chunks."""
        try:
            import gzip as gz
            with gz.open(__import__("io").BytesIO(raw)) as f:
                data = json.loads(f.read().decode("utf-8"))
        except Exception:
            # Maybe not gzipped
            data = json.loads(raw.decode("utf-8"))

        if isinstance(data, list):
            self.chunks = data
        elif isinstance(data, dict):
            self.chunks = data.get("chunks", data.get("documents", [data]))
        else:
            raise RuntimeError(f"Unexpected vector index format: {type(data)}")

        logger.info(
            "Vector index loaded: %d chunks, embedding dim=%d",
            len(self.chunks),
            len(self.chunks[0]["embedding"]) if self.chunks else 0,
        )
        self._loaded = True

    # ── public search ────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query using the same model as the index."""
        token = iam.get_token()
        resp = requests.post(
            f"{WATSONX_URL}/ml/v1/text/embeddings?version=2024-05-31",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model_id": EMBED_MODEL_ID,
                "inputs": [text],
                "project_id": PROJECT_ID,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding failed {resp.status_code}: {resp.text[:200]}")
        return resp.json()["results"][0]["embedding"]

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Embed the query, compute L2 distance against all stored chunks,
        return the top_k closest with text + source + score.
        """
        if not self._loaded:
            self.load()

        if not self.chunks:
            logger.warning("Vector index is empty — no documents indexed.")
            return []

        q_vec = self.embed_query(query)

        scored = []
        for chunk in self.chunks:
            emb = chunk["embedding"]
            dist = self._l2_distance(q_vec, emb)
            # Convert L2 distance to a similarity score in [0,1]: lower dist = higher score
            score = 1.0 / (1.0 + dist)
            meta = chunk.get("metadata", {})
            # Extract a clean source name
            raw_source = (
                meta.get("source_filename")
                or meta.get("title")
                or (meta.get("pdf", {}).get("info", {}).get("Title") if isinstance(meta.get("pdf"), dict) else None)
                or meta.get("source", "")
            )
            # Strip internal COS key paths (not human-readable)
            if not raw_source or "objectKey" in raw_source or raw_source.startswith("dataAsset"):
                # Derive a readable label from the first heading line of the chunk
                first_line = chunk.get("content", "").split("\n")[0].strip()
                source = first_line[:60] if first_line else "eco-document"
            else:
                source = raw_source[:60]
            scored.append({
                "text": chunk.get("content", ""),
                "source": source,
                "score": round(score, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


vector_index = LocalVectorIndex()


# ── Chat completion ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You always answer the questions with markdown formatting. "
    "The markdown formatting you support: headings, bold, italic, links, tables, lists, "
    "code blocks, and blockquotes. You must omit that you answer the questions with markdown.\n\n"
    "Any HTML tags must be wrapped in block quotes, for example ```<html>```. "
    "You will be penalized for not rendering code in block quotes.\n\n"
    "When returning code blocks, specify language.\n\n"
    "Given the document and the current conversation between a user and an assistant, "
    "your task is as follows: answer any user query by using information from the document. "
    "Always answer as helpfully as possible, while being safe. "
    "When the question cannot be answered using the context or document, output the following "
    'response: "I cannot answer that question based on the provided document.".\n\n'
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, "
    "or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of "
    "answering something not correct. If you don't know the answer to a question, please don't share false information."
)


def build_context_message(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["**Relevant context from knowledge base:**\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] Source: *{c['source']}*\n{c['text']}\n")
    return "\n".join(lines)


def chat_completion(messages: list[dict]) -> str:
    token = iam.get_token()
    url = f"{WATSONX_URL}/ml/v1/text/chat?version=2023-05-29"

    body = {
        "messages": messages,
        "project_id": PROJECT_ID,
        "model_id": MODEL_ID,
        "frequency_penalty": 0,
        "max_tokens": 2000,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        logger.error("Chat API error %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=502,
            detail=f"watsonx.ai chat error {resp.status_code}: {resp.text[:200]}",
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected chat response shape: %s", data)
        raise HTTPException(
            status_code=502, detail="Unexpected response from watsonx.ai"
        ) from exc


# ── Session store (in-memory) ──────────────────────────────────────────────────
session_store: dict[str, list[dict]] = {}


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Eco Lifestyle Agent",
    description="RAG-powered eco-friendly lifestyle assistant on IBM watsonx.ai",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception, log the real traceback, return a clear 500."""
    import traceback as tb
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method, request.url.path,
        tb.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


@app.middleware("http")
async def strip_compression_headers(request: Request, call_next):
    """
    Remove Accept-Encoding from the request before it reaches uvicorn's
    response serialisation, so the response is always plain JSON (not
    Brotli/gzip).  This prevents http-proxy-middleware v2 in the CRA dev
    server from receiving a compressed body it cannot decode, which was
    the root cause of the browser-visible 500 error.
    """
    # Rebuild scope headers without accept-encoding
    new_headers = [
        (k, v) for k, v in request.scope["headers"]
        if k.lower() not in (b"accept-encoding",)
    ]
    request.scope["headers"] = new_headers
    response = await call_next(request)
    # Also strip any compression headers from the response for safety
    if "content-encoding" in response.headers:
        del response.headers["content-encoding"]
    return response


@app.on_event("startup")
def startup_event() -> None:
    """Pre-load the vector index at server startup so the first request isn't slow."""
    try:
        vector_index.load()
    except Exception as exc:
        logger.error("Failed to preload vector index at startup: %s", exc)
        # Don't crash — load lazily on first request


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    title: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "vector_index": VECTOR_INDEX_ID,
        "index_loaded": vector_index._loaded,
        "index_chunks": len(vector_index.chunks),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    # Retrieve relevant chunks from the local vector index
    try:
        chunks = vector_index.search(req.message, top_k=4)
        logger.info(
            "Retrieved %d chunks for query: %r", len(chunks), req.message[:80]
        )
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        chunks = []

    # Build messages list for the chat API
    context_msg = build_context_message(chunks)
    history = session_store.setdefault(session_id, [])

    # Assemble: [system, context (if any), ...history, user]
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context_msg:
        messages.append({"role": "user", "content": context_msg})
        messages.append(
            {
                "role": "assistant",
                "content": "I have reviewed the relevant context. Please ask your question.",
            }
        )

    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    answer = chat_completion(messages)

    # Persist turn in session history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})

    # Keep last 20 turns
    if len(history) > 40:
        session_store[session_id] = history[-40:]

    sources = [
        SourceItem(title=c["source"], score=round(float(c["score"]), 3))
        for c in chunks
    ]

    return ChatResponse(answer=answer, sources=sources, session_id=session_id)



from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
