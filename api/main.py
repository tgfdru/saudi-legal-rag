"""FastAPI service.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline import get_rag

app = FastAPI(title="Saudi Legal RAG API", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["ما مدة التظلم من القرار الإداري؟"])
    k: int = Field(5, ge=1, le=10)
    retriever: str | None = Field(None, description="bm25 | dense | hybrid | hybrid+rerank")


@app.on_event("startup")
def _warmup():
    get_rag()  # load models once


@app.get("/health")
def health():
    rag = get_rag()
    return {"status": "ok", "chunks": len(rag.chunks), "retrievers": list(rag.retrievers)}


@app.post("/ask")
def ask(req: AskRequest):
    return get_rag().ask(req.question, k=req.k, mode=req.retriever)
