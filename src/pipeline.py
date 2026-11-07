"""End-to-end RAG pipeline: query -> retrieve -> generate -> answer + sources."""
from __future__ import annotations

import time
from functools import lru_cache

from . import config
from .generator import generate, DISCLAIMER
from .retriever import build_retrievers


class LegalRAG:
    def __init__(self, mode: str | None = None, use_dense: bool = True, chunks=None, retrievers=None):
        if retrievers is None:                      # reuse already-built retrievers (e.g. in the notebook)
            chunks, retrievers = build_retrievers(use_dense=use_dense)
        self.chunks, self.retrievers = chunks, retrievers
        # Best available retriever by default
        self.mode = mode or next(
            m for m in ("hybrid+rerank", "hybrid", "dense", "bm25") if m in self.retrievers
        )

    def ask(self, question: str, k: int = config.TOP_K, mode: str | None = None) -> dict:
        t0 = time.time()
        retriever = self.retrievers[mode or self.mode]
        hits = retriever.search(question, k)
        docs = [self.chunks[i] for i, _ in hits]
        out = generate(question, docs)
        return {
            "question": question,
            "answer": out["answer"],
            "mode": out["mode"],
            "retriever": retriever.name,
            "disclaimer": DISCLAIMER,
            "latency_s": round(time.time() - t0, 2),
            "sources": [
                {"rank": r, "citation": c.citation, "chapter": c.chapter, "score": round(s, 4),
                 "text": c.text, "url": c.source, "issued": c.issued}
                for r, (c, (_, s)) in enumerate(zip(docs, hits), start=1)
            ],
        }


@lru_cache(maxsize=1)
def get_rag() -> LegalRAG:
    """Singleton for the API / UI (models load once)."""
    return LegalRAG()


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "ما مدة التظلم من القرار الإداري قبل رفع دعوى الإلغاء؟"
    res = get_rag().ask(q)
    print(res["answer"], "\n")
    for s in res["sources"]:
        print(f"  [{s['rank']}] {s['citation']}  (score={s['score']})")
