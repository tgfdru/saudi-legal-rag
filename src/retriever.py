"""Retrieval models compared in the project.

| name        | representation                      | course topic                  |
|-------------|-------------------------------------|-------------------------------|
| tfidf       | TF-IDF word uni+bi-grams, cosine    | Bag-of-Words / TF-IDF         |
| bm25        | Okapi BM25 over the same tokens     | probabilistic BoW ranking     |
| fasttext    | FastText trained on the corpus,     | word embeddings               |
|             | SIF-style weighted mean of vectors  |                               |
| dense       | bge-m3 contextual sentence vectors  | transformers (pretrained)     |
| hybrid      | weighted Reciprocal Rank Fusion of  | ensembling sparse + dense     |
|             | bm25 + dense                        |                               |
| +rerank     | bge-reranker-v2-m3 cross-encoder    | transformers (cross-encoder)  |

Shared interface:  retriever.search(query, k) -> list[(chunk_index, score)]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .chunker import Chunk, load_corpus
from .normalize import normalize, tokenize


def _topk(scores: np.ndarray, k: int):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


# ------------------------------------------------------------------ TF-IDF
class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # tokens are already normalized + stemmed; analyzer=str.split keeps them as-is
        self.vec = TfidfVectorizer(analyzer=lambda s: _ngrams(s.split(), (1, 2)),
                                   sublinear_tf=True, min_df=1)
        self.X = self.vec.fit_transform([" ".join(tokenize(c.embed_text())) for c in chunks])

    def search(self, query: str, k: int = config.CANDIDATES):
        q = self.vec.transform([" ".join(tokenize(query))])
        return _topk((self.X @ q.T).toarray().ravel(), k)


def _ngrams(toks, rng):
    out = []
    for n in range(rng[0], rng[1] + 1):
        out += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return out


# ------------------------------------------------------------------ BM25
class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = config.CANDIDATES):
        return _topk(np.asarray(self.bm25.get_scores(tokenize(query))), k)


# ------------------------------------------------------------------ FastText
class FastTextRetriever:
    """Word embeddings trained on our own corpus (gensim FastText).

    FastText (not plain Word2Vec) because Arabic is morphologically rich:
    character n-grams give vectors to unseen inflections in user questions.
    Document vector = IDF-weighted mean of word vectors (a simple SIF variant).
    """
    name = "fasttext"

    def __init__(self, chunks: list[Chunk], dim: int = 200, epochs: int = 30, seed: int = 42):
        from gensim.models import FastText
        docs = [tokenize(c.embed_text()) for c in chunks]
        self.model = FastText(sentences=docs, vector_size=dim, window=5, min_count=2,
                              min_n=2, max_n=5, sg=1, epochs=epochs, seed=seed, workers=1)
        df = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: np.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.default_idf = np.log(n + 1) + 1
        self.D = np.vstack([self._embed(d) for d in docs])

    def _embed(self, toks):
        if not toks:
            return np.zeros(self.model.vector_size, dtype=np.float32)
        w = np.array([self.idf.get(t, self.default_idf) for t in toks])
        v = (self.model.wv[toks] * w[:, None]).sum(0) / w.sum()
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, query: str, k: int = config.CANDIDATES):
        return _topk(self.D @ self._embed(tokenize(query)), k)


# ------------------------------------------------------------------ dense (bge-m3)
class DenseRetriever:
    name = "dense"
    _model = None  # shared, heavy

    def __init__(self, chunks: list[Chunk], cache: Path = config.INDEX_DIR / "bge_m3.npy"):
        self.model = self.load_model()
        ids_file = cache.with_suffix(".ids.json")
        ids = [c.id for c in chunks]
        if cache.exists() and ids_file.exists() and json.loads(ids_file.read_text()) == ids:
            self.emb = np.load(cache)                         # reuse cached vectors
        else:
            self.emb = self.model.encode(
                [normalize(c.embed_text()) for c in chunks], batch_size=16,
                normalize_embeddings=True, show_progress_bar=True,
            ).astype(np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)
            ids_file.write_text(json.dumps(ids))

    @classmethod
    def load_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(config.EMBED_MODEL)
            cls._model.max_seq_length = 512
        return cls._model

    def search(self, query: str, k: int = config.CANDIDATES):
        q = self.model.encode([normalize(query)], normalize_embeddings=True)[0]
        return _topk(self.emb @ q, k)


# ------------------------------------------------------------------ hybrid
class HybridRetriever:
    """Weighted Reciprocal Rank Fusion:  score(d) = Σ_r  w_r / (RRF_K + rank_r(d)).

    Rank-based, so BM25 and cosine scores never need calibration.
    The dense weight `w` is a hyperparameter tuned on the DEV questions only.
    """

    def __init__(self, sparse, dense, w_dense: float = 0.5):
        self.sparse, self.dense, self.w = sparse, dense, w_dense
        self.name = "hybrid"

    def search(self, query: str, k: int = config.CANDIDATES):
        fused: dict[int, float] = {}
        for r, w in ((self.sparse, 1 - self.w), (self.dense, self.w)):
            for rank, (idx, _) in enumerate(r.search(query, config.CANDIDATES), start=1):
                fused[idx] = fused.get(idx, 0.0) + w / (config.RRF_K + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


# ------------------------------------------------------------------ rerank
class RerankedRetriever:
    """Cross-encoder reads (question, article) jointly and re-scores the top candidates."""
    _model = None

    def __init__(self, base, chunks: list[Chunk], n_candidates: int = config.RERANK_CANDIDATES):
        self.base, self.chunks, self.n = base, chunks, n_candidates
        self.name = f"{base.name}+rerank"
        if RerankedRetriever._model is None:
            from sentence_transformers import CrossEncoder
            RerankedRetriever._model = CrossEncoder(config.RERANK_MODEL, max_length=512)
        self.model = RerankedRetriever._model
        self._cache: dict[tuple[str, int], float] = {}   # (query, article) → score; reused across CV folds

    def search(self, query: str, k: int = config.TOP_K):
        cands = self.base.search(query, self.n)
        todo = [i for i, _ in cands if (query, i) not in self._cache]
        if todo:
            new = self.model.predict([(query, self.chunks[i].embed_text()) for i in todo])
            self._cache.update({(query, i): float(s) for i, s in zip(todo, new)})
        scores = np.array([self._cache[(query, i)] for i, _ in cands])
        order = np.argsort(-scores)
        reranked = [(cands[j][0], float(scores[j])) for j in order]
        tail = [c for c in self.base.search(query, max(k, self.n))][self.n:]   # keep ranks beyond n
        return (reranked + tail)[:k]


# ------------------------------------------------------------------ factory
def build_retrievers(chunks: list[Chunk] | None = None, use_dense: bool = True,
                     use_rerank: bool = config.USE_RERANKER, w_dense: float = config.W_DENSE):
    """Returns (chunks, {name: retriever}) so evaluation can compare all of them."""
    chunks = chunks or load_corpus(config.RAW_DIR)
    out = {"tfidf": TfidfRetriever(chunks), "bm25": BM25Retriever(chunks),
           "fasttext": FastTextRetriever(chunks)}
    if use_dense:
        out["dense"] = DenseRetriever(chunks)
        out["hybrid"] = HybridRetriever(out["bm25"], out["dense"], w_dense)
        if use_rerank:
            out["hybrid+rerank"] = RerankedRetriever(out["hybrid"], chunks)
    return chunks, out
