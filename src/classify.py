"""Supporting experiment — "law router": predict WHICH LAW an article belongs to.

Why: (1) it gives a supervised task with real labels (the law each article was
published in — no manual labeling, nothing generated), so we can report
accuracy / precision / recall / F1 / confusion matrix on held-out data;
(2) in the RAG system a router could restrict search to the predicted law.

Evaluation protocol — NESTED stratified cross-validation
- Outer loop: 5 stratified folds. Every article is a test item exactly once,
  so results are reported as mean ± std over 5 folds (not one lucky split),
  and the confusion matrix uses out-of-fold predictions for ALL 1,474 articles.
- Inner loop: 3 stratified folds inside each outer TRAINING part, used only to
  choose hyperparameters (GridSearchCV). The outer test fold is never seen
  while tuning → no leakage.
- Input is the article BODY only (no law name, no chapter heading); every
  vectorizer / FastText model is refit inside each training fold.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

from . import config
from .normalize import tokenize

OUTER_FOLDS, INNER_FOLDS = 5, 3


def _inner_cv():
    return StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=config.SEED)


def _tok(s):          # module-level so pipelines stay picklable
    return tokenize(s)


# ------------------------------------------------------------------ models (each = GridSearchCV over its own params)
def tfidf_logreg():
    pipe = make_pipeline(
        TfidfVectorizer(tokenizer=_tok, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    return GridSearchCV(pipe, {"logisticregression__C": [1, 10, 100]}, cv=_inner_cv(), scoring="f1_macro", n_jobs=1)


def tfidf_nb():
    # Complement NB is the NB variant designed for imbalanced text classes
    pipe = make_pipeline(
        TfidfVectorizer(tokenizer=_tok, token_pattern=None, ngram_range=(1, 2), min_df=2),
        ComplementNB(),
    )
    return GridSearchCV(pipe, {"complementnb__alpha": [0.01, 0.1, 0.5, 1.0]}, cv=_inner_cv(),
                        scoring="f1_macro", n_jobs=1)


class FastTextFeatures:
    """sklearn-style transformer: FastText trained on the TRAIN texts only, mean-pooled."""

    def __init__(self, dim=200, epochs=20):
        self.dim, self.epochs = dim, epochs

    def fit(self, X, y=None):
        from gensim.models import FastText
        docs = [tokenize(t) for t in X]
        self.m = FastText(sentences=docs, vector_size=self.dim, window=5, min_count=2, min_n=2, max_n=5,
                          sg=1, epochs=self.epochs, seed=config.SEED, workers=1)
        return self

    def transform(self, X):
        out = []
        for t in X:
            toks = tokenize(t)
            v = self.m.wv[toks].mean(0) if toks else np.zeros(self.dim)
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.vstack(out)

    def get_params(self, deep=True):
        return {"dim": self.dim, "epochs": self.epochs}

    def set_params(self, **p):
        self.__dict__.update(p)
        return self


def fasttext_logreg():
    pipe = make_pipeline(FastTextFeatures(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return GridSearchCV(pipe, {"logisticregression__C": [1, 10]}, cv=_inner_cv(), scoring="f1_macro", n_jobs=1)


def bge_logreg():
    """Frozen bge-m3 sentence vectors (pretrained transformer) + logistic regression."""
    return GridSearchCV(LogisticRegression(max_iter=3000, class_weight="balanced"),
                        {"C": [1, 10, 100]}, cv=_inner_cv(), scoring="f1_macro", n_jobs=1)


def encode_bge(texts, cache_name="bge_m3_body.npy"):
    """bge-m3 vectors of article bodies (no law name → no label leakage). Cached."""
    import hashlib
    import json
    from .normalize import normalize
    from .retriever import DenseRetriever
    cache = config.INDEX_DIR / cache_name
    key = cache.with_suffix(".key.json")
    texts = list(texts)
    sig = [len(texts), hashlib.md5("\n".join(texts).encode()).hexdigest()]   # stable across runs
    if cache.exists() and key.exists() and json.loads(key.read_text()) == sig:
        return np.load(cache)
    m = DenseRetriever.load_model()
    E = m.encode([normalize(t) for t in texts], batch_size=16, normalize_embeddings=True,
                 show_progress_bar=True).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    key.write_text(json.dumps(sig))
    return E


# ------------------------------------------------------------------ nested CV
def nested_cv(name, make_model, X, y, labels, verbose=True):
    """Outer stratified 5-fold; `make_model()` returns a fresh GridSearchCV (the inner loop)."""
    X = X if isinstance(X, np.ndarray) else np.asarray(list(X), dtype=object)
    y = np.asarray(y)
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.empty(len(y), dtype=object)
    folds, chosen = [], []
    for k, (tr, te) in enumerate(outer.split(X, y), 1):
        model = make_model()
        model.fit(X[tr], y[tr])                       # inner CV tunes params on the training part only
        pred = model.predict(X[te])
        oof[te] = pred
        folds.append({"fold": k, "accuracy": accuracy_score(y[te], pred),
                      "macro_F1": f1_score(y[te], pred, average="macro"),
                      "weighted_F1": f1_score(y[te], pred, average="weighted")})
        chosen.append(str(model.best_params_))
        if verbose:
            print(f"  {name} fold {k}: macro-F1 {folds[-1]['macro_F1']:.3f}  {model.best_params_}")
    f = pd.DataFrame(folds)
    return {
        "model": name,
        "accuracy": f"{f.accuracy.mean():.3f} ± {f.accuracy.std():.3f}",
        "macro_F1": f"{f.macro_F1.mean():.3f} ± {f.macro_F1.std():.3f}",
        "weighted_F1": f"{f.weighted_F1.mean():.3f} ± {f.weighted_F1.std():.3f}",
        "macro_F1_mean": f.macro_F1.mean(),
        "params (most chosen)": Counter(chosen).most_common(1)[0][0],
        "folds": f,
        "report": classification_report(y, oof, labels=labels, zero_division=0, output_dict=True),
        "confusion": confusion_matrix(y, oof, labels=labels),
        "pred": oof,
    }
