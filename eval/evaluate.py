"""Retrieval evaluation on the question set (eval/questions.csv).

questions.csv columns
    qid        unique id (e.g. "nader-007", "ai-042")
    question   the question
    gold_ids   ';'-separated chunk ids of the governing article(s), e.g. "murafaat_diwan#008"
    author     who wrote / labeled it
    source     "ai"   – generated with an AI assistant from the article text, then checked
               "team" – written by a team member in their own words
               "real" – collected from a real person / forum

Protocol — 5-fold cross-validation over the questions
    Only one hyperparameter is tuned: the dense weight w of the hybrid fusion.
    For each fold k: choose w on the other 4 folds (max MRR@10), score fold k with it.
    Every question is therefore scored exactly once by a model that never saw it
    during tuning, and ALL questions count (no small fixed test set).
    Untuned models (tfidf, bm25, fasttext, dense) are simply scored on all questions.
    Results: mean over all questions + 95% bootstrap CI, overall and per source.

Metrics: Recall@k (k=1,3,5,10), MRR@10, nDCG@10.

CLI:  python -m eval.evaluate [--no-dense] [--no-rerank]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config

KS = (1, 3, 5, 10)
METRICS = [f"R@{k}" for k in KS] + ["MRR@10", "nDCG@10"]
N_FOLDS = 5
W_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


# ------------------------------------------------------------------ data
def load_questions(path=config.QUESTIONS_CSV, chunk_ids: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["question"].str.strip().ne("") & df["gold_ids"].str.strip().ne("")].copy()
    df["gold"] = df["gold_ids"].str.split(";").map(lambda xs: [x.strip() for x in xs if x.strip()])
    if chunk_ids is not None:                       # guard against typos / stale ids
        bad = df[~df["gold"].map(lambda g: all(x in chunk_ids for x in g))]
        if len(bad):
            print(f"⚠️  {len(bad)} question(s) reference unknown article ids → skipped: {bad.qid.tolist()}")
        df = df.drop(bad.index)
    df = df.drop_duplicates("question").reset_index(drop=True)
    df["source"] = df["source"].replace("", "team")
    # folds stratified by source so every fold has the same ai/team mix
    strat = df["source"] if df["source"].value_counts().min() >= N_FOLDS else np.zeros(len(df))
    df["fold"] = -1
    if len(df) >= N_FOLDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=config.SEED)
        for k, (_, te) in enumerate(skf.split(df, strat)):
            df.loc[te, "fold"] = k
    return df


# ------------------------------------------------------------------ metrics
def score_ranking(ranked_ids: list[str], gold: set[str]) -> dict:
    m = {f"R@{k}": float(any(c in gold for c in ranked_ids[:k])) for k in KS}
    first = next((i for i, c in enumerate(ranked_ids[:10], 1) if c in gold), None)
    m["MRR@10"] = 1.0 / first if first else 0.0
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(ranked_ids[:10], 1) if c in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    m["nDCG@10"] = dcg / idcg
    return m


def run(retriever, chunks, qs: pd.DataFrame) -> pd.DataFrame:
    """Per-question scores for one retriever (index aligned with qs)."""
    rows = []
    for q in qs.itertuples():
        ranked = [chunks[i].id for i, _ in retriever.search(q.question, 10)]
        rows.append({"qid": q.qid, "top1": ranked[0] if ranked else "", **score_ranking(ranked, set(q.gold))})
    return pd.DataFrame(rows, index=qs.index)


def bootstrap_ci(values, n=2000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ cross-validation
def crossval(retrievers: dict, chunks, qs: pd.DataFrame, grid=W_GRID, verbose=True):
    """Returns (detail: {name: per-question DataFrame}, tuning: DataFrame of w per fold)."""
    detail, tuning = {}, None
    for name, r in retrievers.items():
        if name in ("hybrid", "hybrid+rerank"):
            continue
        if verbose:
            print(f"  scoring {name} …")
        detail[name] = run(r, chunks, qs)

    if "hybrid" in retrievers:
        hy = retrievers["hybrid"]
        per_w = {}
        for w in grid:                                   # score every question once per candidate w
            hy.w = w
            per_w[w] = run(hy, chunks, qs)
        rows, parts = [], []
        for k in range(N_FOLDS):
            train, test = qs.fold != k, qs.fold == k
            best = max(grid, key=lambda w: per_w[w].loc[train, "MRR@10"].mean())
            rows.append({"fold": k + 1, "chosen w": best,
                         "train MRR@10": round(per_w[best].loc[train, "MRR@10"].mean(), 3),
                         "held-out MRR@10": round(per_w[best].loc[test, "MRR@10"].mean(), 3)})
            parts.append(per_w[best].loc[test])
        detail["hybrid"] = pd.concat(parts).sort_index()
        tuning = pd.DataFrame(rows)
        tuning.attrs["mrr_by_w"] = {w: round(d["MRR@10"].mean(), 3) for w, d in per_w.items()}

        if "hybrid+rerank" in retrievers:                # rerank on top of each fold's chosen hybrid
            rr, parts = retrievers["hybrid+rerank"], []
            for k, w in enumerate(tuning["chosen w"]):
                hy.w = w
                if verbose:
                    print(f"  reranking fold {k + 1} (w={w}) …")
                test = qs[qs.fold == k]
                parts.append(run(rr, chunks, test))
            detail["hybrid+rerank"] = pd.concat(parts).sort_index()
        hy.w = float(tuning["chosen w"].mode()[0])       # leave the most-chosen w for the demo
    return detail, tuning


def summarize(detail: dict, qs: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("all", qs.index)] if by is None else list(qs.groupby(by).groups.items())
    for g, idx in groups:
        for name, d in detail.items():
            dd = d.loc[d.index.intersection(idx)]
            lo, hi = bootstrap_ci(dd["MRR@10"])
            rows.append({by or "set": g, "retriever": name, "n": len(dd),
                         **dd[METRICS].mean().round(3).to_dict(), "MRR 95% CI": f"[{lo:.2f}, {hi:.2f}]"})
    out = pd.DataFrame(rows)
    return out.set_index("retriever") if by is None else out.set_index([by, "retriever"])


# ------------------------------------------------------------------ CLI
def main(use_dense=True, use_rerank=True):
    from src.retriever import build_retrievers
    chunks, retrievers = build_retrievers(use_dense=use_dense, use_rerank=use_rerank)
    qs = load_questions(chunk_ids={c.id for c in chunks})
    print(f"{len(qs)} questions | sources: {qs.source.value_counts().to_dict()}")
    if len(qs) < 20:
        print("Not enough questions yet — add more with:  streamlit run app/annotate.py")
        return
    detail, tuning = crossval(retrievers, chunks, qs)
    overall = summarize(detail, qs)
    print("\nALL QUESTIONS (5-fold CV)\n", overall.to_string())
    if qs.source.nunique() > 1:
        print("\nBY SOURCE\n", summarize(detail, qs, "source").to_string())
    if tuning is not None:
        print("\nHybrid w per fold\n", tuning.to_string(index=False))
    out = {"n": len(qs), "overall": overall.reset_index().to_dict("records"),
           "tuning": None if tuning is None else tuning.to_dict("records"),
           "per_question": {n: d.to_dict("records") for n, d in detail.items()}}
    (config.EVAL_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    main(use_dense=not a.no_dense, use_rerank=not a.no_rerank)
