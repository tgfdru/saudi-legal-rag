<div align="center">

# ⚖️ Saudi Legal RAG

**Arabic legal question → the Saudi-law article that answers it, with citations**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgfdru/saudi-legal-rag/blob/main/Saudi_Legal_RAG.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Articles](https://img.shields.io/badge/corpus-1%2C474%20articles-1f6f5c)
![Laws](https://img.shields.io/badge/laws-11-1f6f5c)

NLP Course Project · University of Bisha · Nader Al-Shehri

</div>

---

## What it does

A person facing a dispute with a government body asks a question in everyday Arabic, such as
*«الجهة رفضت تظلمي، كم عندي وقت أرفع دعوى؟»*. The system:

1. **retrieves** the governing article(s) from 1,474 articles of 11 Saudi laws,
2. **answers** only from those articles, citing each one (`نظام … — المادة …`) with a link to the official text,
3. **refuses** when the articles don't cover the question. It never invents law.

## Results

**Retrieval** (main task) uses 5-fold cross-validation over 160 questions. The hybrid weight is tuned on the training folds.

| Retriever | R@1 | R@5 | MRR@10 |
|---|---:|---:|---:|
| **bge-m3 + cross-encoder reranker** | **0.80** | **0.96** | **0.87** |
| bge-m3 (dense) | 0.65 | 0.91 | 0.77 |
| Hybrid BM25 + bge-m3 (CV chose w = 1.0, i.e. dense only) | 0.65 | 0.91 | 0.77 |
| BM25 | 0.37 | 0.61 | 0.48 |
| TF-IDF | 0.32 | 0.59 | 0.43 |
| FastText (trained on the corpus) | 0.21 | 0.48 | 0.33 |

**Law classification** (supporting task, 11 classes) uses nested stratified 5-fold CV and reports mean ± std.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| **TF-IDF + Logistic Regression** | **0.88 ± 0.01** | **0.81 ± 0.03** |
| TF-IDF + Complement Naive Bayes | 0.87 ± 0.02 | 0.76 ± 0.04 |
| bge-m3 (frozen) + Logistic Regression | 0.84 ± 0.02 | 0.76 ± 0.06 |
| FastText + Logistic Regression | 0.75 ± 0.03 | 0.63 ± 0.04 |

**Takeaway.** The task decides the representation. Sparse n-grams best capture *which law* an article belongs to
(style and vocabulary). Dense semantics best capture *which article answers* an everyday question (meaning across
registers). Full analysis is in [`docs/REPORT.md`](docs/REPORT.md).

## Live demo

| Where | How |
|---|---|
| 🤗 Hugging Face Space | [huggingface.co/spaces/tgfdru/saudi-legal-rag](https://huggingface.co/spaces/tgfdru/saudi-legal-rag) |
| Colab (public link) | open the notebook → set `LAUNCH_DEMO = True` in the last section |
| Local | `pip install -r requirements.txt && python app/gradio_app.py` |

The interface has three tabs. **Ask** gives an answer plus source cards. **Compare models** runs the same question
through BM25, bge-m3 and the hybrid side by side and highlights the articles they agree on. **About** shows the
corpus and the evaluation results.

## Repository

```
Saudi_Legal_RAG.ipynb   ← the full pipeline, runs top to bottom (Colab / local)
docs/REPORT.md          ← methodology, insights and findings for every section
data/corpus.csv         ← the dataset: 1,474 articles, raw + cleaned text, metadata, source URL
data/raw/               ← the 11 laws as scraped
eval/questions.csv      ← 160 labeled evaluation questions (+ team questions)
src/                    scraper · Arabic normalization · chunker · EDA · retrievers · classifiers · generator
eval/evaluate.py        5-fold CV retrieval evaluation
app/gradio_app.py       interactive demo        app/annotate.py   question-labeling tool
figures/                every chart used in the report and slides
deploy/                 GitHub + Hugging Face Spaces guide
```

## Data

The data is the official consolidated text from the **Bureau of Experts at the Council of Ministers**
([laws.boe.gov.sa](https://laws.boe.gov.sa)), collected with `src/scraper.py`. It keeps only in-force article
text, uses a 2 s delay between requests, and contains no personal data.

| Law | Articles |
|---|---:|
| نظام المعاملات المدنية | 721 |
| نظام المرافعات الشرعية | 242 |
| نظام الإثبات | 129 |
| نظام القضاء | 85 |
| نظام التنفيذ | 65 |
| نظام المرافعات أمام ديوان المظالم | 63 |
| نظام المحاماة | 55 |
| نظام التنفيذ أمام ديوان المظالم | 37 |
| نظام ديوان المظالم | 26 |
| اللائحة التنفيذية لنظام نزع ملكية العقارات | 26 |
| نظام الانضباط الوظيفي | 25 |

## Run

```bash
git clone https://github.com/tgfdru/saudi-legal-rag && cd saudi-legal-rag
pip install -r requirements.txt
jupyter notebook Saudi_Legal_RAG.ipynb    # or: python -m eval.evaluate
```
The first run downloads bge-m3 (~2.3 GB) and the reranker (~2.2 GB). Use a GPU (Colab T4) for reranking.

## Acknowledgements

Libraries: scikit-learn, gensim, rank_bm25, sentence-transformers (BAAI/bge-m3, BAAI/bge-reranker-v2-m3), Gradio,
matplotlib. An AI assistant (Claude) helped write code and drafted the evaluation questions marked `source = ai`.
Team-written questions are marked `source = team`. All corpus text is real legislation.

> ⚠️ Educational tool. Not legal advice.
