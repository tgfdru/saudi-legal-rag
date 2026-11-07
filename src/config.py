"""Central configuration (every value can be overridden with an environment variable)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
CORPUS_CSV = ROOT / "data" / "corpus.csv"
EVAL_DIR = ROOT / "eval"
QUESTIONS_CSV = EVAL_DIR / "questions.csv"
FIG_DIR = ROOT / "figures"

# --- Models -------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")             # multilingual, strong on Arabic
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")           # any Anthropic model id
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")                # optional

# --- Retrieval ----------------------------------------------------------
TOP_K = 5                                               # articles passed to the generator
CANDIDATES = int(os.getenv("CANDIDATES", 50))           # per-retriever list size before fusion
RRF_K = 60                                              # Reciprocal Rank Fusion constant
W_DENSE = float(os.getenv("W_DENSE", 1.0))              # chosen by 5-fold CV in every fold (eval/evaluate.py)
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 20))

# --- Experiments ----------------------------------------------------------
SEED = 42
