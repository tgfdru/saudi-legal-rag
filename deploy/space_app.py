"""Hugging Face Space entry point.

Pulls the project from GitHub (single source of truth) and starts its Gradio demo.
Code, data and the evaluation: https://github.com/tgfdru/saudi-legal-rag

The Space runs on ZeroGPU hardware, which requires one @spaces.GPU function. The
cross-encoder reranker can use it when ENABLE_GPU_RERANK=1 is set in the Space
settings; by default the demo serves dense retrieval (bge-m3), which needs no GPU
(R@5 0.91 in our 5-fold evaluation).
"""
import os
import runpy
import subprocess
import sys

import spaces  # must be imported before torch on ZeroGPU

REPO = "https://github.com/tgfdru/saudi-legal-rag"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project")
GPU_RERANK = os.getenv("ENABLE_GPU_RERANK") == "1"

if not os.path.isdir(os.path.join(DST, "src")):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DST], check=True)

os.environ["USE_RERANKER"] = "1" if GPU_RERANK else "0"
os.environ.setdefault("GITHUB_URL", REPO)
os.chdir(DST)
sys.path.insert(0, DST)

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402
from src import config, retriever  # noqa: E402

# ZeroGPU reports CUDA as available in the main process: pin the bi-encoder to CPU.
_dense = SentenceTransformer(config.EMBED_MODEL, device="cpu")
_dense.max_seq_length = 512
retriever.DenseRetriever._model = _dense

_CE = CrossEncoder(config.RERANK_MODEL, max_length=512, device="cuda") if GPU_RERANK else None


@spaces.GPU(duration=30)
def _gpu_predict(pairs):
    return _CE.predict(pairs, batch_size=32)


class _GPUCrossEncoder:
    def predict(self, pairs, **_):
        return _gpu_predict(pairs)


if GPU_RERANK:
    retriever.RerankedRetriever._model = _GPUCrossEncoder()

runpy.run_path(os.path.join(DST, "app", "gradio_app.py"), run_name="__main__")
