"""Interactive demo (Gradio) — runs locally, in Colab (public share link) and on Hugging Face Spaces.

    python app/gradio_app.py            # local  → http://127.0.0.1:7860
    python app/gradio_app.py --share    # public link (e.g. from Colab, for the presentation)

Tabs
  1. اسأل            question → answer with [n] citations + source cards linking to the official text
  2. قارن النماذج     the same question through 3 retrievers side by side (lexical vs semantic vs hybrid)
  3. عن المشروع       corpus, evaluation results, links
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

if not torch.cuda.is_available():
    os.environ.setdefault("USE_RERANKER", "0")          # cross-encoder is too slow on CPU for a live demo

import gradio as gr  # noqa: E402

from src import config  # noqa: E402
from src.pipeline import get_rag  # noqa: E402

GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/tgfdru/saudi-legal-rag")
EXAMPLES = [
    "الجهة رفضت تظلمي، كم عندي وقت أرفع دعوى؟",
    "هل رفع الدعوى يوقف تنفيذ القرار الإداري؟",
    "كسبت حكم ضد وزارة وما نفذته، وش أسوي؟",
    "صدر حكم ضدي من المحكمة الإدارية، كم يوم عندي أعترض؟",
    "هل تقبل شهادة الابن لأبوه؟",
    "المقاول توفي قبل ما يكمل الشغل، وش يصير بالعقد؟",
]
NAMES = {"tfidf": "TF-IDF", "bm25": "BM25", "fasttext": "FastText", "dense": "bge-m3",
         "hybrid": "Hybrid (BM25 + bge-m3)", "hybrid+rerank": "Hybrid + Reranker"}

rag = get_rag()
LAWS = sorted({c.law for c in rag.chunks})

CSS = """
.rtl, .rtl * {direction: rtl; text-align: right;}
#hero {background: linear-gradient(135deg, #0f3d34 0%, #1f6f5c 60%, #3d8f78 100%); color: #fff;
       border-radius: 16px; padding: 22px 26px; margin-bottom: 8px;}
#hero h1 {color: #fff; margin: 0 0 4px 0; font-size: 1.8rem;}
#hero p {color: #d7ece5; margin: 0;}
.stat {display:inline-block; background: rgba(255,255,255,.14); border-radius: 999px; padding: 4px 12px;
       margin: 10px 0 0 6px; font-size: .9rem; color:#fff;}
.card {border: 1px solid #d5e5df; border-right: 5px solid #1f6f5c; border-radius: 10px; padding: 12px 14px;
       margin-bottom: 10px; background: #fbfdfc; direction: rtl; text-align: right;}
.card .cit {font-weight: 700; color: #0f3d34;}
.card .meta {color: #6b7f78; font-size: .82rem; margin: 2px 0 6px;}
.card .txt {line-height: 1.9; font-size: .95rem; white-space: pre-line; color:#1d2b27;}
.card a {color: #1f6f5c;}
.badge {display:inline-block; background:#e6f2ee; color:#0f3d34; border-radius:6px; padding:1px 8px; font-size:.78rem; margin-left:6px;}
.hit {background:#fff6d6 !important; border-right-color:#c99a06 !important;}
.col-title {font-weight:700; color:#0f3d34; margin-bottom:6px; direction: rtl; text-align:right;}
@media (prefers-color-scheme: dark) {
  .card {background:#14201d; border-color:#2b3d38;} .card .txt{color:#e3ece9;} .card .cit{color:#9fd8c5;}
  .badge{background:#21362f; color:#cfe9df;} .hit{background:#3a3214 !important;} .col-title{color:#9fd8c5;}
}
"""


def _card(rank, c, score, highlight=False, full=True):
    text = html.escape(c.text if full else c.text[:260] + ("…" if len(c.text) > 260 else ""))
    meta = " · ".join(x for x in [html.escape(c.chapter), f"تاريخ الإصدار {c.issued}" if c.issued else ""] if x)
    return (f"<div class='card{' hit' if highlight else ''}'><div class='cit'><span class='badge'>{rank}</span>"
            f"{html.escape(c.citation)}</div><div class='meta'>{meta} · score {score:.3f} · "
            f"<a href='{c.source}' target='_blank'>النص الرسمي ↗</a></div><div class='txt'>{text}</div></div>")


def ask(question, retriever, k):
    if not question or not question.strip():
        return "اكتب سؤالك أولاً.", ""
    res = rag.ask(question.strip(), k=int(k), mode=retriever)
    hits = rag.retrievers[retriever].search(question.strip(), int(k))
    cards = "".join(_card(i, rag.chunks[idx], s) for i, (idx, s) in enumerate(hits, 1))
    mode = "🤖 إجابة مولّدة مع الاستشهاد" if res["mode"] == "llm" else "📜 أقرب المواد النظامية (بدون نموذج لغوي)"
    answer = f"{res['answer']}\n\n---\n<small>{mode} · {NAMES.get(retriever, retriever)} · {res['latency_s']} ث · {res['disclaimer']}</small>"
    if res.get("llm_error"):  # key is set but the Claude call failed: show why (never the key itself)
        answer += f"\n\n<small>⚠️ تعذّر استخدام Claude: {str(res['llm_error'])[:200]}</small>"
    return answer, cards


def compare(question, a, b, c):
    if not question or not question.strip():
        return "", "", ""
    q = question.strip()
    runs = {name: rag.retrievers[name].search(q, 5) for name in (a, b, c)}
    ids = {name: [i for i, _ in r] for name, r in runs.items()}
    common = set(ids[a]) & set(ids[b]) & set(ids[c])
    outs = []
    for name in (a, b, c):
        body = "".join(_card(i, rag.chunks[idx], s, highlight=idx in common, full=False)
                       for i, (idx, s) in enumerate(runs[name], 1))
        outs.append(f"<div class='col-title'>{NAMES.get(name, name)}</div>{body}")
    return tuple(outs)


def about_md():
    lines = [f"**المدونة:** {len(rag.chunks):,} مادة من {len(LAWS)} نظاماً سعودياً، جُمعت من موقع هيئة الخبراء بمجلس الوزراء.\n"]
    lines += [f"- {law}" for law in LAWS]
    res = config.EVAL_DIR / "results.json"
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        lines.append(f"\n**تقييم الاسترجاع** (تحقق متقاطع 5-fold على {r['n']} سؤالاً):\n")
        lines.append("| النموذج | R@1 | R@5 | MRR@10 |\n|---|---|---|---|")
        for row in sorted(r["overall"], key=lambda x: -x["MRR@10"]):
            lines.append(f"| {NAMES.get(row['retriever'], row['retriever'])} | {row['R@1']:.2f} | {row['R@5']:.2f} | {row['MRR@10']:.2f} |")
    lines.append(f"\n🔗 **الكود والتقرير:** [{GITHUB_URL}]({GITHUB_URL})")
    lines.append("\n⚠️ أداة تعليمية استرشادية من نصوص الأنظمة، وليست استشارة قانونية.")
    return "\n".join(lines)


modes = list(rag.retrievers)
default = rag.mode
with gr.Blocks(title="المساعد النظامي — Saudi Legal RAG") as demo:
    gr.HTML(f"""<div id='hero' class='rtl'><h1>⚖️ المساعد النظامي</h1>
        <p>اسأل بلغتك عن التقاضي ضد جهة حكومية، ويجيبك النظام من نصوص الأنظمة السعودية مع ذكر النظام ورقم المادة.</p>
        <span class='stat'>{len(rag.chunks):,} مادة</span><span class='stat'>{len(LAWS)} نظاماً</span>
        <span class='stat'>{len(modes)} نماذج استرجاع</span></div>""")

    with gr.Tab("💬 اسأل"):
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="سؤالك", placeholder="مثال: الجهة رفضت تظلمي، كم عندي وقت أرفع دعوى؟",
                               lines=2, rtl=True)
                gr.Examples(EXAMPLES, inputs=q, label="جرّب")
            with gr.Column(scale=1):
                r_sel = gr.Dropdown(modes, value=default, label="نموذج الاسترجاع")
                k_sel = gr.Slider(1, 10, value=5, step=1, label="عدد المواد")
                btn = gr.Button("اسأل", variant="primary")
        ans = gr.Markdown(elem_classes="rtl")
        src = gr.HTML()
        btn.click(ask, [q, r_sel, k_sel], [ans, src])
        q.submit(ask, [q, r_sel, k_sel], [ans, src])

    with gr.Tab("⚔️ قارن النماذج"):
        gr.Markdown("نفس السؤال عبر ثلاثة نماذج. المواد **المظللة بالأصفر** اتفقت عليها النماذج الثلاثة.",
                    elem_classes="rtl")
        cq = gr.Textbox(label="السؤال", lines=2, rtl=True, value=EXAMPLES[0])
        with gr.Row():
            ca = gr.Dropdown(modes, value="bm25", label="النموذج ١")
            cb = gr.Dropdown(modes, value="dense" if "dense" in modes else modes[0], label="النموذج ٢")
            cc = gr.Dropdown(modes, value=default, label="النموذج ٣")
        cbtn = gr.Button("قارن", variant="primary")
        with gr.Row():
            o1, o2, o3 = gr.HTML(), gr.HTML(), gr.HTML()
        cbtn.click(compare, [cq, ca, cb, cc], [o1, o2, o3])
        gr.Examples(EXAMPLES, inputs=cq, label="جرّب")

    with gr.Tab("📊 عن المشروع"):
        gr.Markdown(about_md(), elem_classes="rtl")



def launch(share: bool = False):
    """Start the demo. share=True prints a public *.gradio.live link (works from Colab)."""
    return demo.queue().launch(share=share, css=CSS,
                               theme=gr.themes.Soft(primary_hue="emerald",
                                                    font=[gr.themes.GoogleFont("Tajawal"), "sans-serif"]))


if __name__ == "__main__":
    launch(share="--share" in sys.argv)
