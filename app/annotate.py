"""Annotation tool for the evaluation question set.

Each team member writes questions IN THEIR OWN WORDS (or pastes a real question
collected from a real person), then marks which article(s) answer it.
Saved to eval/questions.csv — the only evaluation data in the project.

Run:  streamlit run app/annotate.py

Two ways to find the gold article:
  1. Candidate list from BM25 + TF-IDF (lexical only — dense models are NOT
     shown, so labeling does not favour the model we want to evaluate).
  2. Manual lookup by law + article number (for questions whose answer the
     lexical list misses — these are exactly the hard cases we need).
"""
import csv
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config                                  # noqa: E402
from src.chunker import load_corpus                     # noqa: E402
from src.retriever import BM25Retriever, TfidfRetriever  # noqa: E402

st.set_page_config(page_title="توسيم أسئلة التقييم", page_icon="📝", layout="wide")
st.markdown("<style>.stMarkdown,.stTextInput,.stTextArea{direction:rtl;text-align:right}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def load():
    chunks = load_corpus(config.RAW_DIR)
    return chunks, BM25Retriever(chunks), TfidfRetriever(chunks)


chunks, bm25, tfidf = load()
by_law = {}
for i, c in enumerate(chunks):
    by_law.setdefault(c.law, []).append(i)

path = config.QUESTIONS_CSV
if not path.exists() or path.stat().st_size == 0:
    path.write_text("qid,question,gold_ids,author,source\n", encoding="utf-8")
with path.open(encoding="utf-8") as f:
    existing = list(csv.DictReader(f))

st.title("📝 توسيم أسئلة التقييم")
st.caption(f"عدد الأسئلة المحفوظة: {len(existing)}  —  الهدف: ١٥٠ سؤالاً على الأقل للفريق")

with st.sidebar:
    author = st.text_input("اسمك (بالإنجليزية، مثل nader)", key="author").strip().lower()
    source = st.radio("مصدر السؤال", ["team", "real"],
                      format_func=lambda s: "كتبته بنفسي" if s == "team" else "سؤال حقيقي جمعته من شخص/منتدى")
    st.markdown("**قواعد:**\n- اكتب السؤال بأسلوبك، عامي أو فصيح\n- لا تنسخ نص المادة\n"
                "- لا تستخدم أي أداة ذكاء اصطناعي لكتابة الأسئلة\n- نوّع بين الأنظمة")
    mine = [r for r in existing if r["author"] == author]
    st.metric("أسئلتك", len(mine))

q = st.text_area("السؤال", key="q", height=80)
selected: set[int] = set(st.session_state.get("sel", set()))

tab1, tab2 = st.tabs(["مرشحات (بحث نصي)", "بحث يدوي بالنظام والمادة"])
with tab1:
    if q.strip():
        cand = {i for i, _ in bm25.search(q, 10)} | {i for i, _ in tfidf.search(q, 10)}
        for i in sorted(cand):
            c = chunks[i]
            if st.checkbox(f"{c.citation}", key=f"c{i}", value=i in selected):
                selected.add(i)
            else:
                selected.discard(i)
            st.caption(c.text[:400] + ("…" if len(c.text) > 400 else ""))
with tab2:
    law = st.selectbox("النظام", list(by_law))
    num = st.text_input("رقم/اسم المادة (مثال: الثامنة)")
    if num.strip():
        hits = [i for i in by_law[law] if num.strip() in chunks[i].article]
        for i in hits[:10]:
            c = chunks[i]
            if st.checkbox(c.citation, key=f"m{i}", value=i in selected):
                selected.add(i)
            st.caption(c.text[:600])
st.session_state.sel = selected

if selected:
    st.success("المواد المختارة: " + " ، ".join(chunks[i].citation for i in sorted(selected)))

if st.button("💾 حفظ السؤال", type="primary", disabled=not (q.strip() and selected and author)):
    if any(r["question"].strip() == q.strip() for r in existing):
        st.error("هذا السؤال محفوظ من قبل.")
    else:
        n = 1 + max([int(m.group(1)) for r in existing if r["author"] == author
                     for m in [re.search(r"-(\d+)$", r["qid"])] if m] or [0])
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([f"{author}-{n:03d}", q.strip(),
                                    ";".join(chunks[i].id for i in sorted(selected)), author, source])
        st.session_state.sel = set()
        st.toast("تم الحفظ ✅")
        st.rerun()
