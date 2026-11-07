"""Exploratory data analysis helpers (matplotlib + Arabic shaping).

Matplotlib does not shape/reorder Arabic by itself; every Arabic label goes
through `ar()` = arabic_reshaper (letter joining) + python-bidi (RTL order).
"""
from __future__ import annotations

import re
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .normalize import STOPWORDS, _DIACRITICS, normalize, tokenize

FONT = str(matplotlib.get_data_path()) + "/fonts/ttf/DejaVuSans.ttf"   # ships with matplotlib, has Arabic glyphs
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})
ACCENT, MUTED = "#1f6f5c", "#9bb5ad"


def ar(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


SHORT = {  # short display names for the 11 laws
    "diwan_al_mazalim": "ديوان المظالم", "murafaat_diwan": "المرافعات أمام الديوان",
    "tanfeedh_diwan": "التنفيذ أمام الديوان", "qada": "القضاء", "murafaat_sharia": "المرافعات الشرعية",
    "ithbat": "الإثبات", "tanfeedh": "التنفيذ", "indibat": "الانضباط الوظيفي", "muhamah": "المحاماة",
    "nazaa_reg": "لائحة نزع الملكية", "muamalat": "المعاملات المدنية",
}


# ------------------------------------------------------------------ charts
def articles_per_law(df: pd.DataFrame):
    s = df["law_key"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([ar(SHORT.get(k, k)) for k in s.index], s.values, color=ACCENT)
    for y, v in enumerate(s.values):
        ax.text(v + 5, y, str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of articles")
    ax.set_title(f"Articles per law (n = {len(df):,})")
    fig.tight_layout()
    return fig


def length_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["n_words"], bins=np.logspace(np.log10(4), np.log10(df.n_words.max() + 1), 40), color=ACCENT)
    axes[0].set_xscale("log")
    axes[0].axvline(df.n_words.median(), color="black", ls="--", lw=1)
    axes[0].text(df.n_words.median() * 1.1, axes[0].get_ylim()[1] * .9, f"median = {df.n_words.median():.0f}")
    axes[0].set(xlabel="Words per article (log scale)", ylabel="Articles", title="Article length")
    order = df.groupby("law_key")["n_words"].median().sort_values().index
    data = [df.loc[df.law_key == k, "n_words"] for k in order]
    axes[1].boxplot(data, vert=False, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=MUTED, color=ACCENT), medianprops=dict(color="black"))
    axes[1].set_yticks(range(1, len(order) + 1), [ar(SHORT.get(k, k)) for k in order])
    axes[1].set(xlabel="Words per article", title="Length by law (outliers hidden)")
    fig.tight_layout()
    return fig


def top_ngrams(df: pd.DataFrame, n=1, top=20, stemmed=False):
    cnt = Counter()
    for t in df["text"]:
        toks = tokenize(t) if stemmed else [w for w in re.sub(r"[^\w\s]", " ", normalize(t)).split()
                                            if w not in STOPWORDS and len(w) > 1]
        cnt.update(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return cnt.most_common(top)


def ngram_bar(pairs, title):
    words, counts = zip(*pairs[::-1])
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(words) + 1))
    ax.barh([ar(w) for w in words], counts, color=ACCENT)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    fig.tight_layout()
    return fig


def wordcloud(df: pd.DataFrame, max_words=120):
    from wordcloud import WordCloud
    from PIL import features
    # Pillow built with libraqm shapes Arabic itself; otherwise shape + reorder manually
    fix = (lambda w: w) if features.check("raqm") else ar
    freqs = dict(top_ngrams(df, 1, 400))
    wc = WordCloud(font_path=FONT, width=1100, height=550, background_color="white", max_words=max_words,
                   colormap="viridis", random_state=42).generate_from_frequencies({fix(k): v for k, v in freqs.items()})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most frequent words (normalized, stopwords removed)")
    fig.tight_layout()
    return fig


def zipf(df: pd.DataFrame):
    cnt = Counter(w for t in df["text"] for w in normalize(t).split())
    f = np.array(sorted(cnt.values(), reverse=True))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(np.arange(1, len(f) + 1), f, color=ACCENT)
    ax.loglog(np.arange(1, len(f) + 1), f[0] / np.arange(1, len(f) + 1), "--", color="gray", label="ideal Zipf (1/r)")
    ax.set(xlabel="Rank", ylabel="Frequency", title="Zipf's law check")
    ax.legend()
    fig.tight_layout()
    return fig


def preprocessing_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Vocabulary size after each preprocessing step — justifies each step with a number."""
    steps = {
        "1. raw whitespace tokens": lambda t: t.split(),
        "2. + remove punctuation": lambda t: re.sub(r"[^\w\s]", " ", t).split(),
        "3. + remove diacritics/tatweel": lambda t: re.sub(r"[^\w\s]", " ", _DIACRITICS.sub("", t).replace("\u0640", "")).split(),
        "4. + letter normalization (أإآ→ا, ى→ي, ة→ه)": lambda t: re.sub(r"[^\w\s]", " ", normalize(t)).split(),
        "5. + stopword removal": lambda t: [w for w in re.sub(r"[^\w\s]", " ", normalize(t)).split()
                                            if w not in STOPWORDS and len(w) > 1],
        "6. + light stemming (strip ال/وال/بال…)": tokenize,
    }
    rows = []
    for name, f in steps.items():
        toks = [w for t in df["text"] for w in f(t)]
        rows.append({"step": name, "tokens": len(toks), "vocabulary": len(set(toks))})
    out = pd.DataFrame(rows)
    out["vocab Δ vs raw"] = (out["vocabulary"] / out["vocabulary"].iloc[0] - 1).map("{:+.0%}".format)
    return out
