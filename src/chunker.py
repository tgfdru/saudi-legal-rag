"""Article-level chunking + corpus export.

Legal text is chunked on its natural unit — the article (المادة) — instead of
fixed token windows. Every chunk keeps the metadata needed for citation:
law name, article label, chapter, issue date, and source URL.

Input format (data/raw/*.txt, produced by scraper.py):
    # law: <name>
    # law_id: <boe id>
    # issued: <date>
    # status: <ساري / لاغي>
    # source: <url>
    ### <chapter>|<article label>
    <article text ...>
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Chunk:
    id: str          # "<law_key>#<nnn>"
    law_key: str     # short ASCII key = raw file stem (used as class label)
    law: str         # full Arabic law name
    article: str     # e.g. "المادة الثامنة"
    chapter: str     # الباب / الفصل heading ("" if the law has none)
    text: str
    issued: str
    source: str

    @property
    def citation(self) -> str:
        return f"{self.law} — {self.article}"

    def embed_text(self) -> str:
        """Text given to retrievers: law + chapter + article header adds context."""
        return f"{self.law} | {self.chapter} | {self.article}\n{self.text}"

    def to_dict(self) -> dict:
        return {**asdict(self), "citation": self.citation}


def parse_law_file(path: Path) -> list[Chunk]:
    meta, chunks = {}, []
    chapter = article = None
    buf: list[str] = []

    def flush():
        if article and buf:
            chunks.append(Chunk(
                id=f"{path.stem}#{len(chunks) + 1:03d}",
                law_key=path.stem,
                law=meta.get("law", path.stem),
                article=article,
                chapter=chapter or "",
                text="\n".join(buf).strip(),
                issued=meta.get("issued", ""),
                source=meta.get("source", ""),
            ))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            flush()
            chapter, _, article = line[4:].partition("|")
            chapter, article = chapter.strip(), article.strip()
            buf = []
        elif line.startswith("# ") and ":" in line and article is None:
            k, _, v = line[2:].partition(":")
            meta[k.strip()] = v.strip()
        elif line.strip():
            buf.append(line.strip())
    flush()
    return chunks


def load_corpus(raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in sorted(raw_dir.glob("*.txt")):
        chunks.extend(parse_law_file(f))
    return chunks


def corpus_dataframe(chunks: list[Chunk]):
    """One row per article, raw + cleaned text — this is the submitted dataset (.csv)."""
    import pandas as pd
    from .normalize import normalize, tokenize

    df = pd.DataFrame([c.to_dict() for c in chunks])
    df["text_clean"] = df["text"].map(normalize)
    df["tokens"] = df["text"].map(lambda t: " ".join(tokenize(t)))
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()
    # exact duplicate texts (e.g. boilerplate "يعمل بهذا النظام…") — flagged, not dropped:
    # each is a distinct article that must stay citable.
    df["is_dup_text"] = df.duplicated("text_clean", keep=False)
    return df
