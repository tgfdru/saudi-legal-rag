"""Scraper for the Bureau of Experts at the Council of Ministers (laws.boe.gov.sa).

Fetches each law page and writes it in the corpus format read by chunker.py.
Only the CURRENT (in-force) text of each article is kept; superseded versions
that the site shows in popups are skipped.

Ethics: one request at a time with a delay between laws, public official text,
no personal data involved.

Usage:
  python -m src.scraper --all                       # every law in LAWS below
  python -m src.scraper --query "نظام التنفيذ"       # search, print hits
  python -m src.scraper --id <uuid> --name <stem>   # one law
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://laws.boe.gov.sa"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic NLP course project; saudi-legal-rag)"}
DELAY_S = 2.0
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Laws related to administrative litigation before the Board of Grievances
# (the Board's own laws + the procedural / substantive laws it applies).
LAWS = {
    "diwan_al_mazalim": "5d3379bd-3547-494e-9fbd-a9a700f26e24",   # نظام ديوان المظالم
    "murafaat_diwan":   "f2f7b465-b576-4f47-a8e7-a9a700f27202",   # نظام المرافعات أمام ديوان المظالم
    "tanfeedh_diwan":   "ae1d79bf-3716-4bc5-a85e-ada100c8a870",   # نظام التنفيذ أمام ديوان المظالم
    "qada":             "ea1765a3-dec3-41a0-a32f-a9a700f26d58",   # نظام القضاء
    "murafaat_sharia":  "f0eaae46-9f84-40ee-815e-a9a700f268b3",   # نظام المرافعات الشرعية
    "ithbat":           "2716057c-c097-4bad-8e1e-ae1400c678d5",   # نظام الإثبات
    "tanfeedh":         "67dd54bc-c33a-48c8-8356-b44700a9ab55",   # نظام التنفيذ
    "indibat":          "896f4986-334b-4c76-b938-ada500eae2eb",   # نظام الانضباط الوظيفي
    "muhamah":          "f42655be-79b0-4fd4-bb90-a9a700f26a3e",   # نظام المحاماة
    "nazaa_reg":        "06272e6b-1fb9-4226-90b7-a9a700f21dc0",   # اللائحة التنفيذية لنظام نزع الملكية
    "muamalat":         "655fdb42-8c96-422b-b8c4-b04f0095c94c",   # نظام المعاملات المدنية
}


def _get(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def search(query: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(_get(f"{BASE}/BoeLaws/Laws/Search?query={urllib.parse.quote(query)}"), "html.parser")
    seen, out = set(), []
    for a in soup.select('a[href*="Id="], a[href*="LawDetails"]'):
        m = re.search(r"[0-9a-f-]{36}", a.get("href", ""))
        if m and m.group() not in seen:
            seen.add(m.group())
            out.append((m.group(), a.find_parent("div").get_text(" ", strip=True)[:80]))
    return out


def fetch_law(law_id: str) -> tuple[dict, list[tuple[str, str, str]]]:
    url = f"{BASE}/BoeLaws/Laws/LawDetails/{law_id}/1"
    soup = BeautifulSoup(_get(url), "lxml")   # lxml handles the site's malformed <p><div> nesting
    page = soup.get_text("\n")
    title = soup.title.get_text(strip=True) if soup.title else law_id
    date = re.search(r"تاريخ الإصدار\s*([\d/]+\s*هـ)", page)
    status = re.search(r"الحالة\s*\n?\s*(\S+)", page)
    meta = {"law": title, "law_id": law_id, "issued": date.group(1) if date else "",
            "status": status.group(1) if status else "", "source": url}

    chapter, articles = "", []
    for item in soup.select("div.article_item"):
        if item.find_parent(class_="article_item_popup"):
            continue                                            # skip historical versions
        heads = [h.get_text(" ", strip=True) for h in item.find_all("h3", recursive=False)]
        for t in heads:
            if "الباب" in t or "الفصل" in t:
                chapter = t.split("المادة")[0].strip()         # heading may embed the article label
        label = next((re.search(r"المادة[^:]*", h).group().strip() for h in heads if "المادة" in h), "")
        body = item.find("div", class_="HTMLContainer", recursive=False)
        if not (label and body):
            continue
        for br in body.find_all("br"):
            br.replace_with("\n")
        text = re.sub(r"[ \t ]+", " ", body.get_text()).strip()
        text = re.sub(r"\s*\n\s*", "\n", text)
        articles.append((chapter, label, text))
    return meta, articles


def write_law(meta: dict, articles, name: str, out_dir: Path = RAW_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    lines = [f"# {k}: {v}" for k, v in meta.items()]
    for chapter, label, text in articles:
        lines += [f"### {chapter}|{label}", text]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ {name:<18} {len(articles):>4} articles  [{meta['status']}]  {meta['law']}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("--id")
    ap.add_argument("--name")
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    a = ap.parse_args(argv)

    if a.query:
        for lid, label in search(a.query)[:10]:
            print(f"  {lid}  {label}")
        return
    jobs = LAWS.items() if a.all else [(a.name or a.id[:8], a.id)]
    total = 0
    for name, law_id in jobs:
        try:
            meta, arts = fetch_law(law_id)
            write_law(meta, arts, name, a.out)
            total += len(arts)
        except Exception as e:  # keep going; report failures
            print(f"✗ {name}: {e}", file=sys.stderr)
        time.sleep(DELAY_S)
    print(f"Total articles: {total}")


if __name__ == "__main__":
    main()
