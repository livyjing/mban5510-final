"""
MedlinePlus Live Retrieval Tool
================================
Adapted from the health_rag_hw2 project (medlineplus_live.py + rag_lib.py).

Provides ``get_health_context(query)`` which runs a four-step pipeline:

  1. Search  – query the official MedlinePlus Web Service API for topic URLs
  2. Fetch   – download each page and extract main content text (BeautifulSoup)
  3. Chunk   – split text into overlapping character windows
  4. Compress– keep only keyword-relevant lines; cap total to 2 500 chars
  5. Format  – return numbered [1] [2] … context blocks ready for the LLM

Returns
-------
(context_str, source_urls)
  context_str   – formatted passages to inject into a system prompt; empty if
                  MedlinePlus returned no relevant content for the query.
  source_urls   – list of MedlinePlus page URLs that contributed to the context.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_WS = "https://wsearch.nlm.nih.gov/ws/query"
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    )
}
_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Step 1 – Search MedlinePlus Web Service → page URLs
# ---------------------------------------------------------------------------

def _medlineplus_search(query: str, max_results: int = 3) -> List[str]:
    """Query the MedlinePlus Web Service and return stable health-topic page URLs."""
    q = (query or "").strip()
    if not q:
        return []

    url = (
        f"{_BASE_WS}?db=healthTopics&rettype=brief"
        f"&retmax={max_results}&term={quote_plus(q)}"
    )
    try:
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception:
        return []

    urls: List[str] = []
    for doc in root.findall(".//list/document"):
        u = doc.attrib.get("url", "")
        if u and u.startswith("https://medlineplus.gov/"):
            urls.append(u)

    # De-duplicate, preserve order
    seen: set = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:max_results]


# ---------------------------------------------------------------------------
# Step 2 – Fetch page text
# ---------------------------------------------------------------------------

def _fetch_page_text(url: str) -> str:
    """Download a MedlinePlus page and return clean plain text from main content."""
    try:
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        r.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    main = (
        soup.find("main")
        or soup.find("div", {"id": "mplus-content"})
        or soup.body
    )
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3 – Chunk text
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    """Split text into overlapping character windows."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


# ---------------------------------------------------------------------------
# Step 4 – Keyword-compress passages
# ---------------------------------------------------------------------------

def _keyword_compress(
    query: str,
    passages: List[Dict],
    max_total_chars: int = 2500,
) -> List[Dict]:
    """Keep only keyword-relevant lines; stop when total length exceeds limit."""
    q_words = [w for w in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(w) >= 4]
    compressed: List[Dict] = []
    total = 0

    for p in passages:
        lines = p["text"].split("\n")
        keep = [ln for ln in lines if any(w in ln.lower() for w in q_words) or not q_words]
        if not keep:
            keep = lines[:2]   # always include at least a snippet

        new_text = "\n".join(keep)[:1200]
        if total + len(new_text) > max_total_chars:
            break

        compressed.append({"text": new_text, "source": p["source"]})
        total += len(new_text)

    return compressed


# ---------------------------------------------------------------------------
# Step 5 – Format context
# ---------------------------------------------------------------------------

def _format_context(passages: List[Dict]) -> str:
    """Return numbered [1] [2] … context blocks for LLM injection."""
    blocks = []
    for idx, p in enumerate(passages, start=1):
        blocks.append(f"[{idx}] {p['source']}\n{p['text']}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_health_context(query: str, max_pages: int = 3) -> Tuple[str, List[str]]:
    """
    Search MedlinePlus live for *query* and return evidence-based context.

    Parameters
    ----------
    query     : natural-language patient question
    max_pages : maximum number of MedlinePlus topic pages to fetch

    Returns
    -------
    (context_str, source_urls)
      context_str  – numbered [1][2]… context blocks ready to inject into an LLM
                     system prompt.  Empty string when nothing was found.
      source_urls  – deduplicated list of MedlinePlus page URLs retrieved.
    """
    # Primary search
    urls = _medlineplus_search(query, max_results=max_pages)

    # Fallback: split multi-concept queries ("bipolar disorder and insomnia")
    if not urls:
        parts = re.split(r"\b(and|with)\b|,|;|\+|/|\|", query.lower())
        parts = [
            p.strip() for p in parts
            if p.strip() and p.strip() not in {"and", "with"}
        ]
        for part in parts[:3]:
            urls.extend(_medlineplus_search(part, max_results=max_pages))

        # De-dup
        seen: set = set()
        uniq: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        urls = uniq[:max_pages]

    if not urls:
        return "", []

    # Fetch → chunk → accumulate passages
    passages: List[Dict] = []
    for u in urls:
        text = _fetch_page_text(u)
        if text:
            for chunk in _chunk_text(text):
                passages.append({"source": u, "text": chunk})

    if not passages:
        return "", []

    compressed = _keyword_compress(query, passages)
    if not compressed:
        return "", []

    context = _format_context(compressed)
    source_urls = list(dict.fromkeys(p["source"] for p in compressed))  # ordered dedup
    return context, source_urls
