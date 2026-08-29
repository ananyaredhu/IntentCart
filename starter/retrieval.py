"""Extra retrieval routes layered on top of the starter's BM25 index.

Route B (category + structured attributes) and Route C (dense vector
similarity) are implemented here, kept separate from agent.py so the
Agent class stays a thin orchestrator over these pieces.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

# Data-mined, not hand-guessed: each term below was checked against real
# frequency across all 50,000 catalog products (title+features+description+
# details) before being kept. The original hand-picked lists were missing
# common catalog terms entirely -- e.g. "steel"/"stainless steel" (2,326 /
# 1,778 products) were absent, which is exactly what caused the
# "Stainless Steel Band" extraction gap found earlier. Compound phrases are
# listed alongside their single-word component (e.g. "faux leather" and
# "leather") since a word-boundary match on the shorter word already covers
# the compound case; both are kept for readability of what's covered.
MATERIAL_VOCAB = (
    "polyester", "cotton", "leather", "rubber", "spandex", "silver", "gold",
    "mesh", "lace", "nylon", "metal", "steel", "stainless steel",
    "sterling silver", "fleece", "suede", "crystal", "rayon", "denim",
    "elastane", "diamond", "jersey", "plastic", "alloy", "canvas", "acrylic",
    "wool", "velvet", "satin", "faux leather", "chiffon", "genuine leather",
    "microfiber", "viscose", "cubic zirconia", "silk", "pearl", "pu leather",
    "vinyl", "linen", "brass", "copper", "resin", "terry", "titanium",
    "cashmere", "modal", "zinc", "platinum", "bamboo", "ceramic", "chenille",
)
COLOR_VOCAB = (
    "black", "white", "silver", "blue", "gold", "red", "grey", "gray",
    "green", "pink", "brown", "yellow", "navy", "purple", "orange",
    "rose gold", "rust", "beige", "khaki", "wine", "tan", "turquoise",
    "burgundy", "charcoal", "multicolor", "coral", "cream", "olive",
    "royal blue", "ivory", "emerald", "teal", "mint", "maroon", "fuchsia",
    "lavender", "mustard", "magenta",
)

SIZE_VOCAB = ("small", "medium", "large", "x-large", "xl", "xs", "xxl", "s", "m", "l")
STYLE_VOCAB = (
    "long sleeve", "short sleeve", "sleeveless", "pullover", "slip on",
    "loose fit", "v-neck", "slim fit", "mini", "fitted", "tank top",
    "button down", "lace up", "crew neck", "maxi", "relaxed fit", "oversized",
    "high waisted", "polo", "midi", "regular fit", "a-line", "bodycon",
    "open toe", "scoop neck", "off shoulder", "cropped", "round toe",
    "zip up", "wide leg", "henley", "pointed toe", "straight leg",
    "turtleneck", "strapless", "high neck", "high top", "low top",
    "athletic fit", "low rise", "skinny fit", "boot cut",
)
USE_CASE_VOCAB = (
    "casual", "party", "summer", "work", "outdoor", "beach", "everyday",
    "winter", "running", "sports", "wedding", "spring", "travel", "athletic",
    "fall", "business", "office", "workout", "hiking", "school", "yoga",
    "gym", "autumn", "formal", "swimming", "training", "cycling", "lounge",
    "tennis", "camping", "fishing", "golf", "sleep", "basketball",
    "maternity", "hunting", "soccer",
)

# "feature" previously had no vocab extractor at all -- it only ever got
# filled via raw fallback-capture of the whole reply sentence. Tracing real
# sessions (public_0095, public_0126) showed the evaluator's own "feature"
# disclosures overwhelmingly fall into these recurring sub-patterns
# (closure type, care instructions, sourcing, sole material), each checked
# against real catalog frequency before being kept -- e.g. "imported" alone
# appears on 15,300/50,000 products, which is exactly why Part 1's IDF
# weighting matters: without it, a boost on "imported" would be almost
# meaningless noise. "lace up" is intentionally left out here since it's
# already covered by STYLE_VOCAB.
CLOSURE_VOCAB = (
    "pull on", "zipper", "elastic", "button", "tie", "drawstring", "buckle",
    "snap", "velcro", "hook and eye",
)
CARE_VOCAB = ("machine wash", "hand wash only", "hand wash", "dry clean")
SOURCING_VOCAB = ("imported", "made in the usa", "made in usa")
SOLE_VOCAB = ("rubber sole", "synthetic sole", "rubber outsole", "leather sole", "synthetic outsole")
FEATURE_SIGNAL_VOCAB = CLOSURE_VOCAB + CARE_VOCAB + SOURCING_VOCAB + SOLE_VOCAB

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIAL_VOCAB) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_VOCAB) + r")\b", re.I)
# "under $80", "below 50", "budget of 100", "max 60" -> treated as an upper bound
BUDGET_BOUND_RE = re.compile(
    r"(?:under|below|less than|up to|max(?:imum)?|budget(?: of)?)\D{0,10}\$?\s*(\d+(?:\.\d+)?)",
    re.I,
)
# a bare "$80" anywhere, with no directional word nearby, is also treated as an upper bound
BUDGET_BARE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
# "size 9", "size medium", "a size large" -- requires the word "size" nearby so a bare
# number (e.g. from a price) is never mistaken for a size
SIZE_RE = re.compile(r"\bsize\s*(\d{1,2}(?:\.\d)?|" + "|".join(SIZE_VOCAB) + r")\b", re.I)
STYLE_RE = re.compile(r"\b(" + "|".join(STYLE_VOCAB) + r")\b", re.I)
USE_CASE_RE = re.compile(r"\b(" + "|".join(USE_CASE_VOCAB) + r")\b", re.I)
FEATURE_SIGNAL_RE = re.compile(r"\b(" + "|".join(FEATURE_SIGNAL_VOCAB) + r")\b", re.I)


def extract_budget_max(text: str) -> float | None:
    match = BUDGET_BOUND_RE.search(text)
    if match:
        return float(match.group(1))
    match = BUDGET_BARE_RE.search(text)
    if match:
        return float(match.group(1))
    return None


def extract_material(text: str) -> str | None:
    match = MATERIAL_RE.search(text)
    return match.group(1).lower() if match else None


def extract_color(text: str) -> str | None:
    match = COLOR_RE.search(text)
    return match.group(1).lower() if match else None


def extract_size(text: str) -> str | None:
    match = SIZE_RE.search(text)
    return match.group(1).lower() if match else None


def extract_style(text: str) -> str | None:
    match = STYLE_RE.search(text)
    return match.group(1).lower() if match else None


def extract_use_case(text: str) -> str | None:
    match = USE_CASE_RE.search(text)
    return match.group(1).lower() if match else None


def extract_feature_signal(text: str) -> str | None:
    """Recognizes closure/care/sourcing/sole sub-patterns within a "feature"
    disclosure. Returning None (no match) is expected and fine -- update_slots'
    existing fallback-capture path still stores the raw text in that case,
    exactly as before this was added; this only gives a chance at a cleaner,
    more specific value when one of these common sub-patterns is present."""
    match = FEATURE_SIGNAL_RE.search(text)
    return match.group(1).lower() if match else None


def build_document_frequencies(corpus: list[str], tokenizer: Callable[[str], list[str]]) -> dict[str, int]:
    """Global term -> document-frequency table across the whole catalog,
    used to scale structured-attribute boosts by rarity (IDF-style) so a
    common word (e.g. "imported," present on 15,300/50,000 products) barely
    moves the ranking while a rare, genuinely distinguishing word moves it
    much more. Takes the caller's own tokenizer (agent.py's `_terms`) rather
    than defining a second one here, to avoid a circular import and to stay
    consistent with how the rest of the pipeline tokenizes text."""
    doc_freq: dict[str, int] = {}
    for text in corpus:
        for term in set(tokenizer(text)):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return doc_freq


def idf_weight(term: str, doc_freq: dict[str, int], total_docs: int, min_weight: float = 0.15) -> float:
    """Normalized IDF in [min_weight, 1.0]: a term on every product (idf~0)
    still contributes a small floor rather than literally nothing (it may
    still carry some signal), while a term on a handful of products gets
    close to the full boost. A term never seen in the catalog at all (e.g.
    an unrecognized fallback-captured word) gets the max weight, treated as
    maximally specific/rare rather than penalized for being unknown."""
    if total_docs <= 0:
        return 1.0
    df = doc_freq.get(term, 0)
    idf = math.log((total_docs + 1) / (df + 1))
    normalized = idf / math.log(total_docs + 1)
    return max(min_weight, min(1.0, normalized))


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], weights: list[float] | None = None, k: int = 60
) -> dict[str, float]:
    """Weights let the caller continuously lean the fusion toward one route
    over another (e.g. Track D's Buying<->Browsing spectrum) without
    changing the underlying rank-fusion mechanics -- default weight is 1.0
    per list, identical to unweighted RRF."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, parent_asin in enumerate(ranked, start=1):
            scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (k + rank)
    return scores


class CategoryIndex:
    """Route B: ranks products by overlap with category-taxonomy tokens
    the customer has mentioned, using an inverted index for speed."""

    def __init__(self, parent_asins: list[str], category_tokens: list[set[str]]) -> None:
        self.postings: dict[str, list[str]] = {}
        for parent_asin, tokens in zip(parent_asins, category_tokens):
            for token in tokens:
                self.postings.setdefault(token, []).append(parent_asin)

    def query(self, terms: list[str], top_n: int) -> list[str]:
        counts: dict[str, int] = {}
        for term in terms:
            for parent_asin in self.postings.get(term, ()):
                counts[parent_asin] = counts.get(parent_asin, 0) + 1
        ranked = sorted(counts, key=lambda a: -counts[a])[:top_n]
        return ranked

    def filter_set(self, terms: list[str], min_overlap: int) -> set[str]:
        """All candidates matching at least `min_overlap` of the given terms
        -- used as a hard pre-filter, unlike `query`, which just ranks by
        count for RRF fusion. A hard filter needs a precision threshold: a
        single shared token (e.g. "women," shared by thousands of unrelated
        products) isn't enough to trust on its own."""
        counts: dict[str, int] = {}
        for term in terms:
            for parent_asin in self.postings.get(term, ()):
                counts[parent_asin] = counts.get(parent_asin, 0) + 1
        return {asin for asin, count in counts.items() if count >= min_overlap}


class EmbeddingIndex:
    """Route C: dense vector similarity over a small local sentence-transformer.

    Embeddings for the whole catalog are computed once and cached to disk.
    If the model can't be loaded (no cache and no network), `available`
    stays False and callers should simply skip this route rather than fail.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, catalog_path: Path, parent_asins: list[str], corpus: list[str]) -> None:
        self.parent_asins = parent_asins
        self.asin_to_index = {asin: i for i, asin in enumerate(parent_asins)}
        self.available = False
        self.model = None
        self.vectors: np.ndarray | None = None
        cache_path = catalog_path.with_suffix(".embeddings.npy")
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.MODEL_NAME)
            vectors = None
            if cache_path.exists():
                cached = np.load(cache_path)
                if cached.shape[0] == len(parent_asins):
                    vectors = cached
            if vectors is None:
                vectors = self.model.encode(
                    corpus,
                    batch_size=128,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                vectors = np.asarray(vectors, dtype=np.float32)
                np.save(cache_path, vectors)
            self.vectors = vectors
            self.available = True
        except Exception:
            self.available = False

    def query(self, text: str, top_n: int) -> list[str]:
        if not self.available or not text.strip():
            return []
        query_vector = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_n]
        return [self.parent_asins[i] for i in top_indices]

    def similarities(self, text: str, candidates: list[str]) -> dict[str, float]:
        """Cosine similarity between `text` and a specific set of candidates
        (rather than a full-catalog top-n query) -- used by the Track C
        reranker to score the top fused pool against a distilled memory
        summary, separately from Route C's own raw-text retrieval query."""
        if not self.available or not text.strip():
            return {}
        query_vector = self.model.encode([text], normalize_embeddings=True)[0]
        result: dict[str, float] = {}
        for asin in candidates:
            index = self.asin_to_index.get(asin)
            if index is not None:
                result[asin] = float(self.vectors[index] @ query_vector)
        return result
