"""TF-IDF retrieval over `knowledge_base.KNOWLEDGE_BASE` -- the "R" in this
module's RAG.

Deliberately not embedding-based (no sentence-transformers/FAISS, unlike
e.g. MA-IDS's all-MiniLM-L6-v2 + FAISS approach). `ml/README.md` already
documents that PyTorch is unusable in the environment this whole project
was built in (blocked by the host's Application Control/WDAC policy at
DLL load time -- see that README's sequence-model section for the exact
error), and sentence-transformers depends on it. scikit-learn's TF-IDF
vectorizer + cosine similarity needs only numpy/scipy (both already
dependencies elsewhere in this project) and is a reasonable, honestly-
motivated substitute for a knowledge base this small (single digits of
entries) -- the retrieval quality difference between TF-IDF and dense
embeddings matters far more at corpus sizes where semantic-but-not-lexical
matches are common, which isn't the regime a handful of curated technique
descriptions is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import KNOWLEDGE_BASE, TechniqueEntry


@dataclass
class RetrievalResult:
    entry: TechniqueEntry
    score: float  # cosine similarity in [0, 1]


class TechniqueRetriever:
    """Fits a TF-IDF index over `KNOWLEDGE_BASE`'s technique descriptions
    at construction time (the knowledge base is small and static within a
    process lifetime, so there's no separate "rebuild index" step)."""

    def __init__(self, knowledge_base: Optional[List[TechniqueEntry]] = None) -> None:
        self.entries = knowledge_base if knowledge_base is not None else KNOWLEDGE_BASE
        if not self.entries:
            raise ValueError("knowledge base must be non-empty")
        # `category` is included in the indexed text (not just name/
        # description) because reasoner.py's primary query is the alert's
        # exact `stage2_predicted_class` string (e.g. "PortScan") -- and
        # that label doesn't necessarily appear verbatim in a technique's
        # prose description (T1046's description says "port scan" as two
        # words, not "PortScan"). Without this, an exact-category query
        # can retrieve the wrong entry entirely; caught by
        # test_tier2_reasoner.py's missing-optional-fields test.
        corpus = [f"{e.category}. {e.name}. {e.description}" for e in self.entries]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[RetrievalResult]:
        """Returns up to `top_k` entries most similar to `query`, sorted by
        score descending, filtered to `score >= min_score`. An empty list
        (not an error) means nothing in the knowledge base is a plausible
        match -- the reasoner treats that as "no grounding available" and
        says so explicitly rather than letting the LLM guess ungrounded.
        """
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [RetrievalResult(entry=self.entries[i], score=float(scores[i])) for i in order if scores[i] >= min_score]

    def retrieve_by_category(self, category: str, top_k: int = 3, min_score: float = 0.0) -> List[RetrievalResult]:
        """Convenience wrapper: retrieves using the category name itself as
        the query (e.g. "Brute Force"), which is normally the strongest
        available signal since `stage2_predicted_class` already names the
        family -- see `reasoner.py` for how this composes with a richer
        query built from the alert's SHAP explanation.
        """
        return self.retrieve(category, top_k=top_k, min_score=min_score)
