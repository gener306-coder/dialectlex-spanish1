"""
dialectlex.py — DialectLex: Dialectal enrichment strategy for Spanish NLP.

Enriches Spanish texts with regional dialect tags ([MX], [CO], [AR], etc.)
based on a vocabulary built from the BILMA corpus.

Usage:
    from src.dialectlex import DialectLex
    dlx = DialectLex(bilma_dir="path/to/bilma/")
    enriched = dlx.enrich("ese wey llegó tarde")
    # "[MX] ese wey llegó tarde"
"""

import os
import re
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Supported dialect tags
DIALECTS = {
    "MX": "México",
    "AR": "Argentina",
    "CL": "Chile",
    "CO": "Colombia",
    "ES": "España",
    "PE": "Perú",
    "US": "Estados Unidos",
}


class DialectLex:
    """
    DialectLex enrichment strategy.

    Identifies dialectal markers in Spanish text using a vocabulary
    built from BILMA and prepends a regional tag to the input.

    Parameters
    ----------
    bilma_dir : str
        Path to the directory containing BILMA TSV files.
        Files must be named: MX.tsv, AR.tsv, CL.tsv, CO.tsv, ES.tsv, PE.tsv, US.tsv
        Each TSV has columns: token, occs, ndocs, weight
    min_freq : int
        Minimum term frequency to include a token (default: 5).
    min_len : int
        Minimum token length in characters (default: 2).
    max_len : int
        Maximum token length in characters (default: 30).
    vocabulary : dict, optional
        Pre-built vocabulary dict. If provided, bilma_dir is ignored.
    """

    def __init__(
        self,
        bilma_dir: Optional[str] = None,
        min_freq: int = 5,
        min_len: int = 2,
        max_len: int = 30,
        vocabulary: Optional[dict] = None,
    ):
        self.min_freq = min_freq
        self.min_len = min_len
        self.max_len = max_len

        if vocabulary is not None:
            self.vocabulary = vocabulary
            logger.info(f"DialectLex loaded with {len(vocabulary):,} terms.")
        elif bilma_dir is not None:
            self.vocabulary = self._build_vocabulary(bilma_dir)
        else:
            raise ValueError("Provide either bilma_dir or vocabulary.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def enrich(self, text: str) -> str:
        """
        Enrich a single text with a dialect tag.

        Parameters
        ----------
        text : str
            Input text in Spanish.

        Returns
        -------
        str
            Text with dialect tag prepended, or original text if no
            dialect markers were detected.
        """
        tag = self._detect_dialect(text)
        if tag:
            return f"[{tag}] {text}"
        return text

    def enrich_batch(self, texts: list) -> list:
        """
        Enrich a list of texts with dialect tags.

        Parameters
        ----------
        texts : list of str
            Input texts in Spanish.

        Returns
        -------
        list of str
            Enriched texts.
        """
        return [self.enrich(t) for t in texts]

    def get_dialect_score(self, token: str, dialect: str) -> float:
        """
        Return the dialectal score of a token for a given dialect.

        Parameters
        ----------
        token : str
            Token to look up.
        dialect : str
            Dialect code (e.g., "MX", "AR").

        Returns
        -------
        float
            Dialectal score, or 0.0 if not found.
        """
        token = token.lower()
        if token in self.vocabulary and dialect in self.vocabulary[token]:
            return self.vocabulary[token][dialect]
        return 0.0

    def get_vocabulary_size(self) -> int:
        """Return the total number of terms in the vocabulary."""
        return len(self.vocabulary)

    # ──────────────────────────────────────────────────────────────────────────
    # Private methods
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vocabulary(self, bilma_dir: str) -> dict:
        """
        Build dialectal vocabulary from BILMA TSV files.

        Score formula: log(1 + occs) × weight
        """
        vocabulary = {}
        total_terms = 0

        for dialect in DIALECTS:
            filepath = os.path.join(bilma_dir, f"{dialect}.tsv")
            if not os.path.exists(filepath):
                logger.warning(f"BILMA file not found: {filepath}")
                continue

            dialect_terms = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 4:
                        continue

                    token, occs_str, _, weight_str = parts[:4]

                    # Filters
                    try:
                        occs   = int(occs_str)
                        weight = float(weight_str)
                    except ValueError:
                        continue

                    if occs < self.min_freq:
                        continue
                    if weight <= 0:
                        continue
                    if not (self.min_len <= len(token) <= self.max_len):
                        continue
                    if not re.match(r"^[a-záéíóúüñ]+$", token.lower()):
                        continue

                    # Dialectal score
                    score = math.log(1 + occs) * weight
                    token_lower = token.lower()

                    if token_lower not in vocabulary:
                        vocabulary[token_lower] = {}
                    vocabulary[token_lower][dialect] = score
                    dialect_terms += 1

            total_terms += dialect_terms
            logger.info(f"  {dialect}: {dialect_terms:,} terms loaded.")

        logger.info(f"Vocabulary built: {len(vocabulary):,} unique terms "
                    f"({total_terms:,} total entries).")
        return vocabulary

    def _detect_dialect(self, text: str) -> Optional[str]:
        """
        Detect the dominant dialect of a text.

        Returns the dialect code with the highest cumulative score,
        or None if no dialectal markers are found.
        """
        tokens = re.findall(r"\b[a-záéíóúüñ]+\b", text.lower())
        scores = {d: 0.0 for d in DIALECTS}

        for token in tokens:
            if token in self.vocabulary:
                for dialect, score in self.vocabulary[token].items():
                    scores[dialect] += score

        best_dialect = max(scores, key=scores.get)
        if scores[best_dialect] > 0:
            return best_dialect
        return None
