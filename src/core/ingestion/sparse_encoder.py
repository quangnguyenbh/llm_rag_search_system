"""SPLADE sparse vector encoder with TF-IDF fallback."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import structlog

logger = structlog.get_logger()

# Type alias for sparse vector format expected by Qdrant
SparseVector = dict[int, float]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class TFIDFSparseEncoder:
    """TF-IDF based sparse encoder used when SPLADE is unavailable.

    Builds a vocabulary incrementally and returns sparse vectors as
    ``{token_index: tf_idf_weight}`` dicts compatible with Qdrant's
    ``SparseVector`` format.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._doc_freq: Counter[str] = Counter()
        self._num_docs: int = 0

    def _index(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def fit(self, corpus: list[str]) -> TFIDFSparseEncoder:
        """Compute document frequencies from *corpus*."""
        self._num_docs = len(corpus)
        for text in corpus:
            unique_tokens = set(_tokenize(text))
            for tok in unique_tokens:
                self._doc_freq[tok] += 1
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> list[SparseVector]:
        """Return TF-IDF sparse vectors for each text in *texts*."""
        results: list[SparseVector] = []
        num_docs = max(self._num_docs, 1)

        for text in texts:
            tokens = _tokenize(text)
            if not tokens:
                results.append({})
                continue

            tf: Counter[str] = Counter(tokens)
            length = len(tokens)
            vec: SparseVector = {}
            for tok, count in tf.items():
                tf_val = count / length
                df = self._doc_freq.get(tok, 0)
                idf = math.log((num_docs + 1) / (df + 1)) + 1.0
                weight = tf_val * idf
                if weight > 0:
                    idx = self._index(tok)
                    vec[idx] = round(weight, 6)
            results.append(vec)

        return results


class SparseEncoder:
    """SPLADE sparse vector encoder.

    Attempts to load the ``naver/splade-cocondenser-ensembledistil`` model
    via *fastembed* or *transformers*.  Falls back to :class:`TFIDFSparseEncoder`
    if neither is available or the model cannot be loaded.

    Parameters
    ----------
    batch_size:
        Number of texts to process per forward pass (default 64).
    """

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self._backend: str = "tfidf"
        self._model: Any = None
        self._tokenizer: Any = None
        self._tfidf = TFIDFSparseEncoder()
        self._try_load_splade()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_splade(self) -> None:
        """Try loading SPLADE via fastembed, then transformers, else TF-IDF."""
        # Try fastembed first (lighter-weight)
        try:
            from fastembed import SparseTextEmbedding  # type: ignore[import]

            self._model = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1",
                batch_size=self.batch_size,
            )
            self._backend = "fastembed"
            logger.info("sparse_encoder.backend", backend="fastembed")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("sparse_encoder.fastembed_unavailable", error=str(exc))

        # Try transformers (heavier but standard)
        try:
            import torch  # type: ignore[import]
            from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore[import]

            model_name = "naver/splade-cocondenser-ensembledistil"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name)
            self._model.eval()
            self._backend = "transformers"
            logger.info("sparse_encoder.backend", backend="transformers")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("sparse_encoder.transformers_unavailable", error=str(exc))

        # Fallback
        logger.warning("sparse_encoder.using_tfidf_fallback")
        self._backend = "tfidf"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[SparseVector]:
        """Encode *texts* into sparse vectors.

        Returns a list of ``{token_index: weight}`` dicts suitable for
        Qdrant's ``SparseVector`` format.
        """
        bs = batch_size or self.batch_size
        if not texts:
            return []

        if self._backend == "fastembed":
            return self._encode_fastembed(texts, bs)
        if self._backend == "transformers":
            return self._encode_transformers(texts, bs)
        return self._encode_tfidf(texts, bs)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _encode_fastembed(self, texts: list[str], batch_size: int) -> list[SparseVector]:
        results: list[SparseVector] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = list(self._model.embed(batch))
            for emb in embeddings:
                # fastembed SparseEmbedding has .indices and .values
                vec: SparseVector = {
                    int(idx): float(val)
                    for idx, val in zip(emb.indices, emb.values, strict=True)
                }
                results.append(vec)
        logger.info("sparse_encoder.encoded", backend="fastembed", count=len(results))
        return results

    def _encode_transformers(self, texts: list[str], batch_size: int) -> list[SparseVector]:
        import torch  # type: ignore[import]

        results: list[SparseVector] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                output = self._model(**inputs)
            # SPLADE: max-pool log(1 + ReLU(logits)) over tokens
            logits = output.logits  # (batch, seq, vocab)
            sparse = torch.log1p(torch.relu(logits))
            sparse_vec, _ = torch.max(sparse * inputs["attention_mask"].unsqueeze(-1), dim=1)

            for row in sparse_vec:
                nonzero_idx = row.nonzero(as_tuple=False).squeeze(-1).tolist()
                vec: SparseVector = {
                    int(idx): float(row[idx])
                    for idx in nonzero_idx
                    if float(row[idx]) > 1e-4
                }
                results.append(vec)
        logger.info("sparse_encoder.encoded", backend="transformers", count=len(results))
        return results

    def _encode_tfidf(self, texts: list[str], batch_size: int) -> list[SparseVector]:
        # Fit on the provided texts if vocab is empty
        if not self._tfidf._vocab:
            self._tfidf.fit(texts)
        results = self._tfidf.encode(texts, batch_size=batch_size)
        logger.info("sparse_encoder.encoded", backend="tfidf", count=len(results))
        return results
