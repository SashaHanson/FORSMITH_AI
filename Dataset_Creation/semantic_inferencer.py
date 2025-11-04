import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    safe = []
    for ch in text.strip().lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in {".", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "model"


def _ensure_device(preferred: Optional[str]) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is mandatory at runtime
        raise RuntimeError(
            "PyTorch is required for semantic label inference. "
            "Install torch before enabling the semantic pipeline."
        ) from exc

    if preferred:
        return preferred
    if torch.cuda.is_available():  # pragma: no cover - device probing
        return "cuda"
    if torch.backends.mps.is_available():  # pragma: no cover - macOS
        return "mps"
    return "cpu"


def _load_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "sentence-transformers is required for semantic label retrieval "
            f"(failed to import while loading '{model_name}'). "
            "Install sentence-transformers>=2.2.2."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def _load_cross_encoder(model_name: str, device: str):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "transformers is required for cross-encoder reranking "
            f"(failed to import while loading '{model_name}'). "
            "Install transformers>=4.35."
        ) from exc

    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _softmax_entailment(logits, entailment_index: int = 2) -> float:
    import torch

    probs = torch.softmax(logits, dim=-1)
    score = float(probs[..., entailment_index].cpu().item())
    return max(0.0, min(1.0, score))


def _candidate_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    label = str(item.get("label") or "").strip()
    if label:
        parts.append(label)
    synonyms: Sequence[str] = item.get("synonyms") or []
    if synonyms:
        parts.extend(s.strip() for s in synonyms if s)
    sheet = str(item.get("sheet_title") or item.get("sheet") or "").strip()
    if sheet and sheet.lower() not in label.lower():
        parts.append(sheet)
    extra = str(item.get("label_terms") or "").strip()
    if extra and extra.lower() not in label.lower():
        parts.append(extra)
    return " [SEP] ".join(p for p in parts if p)


def _catalog_signature(items: Sequence[Dict[str, Any]]) -> str:
    import hashlib

    key_parts = []
    for it in items:
        label = str(it.get("label") or "")
        ident = str(it.get("id") or "")
        sheet = str(it.get("sheet_key") or "")
        text = _candidate_text(it)
        key_parts.append({"label": label, "id": ident, "sheet": sheet, "text": text})
    payload = json.dumps(key_parts, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


@dataclass
class SemanticMatchCandidate:
    item: Dict[str, Any]
    retrieval_score: float
    rerank_score: float
    confidence: float
    rank: int

    def as_info(self) -> str:
        return (
            f"{self.item.get('label','')} "
            f"(retr={self.retrieval_score:.3f}, entail={self.rerank_score:.3f}, conf={self.confidence:.3f})"
        )


@dataclass
class SemanticMatchResult:
    best: Optional[SemanticMatchCandidate]
    candidates: List[SemanticMatchCandidate]

    def debug_summary(self, limit: int = 3) -> str:
        if not self.candidates:
            return ""
        tops = ", ".join(c.as_info() for c in self.candidates[:limit])
        return f"semantic_topk=[{tops}]"


class SemanticInferencer:
    """
    Two-stage semantic matcher:
      1. bi-encoder retrieval (E5 large) over cached label embeddings.
      2. Cross-encoder reranking (DeBERTa v3 NLI) to produce calibrated confidence.
    """

    def __init__(self, label_items: Sequence[Dict[str, Any]], cfg: Dict[str, Any]):
        self.items: List[Dict[str, Any]] = list(label_items or [])
        self.cfg = cfg or {}

        self.embedding_model = self.cfg.get("neural_embedding_model", "intfloat/e5-large-v2")
        self.cross_model = self.cfg.get("neural_cross_encoder_model", "cross-encoder/nli-deberta-v3-large")
        self.top_k = int(self.cfg.get("neural_top_k", 25))
        self.retrieve_batch = int(self.cfg.get("neural_retrieval_batch_size", 64))
        self.cross_batch = int(self.cfg.get("neural_cross_batch_size", 8))
        self.min_retrieval = float(self.cfg.get("neural_min_retrieval_score", 0.15))
        self.min_cross = float(self.cfg.get("neural_min_cross_score", 0.45))
        self.confidence_floor = float(self.cfg.get("neural_confidence_floor", 0.55))
        self.refresh = bool(self.cfg.get("neural_refresh_embeddings", False))
        self.sheet_bias = float(self.cfg.get("neural_sheet_bias", 0.03))
        self.calibration = self.cfg.get("neural_confidence_calibration") or {}

        self._available = True
        self._disabled_reason: Optional[str] = None

        self.device = "cpu"
        try:
            self.device = _ensure_device(self.cfg.get("neural_device"))
        except RuntimeError as exc:
            self._disable(str(exc))

        if self._available:
            self._check_dependency(
                "sentence_transformers",
                "sentence-transformers>=2.2.2 is required for semantic label retrieval.",
            )
        if self._available:
            self._check_dependency(
                "transformers",
                "transformers>=4.35 is required for semantic cross-encoder reranking.",
            )

        cache_root = Path(self.cfg.get("neural_cache_dir") or ".cache/label_embeddings")
        cache_name = self.cfg.get("neural_embedding_cache_name")
        if not cache_name:
            cache_name = f"{_slugify(self.embedding_model)}.pt"
        self.cache_path = cache_root / cache_name
        if self._available:
            try:
                cache_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - cache dir failure
                logger.warning("Unable to create semantic cache directory %s: %s", cache_root, exc)

        self._label_texts = [_candidate_text(it) for it in self.items]
        self._passages = [f"passage: {txt}" if txt else "passage: " for txt in self._label_texts]
        self._signature = _catalog_signature(self.items)

        self._encoder = None
        self._embeddings = None
        self._cross_tokenizer = None
        self._cross_model = None

        if self._available:
            logger.debug(
                "Initialized SemanticInferencer with %d labels (cache=%s)",
                len(self.items),
                self.cache_path,
            )

    # ----- Public API -------------------------------------------------
    def match(self, query_text: str, *, sheet_hint: Optional[str] = None, top_k: Optional[int] = None) -> SemanticMatchResult:
        query_text = (query_text or "").strip()
        if not query_text or not self.items:
            return SemanticMatchResult(best=None, candidates=[])
        if not self._available:
            return SemanticMatchResult(best=None, candidates=[])

        embed_matrix = self._ensure_embeddings()
        if embed_matrix is None:
            return SemanticMatchResult(best=None, candidates=[])

        query_vec = self._encode_query(query_text)
        if query_vec is None:
            return SemanticMatchResult(best=None, candidates=[])

        import torch

        # Cosine similarity via dot product (embeddings are normalized)
        sims = torch.matmul(embed_matrix, query_vec)
        sims = sims.detach().cpu()

        effective_top_k = min(int(top_k or self.top_k), len(self.items))
        if effective_top_k <= 0:
            return SemanticMatchResult(best=None, candidates=[])

        # Prefer sheet matches first, then fall back to global ordering.
        ordered_indices = torch.argsort(sims, descending=True).tolist()
        selected: List[int] = []
        if sheet_hint:
            for idx in ordered_indices:
                if len(selected) >= effective_top_k:
                    break
                if str(self.items[idx].get("sheet_key") or "").lower() == str(sheet_hint).lower():
                    selected.append(idx)
            if len(selected) < effective_top_k:
                for idx in ordered_indices:
                    if idx in selected:
                        continue
                    selected.append(idx)
                    if len(selected) >= effective_top_k:
                        break
        else:
            selected = ordered_indices[:effective_top_k]

        candidates: List[SemanticMatchCandidate] = []
        cross_inputs: List[str] = []
        retrieval_scores: List[float] = []
        for rank, idx in enumerate(selected):
            score = float(sims[idx])
            if score < self.min_retrieval and rank > 0:
                continue
            candidates.append(
                SemanticMatchCandidate(
                    item=self.items[idx],
                    retrieval_score=score,
                    rerank_score=0.0,
                    confidence=0.0,
                    rank=rank + 1,
                )
            )
            cross_inputs.append(self._label_texts[idx])
            retrieval_scores.append(score)

        if not candidates:
            return SemanticMatchResult(best=None, candidates=[])

        entail_scores = self._rerank_with_cross_encoder(query_text, cross_inputs)
        for cand, retr, entail in zip(candidates, retrieval_scores, entail_scores):
            rerank_score = entail
            adjusted = self._calibrate_confidence(entail)
            cand.rerank_score = rerank_score
            cand.confidence = max(self.confidence_floor if rerank_score >= self.min_cross else 0.0, adjusted)
            if sheet_hint:
                same_sheet = str(cand.item.get("sheet_key") or "").lower() == str(sheet_hint).lower()
                if same_sheet:
                    cand.confidence = min(1.0, cand.confidence + self.sheet_bias)
                    cand.rerank_score = min(1.0, cand.rerank_score + self.sheet_bias)
        candidates.sort(key=lambda c: (c.confidence, c.rerank_score, c.retrieval_score), reverse=True)
        best = candidates[0] if candidates and candidates[0].confidence >= self.min_cross else None
        return SemanticMatchResult(best=best, candidates=candidates)

    def is_available(self) -> bool:
        return self._available

    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    # ----- Internal helpers -------------------------------------------
    def _disable(self, reason: str):
        if not self._available:
            return
        self._available = False
        self._disabled_reason = str(reason)
        logger.warning("Semantic inferencer disabled: %s", reason)

    def _check_dependency(self, module: str, message: str) -> bool:
        try:
            __import__(module)
            return True
        except ImportError as exc:
            self._disable(f"{message} ({exc})")
            return False

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        if not self.refresh and self._load_cached_embeddings():
            return self._embeddings
        return self._compute_and_cache_embeddings()

    def _load_cached_embeddings(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            import torch

            payload = torch.load(self.cache_path, map_location="cpu")
            signature = payload.get("signature")
            if signature != self._signature:
                logger.info("Semantic embedding cache signature mismatch; recomputing.")
                return False
            emb = payload.get("embeddings")
            if emb is None:
                return False
            self._embeddings = emb.to(self.device)
            return True
        except Exception as exc:  # pragma: no cover - cache load failure
            logger.warning("Failed to load semantic embedding cache (%s): %s", self.cache_path, exc)
            return False

    def _compute_and_cache_embeddings(self):
        encoder = self._get_encoder()
        import torch

        texts = self._passages
        with torch.inference_mode():
            embeddings = encoder.encode(
                texts,
                batch_size=self.retrieve_batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        embeddings = embeddings.to(self.device)
        self._embeddings = embeddings
        try:
            import torch

            payload = {
                "signature": self._signature,
                "embeddings": embeddings.cpu(),
            }
            torch.save(payload, self.cache_path)
        except Exception as exc:  # pragma: no cover - cache save failure
            logger.warning("Could not persist semantic embeddings to %s: %s", self.cache_path, exc)
        return self._embeddings

    def _encode_query(self, text: str):
        encoder = self._get_encoder()
        import torch

        query = f"query: {text}"
        with torch.inference_mode():
            vec = encoder.encode(
                [query],
                batch_size=1,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vec[0].to(self.device)

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_sentence_transformer(self.embedding_model, self.device)
        return self._encoder

    def _get_cross_components(self):
        if self._cross_model is None or self._cross_tokenizer is None:
            tokenizer, model = _load_cross_encoder(self.cross_model, self.device)
            self._cross_tokenizer = tokenizer
            self._cross_model = model
        return self._cross_tokenizer, self._cross_model

    def _rerank_with_cross_encoder(self, query: str, candidates: Iterable[str]) -> List[float]:
        tokenizer, model = self._get_cross_components()
        import torch

        texts = list(candidates)
        scores: List[float] = []
        if not texts:
            return scores

        for i in range(0, len(texts), self.cross_batch):
            batch_texts = texts[i : i + self.cross_batch]
            encoded = tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
            for j in range(logits.size(0)):
                scores.append(_softmax_entailment(logits[j]))
        return scores

    def _calibrate_confidence(self, prob: float) -> float:
        scale = float(self.calibration.get("scale", 1.0))
        bias = float(self.calibration.get("bias", 0.0))
        out = (prob * scale) + bias
        if math.isnan(out) or math.isinf(out):
            return prob
        return max(0.0, min(1.0, out))
