from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class MatchResult:
    sheet: str
    defect_id: str
    label: str
    cause_effect: str
    score: float       # raw cosine (0..1)
    confidence: float  # calibrated (0..1)
    item: Dict[str, Any]

def _prep_text(s: str) -> str:
    return (s or "").replace("\u00a0", " ").strip()

def _combine_fields(obs_text: str, cause_text: str, a_obs: float = 0.3, a_cause: float = 0.7) -> str:
    # Weighted concat: emphasize cause/effect.
    obs = _prep_text(obs_text)
    cause = _prep_text(cause_text)
    return ((cause + " ") * int(a_cause * 10)) + ((obs + " ") * int(a_obs * 10))

def build_sheet_index(items: List[Dict[str, Any]], sheet_prefix: str) -> List[Dict[str, Any]]:
    sp = (sheet_prefix or "").strip()
    return [it for it in items if (it.get("sheet_key") or it.get("sheet") or "").startswith(sp)]

class CauseEffectMatcher:
    """
    TF-IDF (char_wb 3–5 grams) over Cause/Effect (+ light Observation).
    Instantiate per-sheet; reuse for speed.
    """
    def __init__(self, catalog_rows: List[Dict[str, Any]]):
        self.rows = catalog_rows
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3,5),
            min_df=1
        )
        self.search_strings = []
        for it in self.rows:
            label_terms = it.get("label_terms") or " ".join(filter(None, [it.get("label", ""), " ".join(it.get("synonyms", []))]))
            self.search_strings.append(_combine_fields(label_terms, it.get("cause_effect", "")))
        self.mat = self.vectorizer.fit_transform(self.search_strings)

    def match(self, extracted_obs: str, extracted_cause: str, top_k: int = 3) -> List[MatchResult]:
        query = _combine_fields(extracted_obs, extracted_cause)
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.mat).ravel()
        order = sims.argsort()[::-1][:max(top_k, 1)]
        top = [(idx, float(sims[idx])) for idx in order]

        # Confidence = raw cosine, nudged by margin and penalized if cause text is short.
        results: List[MatchResult] = []
        runner_up = top[1][1] if len(top) > 1 else 0.0
        best_len = max(1, len(_prep_text(extracted_cause)))
        len_penalty = 1.0 if best_len >= 160 else (0.7 + 0.3 * min(1.0, best_len / 160.0))
        for idx, raw in top:
            margin = max(0.0, raw - runner_up)
            conf = 0.85 * raw + 0.15 * min(1.0, margin / 0.15)
            conf *= len_penalty
            conf = max(0.0, min(1.0, conf))
            it = self.rows[idx]
            results.append(MatchResult(
                sheet=it.get("sheet",""),
                defect_id=it.get("id",""),
                label=it.get("label",""),
                cause_effect=it.get("cause_effect",""),
                score=raw,
                confidence=conf,
                item=dict(it)
            ))
        return results
