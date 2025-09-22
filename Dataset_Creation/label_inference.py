# label_infer_light.py
# Lightweight, free label inference:
# Tier 1: Rules  → Tier 2: BM25  → Tier 3: TF-IDF cosine
# Uses your existing label_items (full distinct rows with label/id/sheet).

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math, re

# ---------- Basic text utils ----------
def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # dash and mojibake cleanup to mirror your normalizer
    s = s.replace("â€“", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok(s: str) -> List[str]:
    return _norm(s).split()

# ---------- Rules engine ----------
@dataclass
class RuleDef:
    must: List[str]
    any: List[str]
    effects: List[str]
    actions: List[str]
    thresh: int = 3
    gap: int = 2

class RuleEngine:
    def __init__(self, rules_by_label: Dict[str, RuleDef]):
        self.rules = rules_by_label

    def predict(self, text: str) -> Optional[Tuple[str, int, Dict[str, List[str]]]]:
        t = " " + _norm(text) + " "
        scored = []
        for label, r in self.rules.items():
            # must: all must appear
            if r.must and not all(re.search(rf"\b{re.escape(m)}\b", t) for m in r.must):
                continue
            score = 0
            matched = {"must": [], "any": [], "effects": [], "actions": []}
            for m in r.must:
                if re.search(rf"\b{re.escape(m)}\b", t): matched["must"].append(m); score += 2
            for m in r.any:
                if re.search(rf"\b{re.escape(m)}\b", t): matched["any"].append(m); score += 1
            for m in r.effects:
                if re.search(rf"\b{re.escape(m)}\b", t): matched["effects"].append(m); score += 1
            for m in r.actions:
                if re.search(rf"\b{re.escape(m)}\b", t): matched["actions"].append(m); score += 1
            if score:
                scored.append((label, score, matched))
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        winner, w, _ = scored[0]
        runner = scored[1][1] if len(scored) > 1 else -1
        thresh = self.rules[winner].thresh
        gap = self.rules[winner].gap
        if w >= thresh and (runner == -1 or (w - runner) >= gap):
            return scored[0]
        return None

DEFAULT_RULES: Dict[str, RuleDef] = {
    # Seed a few; expand as you iterate. Example from your prompt:
    "Drain Plumbing Clogged": RuleDef(
        must=["drain"],
        any=["drainage","screen","strainer","clog","blocked","obstructed","debris","gravel","ballast"],
        effects=["pond","ponding","standing","standing water","slow drain"],
        actions=["clear","clean","remove","vacuum"]
    ),
    "Debris": RuleDef(
        must=[],
        any=["debris","trash","leaves","vegetation","sediment","gravel","ballast"],
        effects=["blocked","impede","flow","pond","ponding"],
        actions=[]
    ),
    # Add more label-specific rules as needed.
}

# ---------- BM25 ----------
class BM25:
    def __init__(self, docs_tokens: List[List[str]], k1=1.5, b=0.75):
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        self.k1, self.b = k1, b
        self.df = Counter()
        for d in docs_tokens:
            for w in set(d):
                self.df[w] += 1
        self.idf = {w: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()}

    def _score_one(self, q: List[str], i: int) -> float:
        doc = self.docs[i]
        if not doc: return 0.0
        freqs = Counter(doc)
        score = 0.0
        for w in q:
            f = freqs.get(w, 0)
            if not f: continue
            idf = self.idf.get(w, 0.0)
            tf = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            score += idf * tf
        return score

    def rank(self, q: List[str]) -> List[Tuple[int, float]]:
        scores = [(i, self._score_one(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# ---------- TF-IDF cosine ----------
class TfIdf:
    def __init__(self, docs: List[List[str]]):
        self.docs = docs
        self.N = len(docs)
        self.vocab = {}
        for d in docs:
            for w in d:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        self.df = [0] * len(self.vocab)
        for d in docs:
            for w in set(d):
                self.df[self.vocab[w]] += 1
        self.idf = [math.log((self.N + 1) / (df + 1)) + 1.0 for df in self.df]
        self.doc_vecs = []
        for d in docs:
            L = max(1, len(d))
            tf = Counter(d)
            vec = [0.0] * len(self.vocab)
            for w, c in tf.items():
                j = self.vocab[w]
                vec[j] = (c / L) * self.idf[j]
            n = math.sqrt(sum(v * v for v in vec)) or 1.0
            self.doc_vecs.append([v / n for v in vec])

    def _vec(self, toks: List[str]) -> List[float]:
        L = max(1, len(toks))
        tf = Counter(toks)
        v = [0.0] * len(self.vocab)
        for w, c in tf.items():
            j = self.vocab.get(w)
            if j is None: continue
            v[j] = (c / L) * self.idf[j]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def rank(self, q: List[str]) -> List[Tuple[int, float]]:
        qv = self._vec(q)
        scores = []
        for i, dv in enumerate(self.doc_vecs):
            s = sum(a * b for a, b in zip(qv, dv))
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# ---------- Public inferencer ----------
class LightInferencer:
    def __init__(self,
                 label_items: List[Dict[str, Any]],
                 rules: Dict[str, RuleDef] = None,
                 bm25_thresh: float = 3.0,
                 bm25_gap: float = 0.15,
                 tfidf_thresh: float = 0.25,
                 tfidf_gap: float = 0.10):
        # Store distinct items & build corpora
        self.items = label_items[:]  # expects dicts with label/id/sheet_key/canon/label_terms metadata
        for it in self.items:
            if "canon" not in it:
                it["canon"] = _norm(it.get("label", ""))
        self.labels = [it["label"] for it in self.items]
        def _label_context(itm):
            parts = []
            label_terms = itm.get("label_terms") or itm.get("label") or ""
            if label_terms:
                parts.append(label_terms)
            sheet_part = itm.get("sheet_key", "")
            if sheet_part:
                parts.append(sheet_part)
            return " ".join(parts)
        self.tokens = [_tok(_label_context(it)) for it in self.items]

        # IR backends
        self.bm25 = BM25(self.tokens)
        self.tfidf = TfIdf(self.tokens)

        # Rules
        self.rules = RuleEngine(rules or DEFAULT_RULES)

        # Thresholds
        self.bm25_thresh = bm25_thresh
        self.bm25_gap = bm25_gap
        self.tfidf_thresh = tfidf_thresh
        self.tfidf_gap = tfidf_gap

    def _ok(self, ranked: List[Tuple[int, float]], abs_thresh: float, gap: float) -> Optional[Tuple[int, float]]:
        if not ranked: return None
        top_i, top_s = ranked[0]
        if top_s < abs_thresh: return None
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if second >= 0 and (top_s - second) < gap * max(1e-6, abs(top_s)):
            return None
        return (top_i, top_s)

    def infer(self, text: str, sheet_hint: Optional[str] = None) -> Dict[str, Any]:
        # Tier 1: RULES
        r = self.rules.predict(text or "")
        if r:
            label, score, matched = r
            item = next((it for it in self.items if it["label"].lower() == label.lower()), None)
            return {
                "hit": True, "method": "rules", "label": label,
                "id": (item or {}).get("id", ""), "score": float(score),
                "debug": {"matched": matched}
            }

        # Tier 2: BM25
        q = _tok(text or "")
        ranked = self.bm25.rank(q)
        ok = self._ok(ranked, self.bm25_thresh, self.bm25_gap)
        if ok:
            i, s = ok
            it = self.items[i]
            return {"hit": True, "method": "bm25", "label": it["label"], "id": it.get("id",""), "score": float(s), "debug": {}}

        # Tier 3: TF-IDF
        ranked = self.tfidf.rank(q)
        ok = self._ok(ranked, self.tfidf_thresh, self.tfidf_gap)
        if ok:
            i, s = ok
            it = self.items[i]
            return {"hit": True, "method": "tfidf", "label": it["label"], "id": it.get("id",""), "score": float(s), "debug": {}}

        return {"hit": False}
