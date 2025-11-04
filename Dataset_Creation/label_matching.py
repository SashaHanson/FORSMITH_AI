# ------------------------------------------------------------
# Label taxonomy loading + observation parsing + fuzzy match
# Encapsulated in LabelMatcher for clean use by image_extraction.
#
# Behavior per spec:
# - observation_raw is empty when no explicit Observation section is found.
# - When no explicit Observation is found, label_source="full text" and the
#   entire cleaned right_text_raw is used for matching/review.
# - discussion_or_description is suppressed (empty string).
# - label_source is only "observation" or "full text" (no "none").
# - observation_category -> observation_id:
#     * If a specific label is matched, emit its JSON id (e.g., 6.01.05).
#     * If only a sheet is inferred, emit the sheet title string (e.g., "6.0 - Shingle").
#     * If the best is "Other" within a sheet, emit that sheet's "Other" id (e.g., 7.09.01) when available.
# - right_text_raw is normalized: fix 'â€“' -> '-', normalize dashes, and
#   collapse mid-sentence newlines (keep newline only after '.').
#
# Matching:
# - Load ALL label items (no collapsing by label text).
# - Fuzzy match against distinct items with a small bias to the current sheet.
#
# Back-compat:
# - load_labels_from_json(json_path, extended=False) returns the original 3-tuple
#   (taxonomy_by_canonical, labels_by_category, all_labels_canonical).
# - If extended=True, returns the 7-tuple (..., other_id_by_sheet, sheet_title_by_key,
#   total_labels, label_items). main.py uses extended=True.
# - LabelMatcher __init__ accepts label_items (full distinct rows).
# - Returned dict also includes "raw_right_text" and "observation_category" shims.
# ------------------------------------------------------------

from typing import Dict, Tuple, Set, Optional, List, Any
from pathlib import Path
from collections import defaultdict
import json, re, logging
from label_inference import LightInferencer, DEFAULT_RULES
from cause_effect_matcher import CauseEffectMatcher, build_sheet_index
from semantic_inferencer import SemanticInferencer, SemanticMatchResult

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:
    _rf_fuzz = None

# ---------------- Canonicalization helpers ----------------
def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_cat_key(cat_raw: str) -> str:
    s = (cat_raw or "").strip()
    m = re.match(r"^\s*([1-7])(?:[\.\)]\s*|(?:\s+|$))", s)
    if m:
        return f"{m.group(1)}.0"
    return s

# ---------------- Text normalization for raw panels ----------------
def _normalize_raw_text(s: str) -> str:
    """
    Fix mojibake and collapse newlines to improve readability and parsing:

      - Replace 'â€“' with '-' (mojibake)
      - Also normalize real en/em dashes to '-' for consistency
      - Keep a newline ONLY if the last non-whitespace character
        immediately before it is a period '.'. Otherwise, replace
        newline with a single space.
    """
    if not s:
        return ""
    t = str(s).replace("\r\n", "\n").replace("\r", "\n")

    # Fix dash variants
    t = t.replace("â€“", "-")
    t = t.replace("–", "-").replace("—", "-")

    out = []
    last_non_ws = None
    for ch in t:
        if ch == "\n":
            if last_non_ws == ".":
                out.append("\n")
            else:
                out.append(" ")
        else:
            out.append(ch)
            if not ch.isspace():
                last_non_ws = ch

    joined = "".join(out)

    # Collapse multiple spaces that might have been introduced
    joined = re.sub(r"[ \t]{2,}", " ", joined)

    # Trim trailing spaces per line, keep intentional newlines after periods
    joined = "\n".join(line.rstrip() for line in joined.split("\n")).strip()
    return joined

# ---------------- Load labels JSON (no collapsing by text) ----------------
# We build:
# - label_items:        list of distinct items [{label,id,sheet_key,sheet_title,canon}, ...]
# - labels_by_category: sheet_key -> set(canon label strings)  (for priors/inference only)
# - other_id_by_sheet:  sheet_key -> the 'Other' id if present
# - sheet_title_by_key: sheet_key -> original sheet title (e.g., "6.0 - Shingle")
# Also synthesize a minimal taxonomy_by_canonical for back-compat APIs.
def load_labels_from_json(json_path, *, extended: bool = False):
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Labels JSON not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "items" in raw:
        items = raw["items"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Unexpected labels JSON structure: expected dict with 'items' or a list.")

    total_labels = len(items)

    label_items: List[dict] = []
    labels_by_category: Dict[str, Set[str]] = defaultdict(set)
    other_id_by_sheet: Dict[str, str] = {}
    sheet_title_by_key: Dict[str, str] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        lab = str(item.get("label", "")).strip()
        if not lab:
            continue
        lab_id = str(item.get("id", "")).strip()  # e.g., "6.01.05"
        sheet_title = str(item.get("sheet", item.get("category", ""))).strip()  # e.g., "6.0 - Shingle"
        sheet_key = _normalize_cat_key(sheet_title)
        can = _canon(lab)
        normalized_synonyms: List[str] = []
        syn_canons: List[str] = []
        match_canons = [can]
        label_terms = lab

        label_items.append({
            "label": lab,
            "id": lab_id,
            "sheet": sheet_title,
            "sheet_key": sheet_key,
            "sheet_title": sheet_title,
            "canon": can,
            "synonyms": normalized_synonyms,
            "syn_canons": syn_canons,
            "match_canons": match_canons,
            "label_terms": label_terms,
        })

        if sheet_key:
            labels_by_category[sheet_key].add(can)
            sheet_title_by_key[sheet_key] = sheet_title

        if can == "other" and lab_id:
            other_id_by_sheet[sheet_key] = lab_id

    # For backward compatibility with callers expecting:
    # (taxonomy_by_canonical, labels_by_category, all_labels_canonical)
    taxonomy_by_canonical: Dict[str, Dict[str, str]] = {}
    for li in label_items:
        can = li["canon"]
        if can not in taxonomy_by_canonical:
            taxonomy_by_canonical[can] = {
                "label": li["label"],
                "id": li["id"],
                "sheet_key": li["sheet_key"],
                "sheet_title": li["sheet_title"],
            }
    all_labels_canonical = set(taxonomy_by_canonical.keys())

    if extended:
        return (
            taxonomy_by_canonical,
            labels_by_category,
            all_labels_canonical,
            other_id_by_sheet,
            sheet_title_by_key,
            total_labels,
            label_items,  # distinct rows; use this for matching
        )
    else:
        return (
            taxonomy_by_canonical,
            labels_by_category,
            all_labels_canonical,
        )

# ---------------- Parsing structured fields ----------------
FIELD_TOKENS = [
    "Observation", "Observations",
    "Discussion", "Description", "Issue",
    "Recommendation", "Location", "Priority",
    "Cause/Effect", "Photograph"
]
PHOTO_ID_RX = re.compile(r"(?i)\bphotograph\s*[:\-–—]?\s*([0-9]+(?:\.[0-9]+)?)")

OBS_REGEX = re.compile(r"^\s*observations?\s*:?\s*$", re.I)
DISC_REGEX = re.compile(r"^\s*(discussion|description)\s*:?\s*$", re.I)
RECO_REGEX = re.compile(r"^\s*recommendations?\s*:?\s*$", re.I)

def parse_structured_fields(right_text: str) -> Dict[str, str]:
    out = {}
    if not right_text:
        return out

    # CLEAN FIRST so heading-based parsing sees a single, tidy paragraph
    t = _normalize_raw_text(right_text)

    headings_rx = r"(?:{})".format("|".join(re.escape(h) for h in FIELD_TOKENS))
    next_heading_lookahead = rf"(?=\b{headings_rx}\b\s*[:\-]|$)"

    def grab(name: str):
        if name.lower().endswith("s"):
            name_rx = re.escape(name)
        else:
            name_rx = re.escape(name) + "s?"
        rx = re.compile(rf"(?is)\b{name_rx}\b\s*[:\-]\s*(.+?){next_heading_lookahead}")
        m = rx.search(t);  return m.group(1).strip() if m else None

    out["observation"] = grab("Observation") or grab("Observations")
    # We intentionally do NOT emit discussion/description anymore (suppressed downstream)
    out["_raw"] = t.strip()
    # NEW: also stash the Cause/Effect (or discussion/description) raw text for CE matching
    out["cause_effect_raw"] = _extract_section(out["_raw"]) or ""
    return out

def parse_panel_sections(text: str):
    # CLEAN FIRST so the fallback uses normalized lines
    text = _normalize_raw_text(text or "")
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    # Minimalistic fallback: if no headings, treat first non-empty line(s) as observation text candidate
    if not lines:
        return {"observation": None}
    return {"observation": "\n".join(lines).strip()}

SECTION_HDR = re.compile(
    r'(?mi)^\s*(Observation|Discussion|Description|Cause\s*/?\s*Effect|Cause\s*&\s*Effect|Recommendations?)\s*:?\s*$'
)

def _extract_section(panel_text: str, want=("cause/effect",)) -> str:
    """
    Slice the right-panel into sections by headings and return the requested block.
    - Primary: "Cause/Effect" (various spellings)
    - Fallback: "Discussion" or "Description"
    """
    if not panel_text: 
        return ""
    # collect sections
    spans = []
    for m in SECTION_HDR.finditer(panel_text):
        spans.append((m.start(), m.end(), m.group(1).lower()))
    if not spans:
        return ""
    spans.append((len(panel_text), len(panel_text), "_end"))

    # build map: header -> text until next header
    blocks = {}
    for i in range(len(spans)-1):
        _, end_hdr, name = spans[i]
        start_next, _, _ = spans[i+1]
        body = panel_text[end_hdr:start_next].strip()
        blocks[name] = body

    # normalize keys and choose
    ce_keys = {"cause/effect", "cause & effect", "cause effect", "cause/effects", "cause & effects"}
    for k in list(blocks.keys()):
        nk = k.replace("&", "/").replace("  ", " ").replace("  ", " ").strip()
        if nk in {"discussion", "description"}:
            blocks["discussion"] = blocks[k]
        if any(nk.startswith(x.split()[0]) for x in ce_keys) or "cause" in nk:
            blocks["cause/effect"] = blocks[k]

    # priority: cause/effect → discussion/description → ""
    return blocks.get("cause/effect") or blocks.get("discussion") or ""

# ---------------- Matching against DISTINCT items ----------------
def _token_set_ratio(a: str, b: str) -> float:
    A = set(_canon(a).split())
    B = set(_canon(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _normalize_plural_token(token: str) -> str:
    token = token.strip()
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _normalized_phrase_tokens(phrase: str) -> List[str]:
    return [_normalize_plural_token(tok) for tok in phrase.split() if tok]


def _containment_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = _normalized_phrase_tokens(a)
    tokens_b = _normalized_phrase_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    if tokens_a == tokens_b:
        return 0.99

    def _has_contiguous(sub, full):
        n = len(sub)
        if n > len(full) or n == 0:
            return False
        for i in range(len(full) - n + 1):
            if full[i:i + n] == sub:
                return True
        return False

    if _has_contiguous(tokens_a, tokens_b):
        shorter, longer = tokens_a, tokens_b
    elif _has_contiguous(tokens_b, tokens_a):
        shorter, longer = tokens_b, tokens_a
    else:
        return 0.0

    shorter_len = sum(len(t) for t in shorter)
    longer_len = max(sum(len(t) for t in longer), 1)
    ratio = shorter_len / longer_len
    if ratio < 0.45:
        return 0.0
    return 0.90 + 0.09 * min(1.0, ratio)


def _match_label_item(text: str,
                      label_items: List[dict],
                      sheet_hint: Optional[str],
                      strict: float = 0.90,
                      loose: float = 0.80):
    """Return (best_item_or_None, best_score) by fuzzy match, with a small bonus for matching the current sheet."""
    if not text:
        return None, 0.0

    C = _canon(text)
    if not C:
        return None, 0.0

    best_item, best_score = None, -1.0
    for it in label_items:
        candidates = it.get("match_canons") or [it.get("canon", "")]
        best_local = -1.0
        for cand in candidates:
            if not cand:
                continue
            score = (_rf_fuzz.token_set_ratio(C, cand) / 100.0) if _rf_fuzz else _token_set_ratio(C, cand)
            contain_score = _containment_score(C, cand)
            if contain_score > score:
                score = contain_score
            if score > best_local:
                best_local = score
        if best_local < 0:
            continue
        base = best_local
        if base < 0.95:
            syn_canons = it.get("syn_canons", [])
            text_tokens = set(_normalized_phrase_tokens(C))
            for syn in syn_canons:
                syn_tokens = set(_normalized_phrase_tokens(syn))
                if syn_tokens and syn_tokens.issubset(text_tokens):
                    base = max(base, 0.95)
                    break
        # small, safe bias toward the current sheet (if known)
        if sheet_hint and it.get("sheet_key") == sheet_hint:
            base += 0.03
        if base > best_score:
            best_score, best_item = base, it

    if best_score >= strict:
        return best_item, best_score
    if best_score >= loose:
        return best_item, best_score
    return None, best_score

KW_PRIORS = {
    "2.0": ["membrane","blister","lap","seam","puncture","fishmouth","fastener","base sheet","cap sheet"],
    "3.0": ["debris","vegetation","organic","algae","stain","surface","fines","gravel","granule"],
    "4.0": ["parapet","coping","counterflashing","edge metal","termination bar","curb","wall"],
    "5.0": ["drain","scupper","gutter","leader","downspout","overflow","ponding","sump"],
    "6.0": ["penetration","pipe","vent","stack","conduit","pitch pocket","equipment","flashing","shingle","tile"],
    "7.0": ["safety","guardrail","access","ladder","fall","tie-off","hatch","metal"],
    "1.0": ["general","deck","structure","insulation","moisture","wet","slope","taper","thermal"],
}
SEC_REGEX  = re.compile(r"^\s*([1-7])\.(\d+)\s+(.+)$")

def infer_sheet_title(panel_text: str, page_text_above: str, current_section: str,
                      labels_by_category, sheet_title_by_key) -> Tuple[str, float, List[str]]:
    """Return (sheet_key, confidence, reasons)."""
    reasons = []
    if current_section and current_section in labels_by_category:
        reasons.append(f"Section prior {current_section}")
        return current_section, 0.85, reasons

    m = SEC_REGEX.search(page_text_above or "")
    if m:
        cat = f"{m.group(1)}.0"
        if cat in labels_by_category:
            reasons.append(f"Nearest heading {m.group(0)} → {cat}")
            return cat, 0.75, reasons

    text = f"{panel_text or ''}".lower()
    scores = {cat: 0 for cat in KW_PRIORS.keys() if cat in labels_by_category}
    for cat, kws in KW_PRIORS.items():
        if cat not in scores: continue
        for k in kws:
            if k in text:
                scores[cat] += 1
    if scores:
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            reasons.append(f"Keyword prior {best} (hits={scores[best]})")
            return best, 0.60 + min(0.2, 0.05 * scores[best]), reasons

    # Fallback to any known sheet (stable, lowest confidence)
    if sheet_title_by_key:
        any_key = next(iter(sheet_title_by_key.keys()))
        return any_key, 0.50, ["Fallback default"]

    return "Unknown", 0.40, ["No sheet context"]

def score_linkage(region_used: str, vertical_overlap_ratio: float, text_density: float, has_headings: bool, penalties: float):
    base = {"narrow": 0.35, "wide": 0.20, "nearest": 0.10}.get(region_used, 0.10)
    s = base + min(0.25, 0.25 * vertical_overlap_ratio) + min(0.15, 0.15 * text_density)
    if has_headings:
        s += 0.15
    s = max(0.0, min(1.0, s - penalties))
    return s

# ---------------- LabelMatcher ----------------
class LabelMatcher:
    def __init__(self,
                 taxonomy_by_canonical,
                 labels_by_category,
                 all_labels_canonical,
                 other_id_by_sheet: Optional[Dict[str, str]] = None,
                 sheet_title_by_key: Optional[Dict[str, str]] = None,
                 cfg=None,
                 label_items: Optional[List[dict]] = None):
        # --- existing fields ---
        self.taxonomy_by_canonical = taxonomy_by_canonical
        self.labels_by_category = labels_by_category
        self.all_labels_canonical = all_labels_canonical
        self.other_id_by_sheet = other_id_by_sheet or {}
        self.sheet_title_by_key = sheet_title_by_key or {}
        self.label_items = label_items or []   # DISTINCT items (no collapsing)

        # Lightweight rule-based helper you already use
        self.light = LightInferencer(self.label_items, rules=DEFAULT_RULES)

        # --- cfg: merge user cfg over sensible defaults ---
        default_cfg = {
            "fuzzy_strict": 0.90,
            "fuzzy_loose":  0.80,
            "header_ymax":  90,
            "footer_ymin_from_bottom": 80,
            "min_panel_width": 80,
            # optional: carry currently selected sheet prefix (e.g., "2.0")
            "sheet_prefix": None,
            "accept_min_sim": 0.80,       # treat obs label "low-confidence" below this
            "obs_min_conf_for_ce": 0.80,  # if obs < 0.80, run CE cross-check
            # neural retrieval + rerank settings
            "neural_enabled": True,
            "neural_device": None,  # auto-select
            "neural_embedding_model": "intfloat/e5-large-v2",
            "neural_cross_encoder_model": "cross-encoder/nli-deberta-v3-large",
            "neural_cache_dir": ".cache/label_embeddings",
            "neural_embedding_cache_name": None,
            "neural_top_k": 25,
            "neural_retrieval_batch_size": 64,
            "neural_cross_batch_size": 8,
            "neural_min_retrieval_score": 0.15,
            "neural_min_cross_score": 0.45,
            "neural_confidence_floor": 0.55,
            "neural_sheet_bias": 0.03,
            "neural_refresh_embeddings": False,
            "neural_accept_min_confidence": 0.60,
            "neural_confidence_calibration": None,
        }
        self.cfg = {**default_cfg, **(cfg or {})}

        self.logger = logger

        self.semantic_inferencer: Optional[SemanticInferencer] = None
        if self.cfg.get("neural_enabled") and self.label_items:
            try:
                self.semantic_inferencer = SemanticInferencer(self.label_items, self.cfg)
            except Exception as exc:
                self.logger.warning("Semantic inferencer disabled: %s", exc)

        # --- NEW: store the full catalog & a per-sheet matcher cache ---
        # all_items is the canonical catalog used by the Cause/Effect fallback
        self.all_items = self.label_items

        # carry sheet prefix (if caller passes it in cfg) for convenience
        sp = self.cfg.get("sheet_prefix")
        self.current_sheet_prefix = (str(sp).strip() if sp else None)

        # small cache: key = sheet prefix (e.g., "2.0"), value = CauseEffectMatcher instance
        # (created lazily where you call the fallback)
        self._ce_cache: Dict[str, Any] = {}

    @staticmethod
    def _derive_label_and_flags(right_text: str) -> Dict:
        # CLEAN FIRST so everything downstream sees normalized text
        cleaned = _normalize_raw_text(right_text or "")

        # Try to extract an explicit Observation section and Cause/Effect text
        sections = parse_structured_fields(cleaned)
        obs_section = (sections.get("observation") or "").strip()

        # If not found via heading, fallback to first non-empty line(s)
        if not obs_section:
            fallback = parse_panel_sections(cleaned or "")
            obs_section = (fallback.get("observation") or "").strip()

        # Remove leading "Photograph x.y" if present
        if obs_section:
            obs_lines = [ln for ln in obs_section.splitlines() if ln.strip()]
            if obs_lines and re.match(r"(?i)^\s*photograph\b", obs_lines[0]):
                obs_section = "\n".join(obs_lines[1:]).strip()

        # Determine label_source:
        # - if we truly have an "Observation:"-style section, label_source="observation"
        # - else label_source="full text" and we'll use the entire cleaned text to attempt a match
        has_true_observation = bool(sections.get("observation"))
        label_source = "observation" if has_true_observation else "full text"

        # observation_raw: keep ONLY when we actually found an Observation section
        observation_raw = obs_section if has_true_observation else ""

        # photo id (informational)
        m = PHOTO_ID_RX.search(cleaned or "")
        photo_id = m.group(1).strip() if m else None

        cause_effect_raw = (sections.get("cause_effect_raw") or "").strip()
        panel_raw = str(sections.get("_raw") or cleaned)

        return {
            "observation_raw": observation_raw,   # empty if we didn't see a real Observation section
            "label_source": label_source,         # "observation" or "full text"
            "raw_right_text": cleaned,            # always store CLEANED text
            "photo_id": photo_id,
            "cause_effect_raw": cause_effect_raw, # extracted Cause/Effect (or discussion/description) text
            "panel_raw": panel_raw,               # normalized right-panel text for CE fallbacks
        }

    def compute_label(self, right_text: str, page_text: str, current_section: Optional[str],
                      img_bbox, right_bbox, page_h: float) -> Dict:
        # Derive base fields and cleaned text
        fields = self._derive_label_and_flags(right_text)
        cleaned_text = fields.get("raw_right_text", right_text or "")
        label_source = fields.get("label_source", "full text")
        observation_raw = fields.get("observation_raw", "")

        panel_raw = str(fields.get("panel_raw") or cleaned_text or "")
        cached_cause_text = None

        def _get_ce_matcher(prefix: str):
            key = prefix or "__ALL__"
            cem = self._ce_cache.get(key)
            if cem is None:
                rows = build_sheet_index(self.all_items, prefix) if prefix else self.all_items
                if rows:
                    cem = CauseEffectMatcher(rows)
                    self._ce_cache[key] = cem
            return cem

        def _strip_observation(block: str, obs_text: str) -> str:
            block = (block or "").strip()
            obs_text = (obs_text or "").strip()
            if not block:
                return ""
            if not obs_text:
                return block
            lower_block = block.lower()
            lower_obs = obs_text.lower()
            idx = lower_block.find(lower_obs)
            if idx == -1:
                return block
            before = block[:idx].strip()
            after = block[idx + len(obs_text):].strip()
            pieces = [seg for seg in (before, after) if seg]
            return "\n".join(pieces).strip()

        def _select_cause_text() -> str:
            nonlocal cached_cause_text
            if cached_cause_text is not None:
                return cached_cause_text
            candidate = (fields.get("cause_effect_raw") or "").strip()
            if not candidate:
                candidate = (_extract_section(panel_raw) or "").strip()
            if not candidate:
                candidate = _strip_observation(panel_raw, observation_raw)
            if not candidate:
                candidate = panel_raw.strip()
            cached_cause_text = candidate
            return cached_cause_text

        # Choose source text for matching
        source_text = observation_raw if label_source == "observation" else cleaned_text

        # Label selection method tracking
        label_method = "fuzzy"   # default; overwritten if light-infer or fallback kicks in
        label_method_info = ""   # optional small note for debugging (e.g., matched keywords)
        alt1_label = alt1_score = alt2_label = alt2_score = ""

        # ---- MATCH AGAINST DISTINCT ITEMS (tiered: rules → BM25 → TF-IDF → fallback fuzz) ----
        # ---- MATCHING STACK (rules -> neural retrieval -> cross-encoder -> fuzzy) ----
        strict = self.cfg.get("fuzzy_strict", 0.90)
        loose  = self.cfg.get("fuzzy_loose", 0.80)

        best_item = None
        sim = 0.0
        semantic_result: Optional[SemanticMatchResult] = None

        if label_source == "full text" and self.label_items:
            light = self.light.infer(source_text, sheet_hint=current_section)
            if light.get("hit"):
                # Accept immediately; map by id if possible, else by label text
                best_item = next((it for it in self.label_items if it.get("id","") == light.get("id","")), None)
                if best_item is None:
                    best_item = next((it for it in self.label_items if it.get("label","").lower() == light["label"].lower()), None)
                # Give a synthetic similarity consistent with your thresholds
                sim = 0.92 if light["method"] == "rules" else (0.88 if light["method"] == "bm25" else 0.85)

                # === Record which method was used + debug info ===
                label_method = f"light:{light.get('method','')}"  # 'light:rules' | 'light:bm25' | 'light:tfidf'
                if light.get("method") == "rules" and light.get("debug", {}).get("matched"):
                    mk = light["debug"]["matched"]
                    parts = []
                    for k in ("must","any","effects","actions"):
                        if mk.get(k):
                            parts.append(f"{k}=" + "|".join(mk[k][:4]))
                    label_method_info = "; ".join(parts)[:160]

                # === Record alternate candidates (for debugging/evaluation) ===
                alts = light.get("alt_candidates") or []
                if len(alts) > 0:
                    alt1_label, alt1_score = alts[0]["label"], round(float(alts[0]["score"]), 3)
                if len(alts) > 1:
                    alt2_label, alt2_score = alts[1]["label"], round(float(alts[1]["score"]), 3)

        sheet_hint_for_semantic = current_section or self.cfg.get("sheet_prefix") or getattr(self, "current_sheet_prefix", None)
        if best_item is None and self.semantic_inferencer:
            if not self.semantic_inferencer.is_available():
                disabled_reason = self.semantic_inferencer.disabled_reason()
                if disabled_reason and not label_method_info:
                    label_method_info = disabled_reason[:160]
            else:
                try:
                    semantic_result = self.semantic_inferencer.match(
                        source_text,
                        sheet_hint=sheet_hint_for_semantic,
                        top_k=self.cfg.get("neural_top_k"),
                    )
                except Exception as exc:
                    self.logger.warning("Semantic inferencer failed: %s", exc)
                    semantic_result = None

                if semantic_result and semantic_result.best:
                    min_neural_conf = float(self.cfg.get("neural_accept_min_confidence", 0.60))
                    if semantic_result.best.confidence >= min_neural_conf:
                        best_item = dict(semantic_result.best.item)
                        sim = float(semantic_result.best.confidence)
                        label_method = "semantic_crossencoder"
                        label_method_info = semantic_result.debug_summary()[:160]
                        if len(semantic_result.candidates) > 1:
                            alt1_label = semantic_result.candidates[1].item.get("label", "")
                            alt1_score = round(float(semantic_result.candidates[1].confidence), 3)
                        if len(semantic_result.candidates) > 2:
                            alt2_label = semantic_result.candidates[2].item.get("label", "")
                            alt2_score = round(float(semantic_result.candidates[2].confidence), 3)
                    else:
                        info = semantic_result.debug_summary()
                        if info:
                            label_method_info = info[:160]

        # If the lightweight tiers didn't confidently hit, use the existing fuzzy token_set matching
        if best_item is None:
            best_item, sim = _match_label_item(source_text, self.label_items, current_section, strict, loose)
        elif sim < self.cfg.get("accept_min_sim", 0.75):
            best_item, sim = None, 0.0  # force CE fallback to run

        # --- NEW: If we matched via Observation but it's weak, cross-check with Cause/Effect ---
        obs_ce_threshold = float(self.cfg.get("obs_min_conf_for_ce", 0.80))
        if label_source == "observation" and sim < obs_ce_threshold:
            sheet_prefix = (self.cfg.get("sheet_prefix") or getattr(self, "current_sheet_prefix", "") or "").strip()
            extracted_obs_text = (observation_raw or "").strip()
            if not extracted_obs_text:
                extracted_obs_text = source_text.strip()
            extracted_cause_text = _select_cause_text().strip()

            if extracted_cause_text:
                cem = _get_ce_matcher(sheet_prefix) or _get_ce_matcher(None)
                if cem is not None:
                    ce_matches = cem.match(extracted_obs_text, extracted_cause_text, top_k=3)
                    if ce_matches:
                        ce_best = ce_matches[0]
                        ce_conf = float(ce_best.confidence)
                        obs_conf = float(sim)
                        if ce_conf > obs_conf:
                            row = ce_best.item if ce_best.item else {
                                "sheet": ce_best.sheet,
                                "id": ce_best.defect_id,
                                "label": ce_best.label,
                                "cause_effect": ce_best.cause_effect,
                            }
                            best_item = dict(row)
                            sim = ce_conf
                            label_method = "cause_effect_crosscheck"
                            label_method_info = f"raw={ce_best.score:.3f}; ce={ce_conf:.3f}; obs={obs_conf:.3f}"
                        elif obs_conf:
                            label_method_info = f"obs={obs_conf:.3f}; ce={ce_conf:.3f}"

        flag_review = False

        # --- Cause/Effect similarity fallback (per-sheet, calibrated) ---
        if best_item is None:
            # pull the user-selected sheet (e.g., "2.0") if you passed it via cfg/__init__
            sheet_prefix = (self.cfg.get("sheet_prefix") or getattr(self, "current_sheet_prefix", "") or "").strip()

            # Use your already-parsed texts (Observation + Cause/Effect heuristics)
            extracted_obs_text = (observation_raw or "").strip()
            if not extracted_obs_text:
                extracted_obs_text = source_text.strip()
            extracted_cause_text = _select_cause_text().strip()

            if extracted_cause_text:
                cem = _get_ce_matcher(sheet_prefix) or _get_ce_matcher(None)
                if cem is not None:
                    ce_matches = cem.match(extracted_obs_text, extracted_cause_text, top_k=3)
                    if ce_matches and ce_matches[0].confidence >= 0.40:  # tune 0.35–0.50 as needed
                        ce_best = ce_matches[0]
                        row = ce_best.item if ce_best.item else {
                            "sheet": ce_best.sheet,
                            "id": ce_best.defect_id,
                            "label": ce_best.label,
                            "cause_effect": ce_best.cause_effect,
                        }
                        best_item = dict(row)
                        # Treat CE confidence as similarity so downstream logic works unchanged
                        sim = float(ce_best.confidence)
                        label_method = "cause_effect_fallback"
                        label_method_info = f"raw={ce_best.score:.3f}; conf={ce_best.confidence:.3f}"

        flag_reasons: List[str] = []

        if best_item:
            obs_label = best_item["label"]
            observation_id = best_item.get("id", "") or ""
            sheet_key = best_item.get("sheet_key")
            sheet_title = best_item.get("sheet_title")
            confidence_label = sim
        else:
            # infer sheet
            sheet_key, cat_conf, reasons = infer_sheet_title(
                cleaned_text, page_text, current_section,
                self.labels_by_category, self.sheet_title_by_key
            )
            sheet_title = self.sheet_title_by_key.get(sheet_key, sheet_key)
            confidence_label = min(0.65, cat_conf)
            other_id = self.other_id_by_sheet.get(sheet_key)
            if other_id:
                observation_id = other_id
                obs_label = "Other"
                flag_reasons += [f"Routed to sheet 'Other' id {other_id}"]
            else:
                observation_id = sheet_title or "Unknown"
                obs_label = "Other"
            flag_review = True
            flag_reasons = ["No exact label match"] + flag_reasons

            label_method = "sheet_fallback"
            label_method_info = "; ".join(flag_reasons)[:160]

        # Linkage confidence (geometry-based)
        if right_bbox is not None:
            p_top, p_bot = right_bbox.y0, right_bbox.y1
            region_used = "narrow"
        else:
            p_top, p_bot = img_bbox.y0, img_bbox.y1
            region_used = "nearest"

        iy0, iy1 = img_bbox.y0, img_bbox.y1
        vo = max(0, min(iy1, p_bot) - max(iy0, p_top))
        vh = max(1, (iy1 - iy0))
        vertical_overlap_ratio = vo / vh

        text_density = 0.0  # neutral default

        penalties = 0.0
        if p_top < self.cfg.get("header_ymax", 90):
            penalties += 0.15
        if (page_h - p_bot) < self.cfg.get("footer_ymin_from_bottom", 80):
            penalties += 0.15
        if right_bbox is not None and (right_bbox.x1 - right_bbox.x0) < self.cfg.get("min_panel_width", 80):
            penalties += 0.10

        has_headings = bool(
            OBS_REGEX.search(cleaned_text or "") or
            DISC_REGEX.search(cleaned_text or "") or
            RECO_REGEX.search(cleaned_text or "")
        )
        confidence_linkage = score_linkage(region_used, vertical_overlap_ratio, text_density, has_headings, penalties)
        if confidence_linkage < 0.35:
            flag_review = True
            flag_reasons.append(f"Low linkage confidence ({confidence_linkage:.2f})")

        return {
            # keep the cleaned raw text for QA; used when label_source="full text"
            "right_text_raw": cleaned_text,
            "raw_right_text": cleaned_text,  # back-compat shim for any legacy caller

            # explicit observation section text only if present
            "observation_raw": observation_raw,

            # suppressed per spec (kept as empty string to avoid breaking CSV schema)
            "discussion_or_description": "",

            # where the label came from
            "label_source": label_source,  # "observation" or "full text"

            # report's "Photograph x.y" if present
            "photo_id": fields.get("photo_id"),

            # chosen label text (may be "Other" when unresolved)
            "observation_label": obs_label,

            # NEW field replacing observation_category
            "observation_id": observation_id,  # exact id when known, else sheet title string

            # optional shim for legacy code that still expects observation_category
            "observation_category": sheet_title or "",

            # confidences
            "confidence_label": round(float(confidence_label), 3),
            "confidence_linkage": round(float(confidence_linkage), 3),

            # flags
            "flag_review": bool(flag_review),
            "flag_reason": "; ".join(flag_reasons),

            # labeling method info
            "label_method": label_method,  # 'light:rules' | 'light:bm25' | 'light:tfidf' | 'semantic_crossencoder' | 'fuzzy' | 'cause_effect_crosscheck' | 'cause_effect_fallback' | 'sheet_fallback'
            "label_method_info": label_method_info,  # small hint like matched keywords or fallback reason

            # top alternative labels (for analysis)
            "alt1_label": alt1_label, "alt1_score": alt1_score,
            "alt2_label": alt2_label, "alt2_score": alt2_score,
        }
