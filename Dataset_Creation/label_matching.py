# ------------------------------------------------------------
# Label taxonomy loading + observation parsing + fuzzy match
# Encapsulated in LabelMatcher for clean use by image_extraction.
# ------------------------------------------------------------

from typing import Dict, Tuple, Set, Optional, List
from pathlib import Path
from collections import defaultdict
import json, re
try:
    from rapidfuzz import fuzz
except:
    fuzz = None

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

# ---------------- Load labels JSON ----------------
def load_labels_from_json(json_path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Set[str]], Set[str]]:
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

    taxonomy_by_canonical: Dict[str, Dict[str, str]] = {}
    labels_by_category: Dict[str, Set[str]] = defaultdict(set)

    for item in items:
        if not isinstance(item, dict):
            continue
        lab = str(item.get("label", "")).strip()
        if not lab:
            continue
        cat_raw = str(item.get("sheet", item.get("category", ""))).strip()
        cat_key = _normalize_cat_key(cat_raw)
        can = _canon(lab)
        taxonomy_by_canonical[can] = {"label": lab, "category": cat_key}
        if cat_key:
            labels_by_category[cat_key].add(can)

    all_labels_canonical = set(taxonomy_by_canonical.keys())
    return taxonomy_by_canonical, labels_by_category, all_labels_canonical

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
    t = right_text.replace("—", "-").replace("–", "-")
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
    out["discussion"]  = grab("Discussion") or grab("Description")
    out["recommendation"] = grab("Recommendation")
    out["_raw"] = right_text.strip()
    return out

def parse_panel_sections(text: str):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    current = None
    buckets = {"observation": [], "discussion": [], "recommendation": []}
    for l in lines:
        if OBS_REGEX.match(l):
            current = "observation"; continue
        if DISC_REGEX.match(l):
            current = "discussion"; continue
        if RECO_REGEX.match(l):
            current = "recommendation"; continue
        if current:
            buckets[current].append(l)
        else:
            if not buckets["observation"]:
                buckets["observation"].append(l)
            else:
                buckets["discussion"].append(l)
    return {k: ("\n".join(v).strip() if v else None) for k, v in buckets.items()}

# ---------------- Matching + inference ----------------
def token_set_ratio(a: str, b: str) -> float:
    A = set(_canon(a).split())
    B = set(_canon(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def match_label_text(obs_text: str, taxonomy_by_canonical, all_labels_canonical, strict=0.90, loose=0.80):
    if not obs_text:
        return None, 0.0
    c = _canon(obs_text)
    if c in all_labels_canonical:
        lab = taxonomy_by_canonical[c]["label"]
        return lab, 1.0
    best_lab, best_score = None, -1.0
    for can in all_labels_canonical:
        cand_lab = taxonomy_by_canonical[can]["label"]
        score = (fuzz.token_set_ratio(c, can) / 100.0) if fuzz else token_set_ratio(c, can)
        if score > best_score:
            best_score, best_lab = score, cand_lab
    if best_score >= strict:
        return best_lab, best_score
    if best_score >= loose:
        return best_lab, best_score
    return None, best_score

KW_PRIORS = {
    "2.0": ["membrane","blister","lap","seam","puncture","fishmouth","fastener","base sheet","cap sheet"],
    "3.0": ["debris","vegetation","organic","algae","stain","surface","fines","gravel","granule"],
    "4.0": ["parapet","coping","counterflashing","edge metal","termination bar","curb","wall"],
    "5.0": ["drain","scupper","gutter","leader","downspout","overflow","ponding","sump"],
    "6.0": ["penetration","pipe","vent","stack","conduit","pitch pocket","equipment","flashing"],
    "7.0": ["safety","guardrail","access","ladder","fall","tie-off","hatch"],
    "1.0": ["general","deck","structure","insulation","moisture","wet","slope","taper","thermal"],
}
SEC_REGEX  = re.compile(r"^\s*([1-7])\.(\d+)\s+(.+)$")

def infer_category_for_other(panel_text: str, page_text_above: str, current_section: str, labels_by_category):
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

    fallback = "1.0" if "1.0" in labels_by_category else next(iter(labels_by_category.keys()), "Unknown")
    return fallback, 0.50, ["Fallback default"]

def score_linkage(region_used: str, vertical_overlap_ratio: float, text_density: float, has_headings: bool, penalties: float):
    base = {"narrow": 0.35, "wide": 0.20, "nearest": 0.10}.get(region_used, 0.10)
    s = base + min(0.25, 0.25 * vertical_overlap_ratio) + min(0.15, 0.15 * text_density)
    if has_headings:
        s += 0.15
    s = max(0.0, min(1.0, s - penalties))
    return s

# ---------------- LabelMatcher ----------------
class LabelMatcher:
    def __init__(self, taxonomy_by_canonical, labels_by_category, all_labels_canonical, cfg=None):
        self.taxonomy_by_canonical = taxonomy_by_canonical
        self.labels_by_category = labels_by_category
        self.all_labels_canonical = all_labels_canonical
        self.cfg = cfg or {
            "fuzzy_strict": 0.90,
            "fuzzy_loose" : 0.80,
            "header_ymax": 90,
            "footer_ymin_from_bottom": 80,
            "min_panel_width": 80,
        }

    @staticmethod
    def _derive_label_and_flags(right_text: str) -> Dict:
        fields = parse_structured_fields(right_text)
        obs = (fields.get("observation") or "").strip()
        if not obs:
            fallback = parse_panel_sections(right_text or "")
            obs = (fallback.get("observation") or "").strip()
        if obs:
            obs_lines = [ln for ln in obs.splitlines() if ln.strip()]
            if obs_lines and re.match(r"(?i)^\s*photograph\b", obs_lines[0]):
                obs = "\n".join(obs_lines[1:]).strip()

        disc = fields.get("discussion") or ""
        label = obs if obs else None
        flagged = False
        label_source = "observation" if obs else "none"
        if not label:
            flagged = True
        elif label.lower() == "other":
            flagged = True
            label_source = "observation=other"

        m = PHOTO_ID_RX.search(right_text or "")
        photo_id = m.group(1).strip() if m else None

        return {
            "label": label,
            "label_source": label_source,
            "flagged": bool(flagged),
            "discussion_or_description": (disc or "").strip(),
            "raw_right_text": (right_text or "").strip(),
            "photo_id": photo_id
        }

    def _match_observation(self, obs_raw: str):
        return match_label_text(
            obs_raw,
            self.taxonomy_by_canonical,
            self.all_labels_canonical,
            strict=self.cfg.get("fuzzy_strict", 0.90),
            loose=self.cfg.get("fuzzy_loose", 0.80)
        )

    def compute_label(self, right_text: str, page_text: str, current_section: Optional[str],
                      img_bbox, right_bbox, page_h: float) -> Dict:
        fields = self._derive_label_and_flags(right_text)
        obs_raw = fields.get("label") or ""
        label_source = fields.get("label_source", "none")

        obs_label, sim = (None, 0.0)
        if obs_raw:
            obs_label, sim = self._match_observation(obs_raw)
            if not obs_label:
                first_line = obs_raw.splitlines()[0].strip()
                obs_label2, sim2 = self._match_observation(first_line)
                if obs_label2:
                    obs_label, sim = obs_label2, max(sim, sim2)

        flag_review = bool(fields.get("flagged", False))
        flag_reasons: List[str] = []

        if not obs_label:
            cat, cat_conf, reasons = infer_category_for_other(right_text, page_text, current_section, self.labels_by_category)
            obs_label = "Other"
            obs_category = cat
            confidence_label = min(0.65, cat_conf)
            flag_review = True
            flag_reasons += ["Unresolved observation → routed to category 'Other'"] + reasons
        else:
            can = _canon(obs_label)
            obs_category = self.taxonomy_by_canonical[can]["category"]
            confidence_label = sim
            if obs_label.lower() == "other":
                cat, cat_conf, reasons = infer_category_for_other(right_text, page_text, current_section, self.labels_by_category)
                obs_category = cat
                confidence_label = min(confidence_label, cat_conf)
                flag_reasons += ["Label 'Other' → disambiguated category"] + reasons

        # Linkage confidence
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

        text_density = 0.0  # caller can pass blocks_used if needed; neutral default here

        penalties = 0.0
        if p_top < self.cfg.get("header_ymax", 90):
            penalties += 0.15
        if (page_h - p_bot) < self.cfg.get("footer_ymin_from_bottom", 80):
            penalties += 0.15
        if right_bbox is not None and (right_bbox.x1 - right_bbox.x0) < self.cfg.get("min_panel_width", 80):
            penalties += 0.10

        has_headings = bool(OBS_REGEX.search(right_text or "") or DISC_REGEX.search(right_text or "") or RECO_REGEX.search(right_text or ""))
        confidence_linkage = score_linkage(region_used, vertical_overlap_ratio, text_density, has_headings, penalties)
        if confidence_linkage < 0.35:
            flag_review = True
            flag_reasons.append(f"Low linkage confidence ({confidence_linkage:.2f})")

        return {
            "raw_right_text": fields.get("raw_right_text", right_text).strip(),
            "observation_raw": obs_raw,
            "discussion_or_description": fields.get("discussion_or_description", ""),
            "label_source": label_source,
            "photo_id": fields.get("photo_id"),

            "observation_label": obs_label,
            "observation_category": obs_category,
            "confidence_label": round(float(confidence_label), 3),
            "confidence_linkage": round(float(confidence_linkage), 3),

            "flag_review": bool(flag_review),
            "flag_reason": "; ".join(flag_reasons),
        }
