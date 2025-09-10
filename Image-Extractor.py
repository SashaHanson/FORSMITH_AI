# Image-Extractor.py
# ------------------------------------------------------------
# Bulk image extractor for roof reports with robust page gating.
# - DOMAIN AWARE (persistent): only extract while inside ROOF domain (skip BUILDING)
# - SECTION AWARE (persistent): stay in OBS until a top heading clearly changes it
# - Heading detection is GEOMETRY-AWARE: ignores right-column text next to photos
# - HARD-SKIPS: Summary / Condition Assessment / Existing Conditions / System Description, etc.
# - REQUIRE inside-OBS: left-image + right-text geometry; filter header/footer & tiny logos
# - SAFETY VALVE outside-OBS: strong inline obs tokens + weak hints + geometry
# - OBS FALLBACK: if no extractable XObjects, rasterize left column crop
# - TOC / Appendix handling
# - Version-safe image rects; robust RGB save (fixes CMYK/alpha crash)
# - CSV safe write with interactive retry / timestamped fallback
# - Per-report table: total pages, pages with images, total images (custom-sorted IDs)
# ------------------------------------------------------------

import fitz  # PyMuPDF
import os, glob, re, json
import pandas as pd
import datetime
from typing import List, Optional, Dict, Tuple, Set

# === CONFIG: YOUR FOLDERS ===
REPORTS_DIR = r"D:\FORSMITH - AI\Dataset\Reports"
IMAGES_DIR  = r"D:\FORSMITH - AI\Dataset\Images"
CSV_OUTPUT  = r"D:\FORSMITH - AI\Dataset\image_metadata.csv"
RUN_HISTORY_TXT = r"D:\FORSMITH - AI\Dataset\run_summary.txt" # Save a one-run summary (overwrites each run)

# === EXTRACT BEHAVIOR SWITCHES ===
OBS_FALLBACK = False   # <- disable raster fallback crops entirely

# === LOGGING SWITCHES ===
DEBUG_TOC     = True
DEBUG_PAGES   = True
DEBUG_SUMMARY = True
DEBUG_IMAGES  = True
DEBUG_SECTION = True   # prints ENTER/EXIT for sections and domain

# === HEADER / FOOTER + SIZE FILTERS ===
HEADER_BAND_FRAC = 0.10
FOOTER_BAND_FRAC = 0.10
MIN_IMAGE_AREA_FRAC = 0.03   # relaxed from 0.02 to catch smaller embedded photos
LEFT_COLUMN_MAX_FRAC = 0.60  # relaxed from 0.55
RIGHT_TEXT_MIN_FRAC  = 0.38  # relaxed from 0.40
MIN_COMBINED_IMAGE_AREA_FRAC = 0.04 # Minimum combined area of all kept images relative to page (to avoid false negatives)
REQUIRE_OVERLAP_Y = True

# === HEADING GEOMETRY FILTERS ===
HEADING_TOP_BAND_FRAC   = 0.30  # consider headings only in top 20% of page
HEADING_LEFT_MAX_FRAC   = 0.45  # consider left/center blocks only (x0 <= 45% pw)
HEADING_WIDE_MIN_FRAC   = 0.60  # or very wide blocks (>=60% pw)

# === IMAGE BBOX KEEP FILTER (PDF points) ===
KEEP_X0_RANGE = (40, 125)
KEEP_X1_RANGE = (225, 350)
KEEP_Y0_RANGE = (50, 600)
KEEP_Y1_RANGE = (200, 760)

# === TEXT HINT PATTERNS ===
TOC_HEADER_PATTERNS = [r"\btable of contents\b", r"^\s*contents\s*$"]
INTRO_PATTERNS = [r"\bintroduction\b"]

SUMMARY_HEADINGS_STRICT = [
    r"(?mi)^\s*(?:\d+(?:\.\d+)*)?\s*summary\s*(?:&|and)?\s*recommendations\b",
    r"(?mi)^\s*executive\s+summary\b",
    r"(?mi)^\s*overall\s+condition\b",
    r"(?mi)^\s*conclusion[s]?\b",
    r"(?mi)^\s*summary\b",
]

# Disallowed top-of-page sections (line-start forms)
BLOCK_TOP_STRICT = [
    r"(?mi)^\s*condition\s+assessment\b",
    r"(?mi)^\s*roof\s+condition\s+assessment\b",
    r"(?mi)^\s*existing\s+conditions\b",
    r"(?mi)^\s*roof\s+system\s+description\b",
    r"(?mi)^\s*roof\s+composition\b",
]

# Observations/Deficiencies/Recommendations headings (start section)
OBS_SECTION_HEADINGS_STRICT = [
    r"(?mi)^\s*(?:\d+(?:\.\d+)*)?\s*observations?\s*(?:&|and)\s*recommendations?\b",
    r"(?mi)^\s*deficienc(?:y|ies)\s*(?:&|and)?\s*recommendations?\b",
    r"(?mi)^\s*observations?\b",
    r"(?mi)^\s*deficienc(?:y|ies)\b",
    r"(?mi)^\s*recommendations?\b",
]
# Allow prefixed headings in the top lines (e.g., "Roof Section 102 ...: Deficiencies & Recommendations")
OBS_SECTION_HEADINGS_UNANCHORED = [
    r"(?mi)\bobservations?\s*(?:&|and)\s*recommendations?\b",
    r"(?mi)\bdeficienc(?:y|ies)\s*(?:&|and)?\s*recommendations?\b",
]

# Weak hints (used with strong inline tokens when outside OBS)
WEAK_PHOTO_HINTS = [
    r"\bphotograph[s]?\b",
    r"\bphoto[s]?\b",
    r"\bfigure[s]?\b",
    r"(?mi)^\s*item\s+\d+\b",
    r"(?mi)^\s*section\s+\d+\b",
    r"\broof\s*(area|section)\s+\d+\b",
]

# Strong inline observation tokens
STRONG_INLINE_TOKENS = [
    r"\bobservation[s]?\b",
    r"\bdeficienc(?:y|ies)\b",
    r"\brecommendation[s]?\b",
    r"\bdeterioration\b",
    r"\bleak\b",
]

# === LABEL / FIELD PARSING ===
FIELD_TOKENS = ["Observation", "Observations", "Discussion", "Description", "Issue",
                "Recommendation", "Location", "Priority"]
FIELD_PATTERN = r"(?i)\b({})\b\s*[:\-–—]\s*".format("|".join(FIELD_TOKENS))

PHOTO_ID_RX = re.compile(r"(?i)\bphotograph\s*[:\-–—]?\s*([0-9]+(?:\.[0-9]+)?)")
OBS_RX = re.compile(
    r"(?is)\bobservation[s]?\b\s*[:\-–—]\s*(.+?)(?=\b(observation[s]?|discussion|description|issue|recommendation|location|priority)\b\s*[:\-–—]|$)"
)
DISC_OR_DESC_RX = re.compile(
    r"(?is)\b(discussion|description)\b\s*[:\-–—]\s*(.+?)(?=\b(observation[s]?|discussion|description|issue|recommendation|location|priority)\b\s*[:\-–—]|$)"
)


# --- Domain headings (top-level) ---
ROOF_DOMAIN_HEADINGS = [
    r"(?mi)^\s*roof\s+assessment\b",
    r"(?mi)^\s*roof\s+(area|section)\s+\d+\b",
    r"(?mi)^\s*canopy\b",
    r"(?mi)^\s*roof\s+plan\b",
]

BUILDING_DOMAIN_HEADINGS = [
    r"(?mi)^\s*building\s+envelope\b",
    r"(?mi)^\s*(north|south|east|west)\s+elevation\b",
    r"(?mi)^\s*fa(?:ç|c)ade\b",
    r"(?mi)^\s*wall\s+assembly\b",
]

# TOC / Appendix
TOC_LINE_PATTERN = r".{3,}(\.{2,}|\s{2,})\s+(\d{1,4})\s*$"
TOC_APPENDIX_PATTERN = r"\bappendix\b"
APPENDIX_HEADING_START = r"(?mi)^\s*appendix(?:\s+[A-Z0-9]+)?\b"

os.makedirs(IMAGES_DIR, exist_ok=True)

# ------------------------ Text / block helpers ------------------------

def page_text_raw(page) -> str:
    return page.get_text("text") or ""

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def page_text_norm(page) -> str:
    return norm_spaces(page.get_text("text") or "")

def has_header(patterns: List[str], raw: str) -> bool:
    for p in patterns:
        if re.search(p, raw, flags=re.I | re.M):
            return True
    return False

def looks_like_intro(text_norm: str) -> bool:
    return any(re.search(p, text_norm, flags=re.I) for p in INTRO_PATTERNS)

def contains_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text or "", flags=re.I) for p in patterns)

def has_strong_inline_obs_tokens(raw: str) -> bool:
    return contains_any(STRONG_INLINE_TOKENS, raw)

def has_weak_photo_hints(raw: str) -> bool:
    return contains_any(WEAK_PHOTO_HINTS, raw)

def get_text_blocks(page) -> List[Tuple[fitz.Rect, str]]:
    blocks = []
    try:
        for b in page.get_text("blocks") or []:
            if len(b) >= 7 and b[6] == 0 and isinstance(b[4], str):
                blocks.append((fitz.Rect(b[0], b[1], b[2], b[3]), b[4]))
            elif len(b) >= 5 and isinstance(b[4], str):
                blocks.append((fitz.Rect(b[0], b[1], b[2], b[3]), b[4]))
    except Exception:
        pass
    return blocks

def merge_rect(a: fitz.Rect, b: fitz.Rect) -> fitz.Rect:
    r = fitz.Rect(a)
    r |= b
    return r

def collect_right_column_blocks(page) -> List[Tuple[fitz.Rect, str]]:
    """Right column = x0 >= RIGHT_TEXT_MIN_FRAC of page width; exclude header/footer bands."""
    blocks = []
    pw, ph = page.rect.width, page.rect.height
    x_cut = page.rect.x0 + pw * RIGHT_TEXT_MIN_FRAC
    y_top = page.rect.y0 + ph * HEADER_BAND_FRAC
    y_bot = page.rect.y1 - ph * FOOTER_BAND_FRAC

    for rect, text in get_text_blocks(page):
        if rect.x0 >= x_cut and rect.y1 > y_top and rect.y0 < y_bot and (text or "").strip():
            blocks.append((rect, text))
    # sort top→bottom, then left→right for stable concatenation
    blocks.sort(key=lambda rt: (round(rt[0].y0, 2), round(rt[0].x0, 2)))
    return blocks

def vertical_overlap_amount(a: fitz.Rect, b: fitz.Rect) -> float:
    top = max(a.y0, b.y0); bot = min(a.y1, b.y1)
    return max(0.0, bot - top)

def right_text_for_image(page, img_bbox: fitz.Rect, y_pad: float = 6.0, x_pad: float = 6.0) -> Dict:
    """
    Pick right-column text region for an image:
      1) collect right-column blocks
      2) select those that vertically overlap the image (with small padding)
      3) if none, take the nearest-by-center block
      4) CLIP-EXTRACT full text from the merged rect so we don't lose lines
    Returns: {"text": str, "bbox": fitz.Rect | None, "blocks_used": int}
    """
    blocks = collect_right_column_blocks(page)
    if not blocks:
        return {"text": "", "bbox": None, "blocks_used": 0}

    used = []
    # Allow a small vertical tolerance so we don't miss adjacent lines
    img_y0 = img_bbox.y0 - y_pad
    img_y1 = img_bbox.y1 + y_pad

    for rect, _ in blocks:
        top = max(rect.y0, img_y0)
        bot = min(rect.y1, img_y1)
        if (bot - top) > 0:  # overlap with tolerance
            used.append((rect, ""))  # we will re-extract text via clip

    if not used:
        # Nearest by vertical center if nothing overlaps
        cy = (img_bbox.y0 + img_bbox.y1) / 2.0
        rect, _ = min(blocks, key=lambda rt: abs(((rt[0].y0 + rt[0].y1)/2.0) - cy))
        # Slightly expand to the left/right to catch full column
        r = fitz.Rect(rect.x0 - x_pad, rect.y0 - y_pad, rect.x1 + x_pad, rect.y1 + y_pad)
        full = page.get_text("text", clip=r) or ""
        return {"text": full.strip(), "bbox": r, "blocks_used": 1}

    # Merge all overlapping blocks into one region, then CLIP-EXTRACT once
    merged = used[0][0]
    for rect, _ in used[1:]:
        merged = merge_rect(merged, rect)

    # Slightly dilate the merged rect horizontally to capture wrapped lines
    r = fitz.Rect(merged.x0 - x_pad, merged.y0 - y_pad, merged.x1 + x_pad, merged.y1 + y_pad)

    full = page.get_text("text", clip=r) or ""
    return {"text": full.strip(), "bbox": r, "blocks_used": len(used)}

def parse_structured_fields(right_text: str) -> Dict[str, str]:
    """
    Pulls Observation/Discussion/Description/etc. values out of the right text.
    Returns a dict of lowercase field -> value (trimmed).
    """
    out = {}
    if not right_text:
        return out
    # Normalize dashes for simpler matching
    t = right_text.replace("—", "-").replace("–", "-")
    # Extract each field value as 'this until next field header'
    for fld in FIELD_TOKENS:
        rx = re.compile(rf"(?is)\b{re.escape(fld)}s?\b\s*[:\-]\s*(.+?)(?=\b({'|'.join(FIELD_TOKENS)})\b\s*[:\-]|$)")
        m = rx.search(t)
        if m:
            out[fld.lower()] = m.group(1).strip()
    return out

def derive_label_and_flags(right_text: str) -> Dict:
    """
    Rule-set:
    - If Observation exists -> label = its value.
    - If Observation == 'Other' -> flag; include Discussion/Description as context.
    - If no Observation -> label=None; flag; include all right text for manual.
    """
    fields = parse_structured_fields(right_text)
    obs = (fields.get("observation") or fields.get("observations") or "").strip()
    disc = fields.get("discussion") or ""
    desc = fields.get("description") or ""

    label = obs if obs else None
    flagged = False
    label_source = "observation" if obs else "none"
    aux = disc or desc or ""

    if not label:
        flagged = True
    elif label.lower() == "other":
        flagged = True
        label_source = "observation=other"

    # Try to capture photo id if present in the block (useful for manual review)
    m = PHOTO_ID_RX.search(right_text or "")
    photo_id = m.group(1).strip() if m else None

    return {
        "label": label,
        "label_source": label_source,
        "flagged": bool(flagged),
        "discussion_or_description": aux.strip(),
        "raw_right_text": right_text.strip(),
        "photo_id": photo_id
    }

# ------------------ Geometry helpers (version-safe) ------------------

def get_image_rects(page) -> List[Tuple[fitz.Rect, int]]:
    rects = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            for r in page.get_image_rects(xref):
                rects.append((fitz.Rect(r), xref))
        except Exception:
            continue
    return rects

def cluster_rects(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    r = fitz.Rect(rects[0])
    for rr in rects[1:]:
        r |= rr
    return r

def rects_vertical_overlap(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.y1 <= b.y0 or b.y1 <= a.y0)

def is_header_or_footer(bbox: fitz.Rect, page_rect: fitz.Rect) -> bool:
    y_top = page_rect.y0 + page_rect.height * HEADER_BAND_FRAC
    y_bot = page_rect.y1 - page_rect.height * FOOTER_BAND_FRAC
    return bbox.y1 <= y_top or bbox.y0 >= y_bot

def is_tiny_logo(bbox: fitz.Rect, page_rect: fitz.Rect) -> bool:
    area = bbox.get_area()
    page_area = page_rect.get_area()
    return (area / max(1, page_area)) < MIN_IMAGE_AREA_FRAC

# ---------- GEOMETRY-AWARE heading capture (ignores right column) ----

def top_heading_lines(page, n: int = 14) -> List[str]:
    pw = page.rect.width
    ph = page.rect.height
    top_cut = page.rect.y0 + ph * HEADING_TOP_BAND_FRAC
    left_max_x0 = page.rect.x0 + pw * HEADING_LEFT_MAX_FRAC
    wide_min_w  = pw * HEADING_WIDE_MIN_FRAC

    lines: List[str] = []
    for rect, text in get_text_blocks(page):
        if rect.y0 > top_cut:
            continue
        block_w = rect.width
        if rect.x0 <= left_max_x0 or block_w >= wide_min_w:
            for ln in (text or "").splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
                    if len(lines) >= n:
                        return lines
    return lines

def top_heading_text(page, n: int = 14) -> str:
    return "\n".join(top_heading_lines(page, n=n))

# -------------------- TOC detection (multi-page) --------------------

def is_toc_like_page(raw: str) -> bool:
    if not raw:
        return False
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return False
    if has_header(TOC_HEADER_PATTERNS, raw):
        return True
    roof_section_hits = sum(bool(re.search(r"\broof section\s+\d+\b", ln, flags=re.I)) for ln in lines)
    if roof_section_hits >= 3:
        return True
    indented_hits = sum(1 for ln in lines if re.match(r"^\s{4,}\S", ln))
    if indented_hits >= 3:
        return True
    hits = sum(bool(re.search(TOC_LINE_PATTERN, ln)) for ln in lines)
    ratio = hits / max(1, len(lines))
    return ratio >= 0.30 or hits >= 6

def find_toc_range(doc) -> Optional[Tuple[int, int]]:
    n = len(doc)
    if n < 3:
        return None
    upper_scan = min(n, 20)
    start = end = None
    for i in range(1, upper_scan):
        raw = page_text_raw(doc[i])
        if is_toc_like_page(raw):
            if start is None:
                start = i
            end = i
        elif start is not None:
            break
    if start is not None and end is not None and end >= start:
        return (start, end)
    return None

def appendix_start_from_toc(doc, toc_range: Tuple[int, int]) -> Optional[int]:
    if toc_range is None:
        return None
    n = len(doc)
    toc_start, toc_end = toc_range
    appendix_candidates = []
    for i in range(toc_start, toc_end + 1):
        raw = page_text_raw(doc[i])
        for line in raw.splitlines():
            if re.search(TOC_APPENDIX_PATTERN, line, flags=re.I):
                m = re.search(r"(\d{1,4})\s*$", line)
                if m:
                    try:
                        pg = int(m.group(1))
                        if 1 <= pg <= n:
                            appendix_candidates.append(pg - 1)
                    except ValueError:
                        pass
    return min(appendix_candidates) if appendix_candidates else None

def fallback_find_appendix_start(doc) -> Optional[int]:
    n = len(doc)
    for i in range(n):
        if i <= 1:
            continue
        raw = page_text_raw(doc[i])
        if is_toc_like_page(raw):
            continue
        if re.search(APPENDIX_HEADING_START, raw):
            return i
    return None

# ---------------- Layout check: left image / right text -------------

def page_has_left_image_right_text_layout(page) -> Tuple[bool, Dict]:
    info = {}
    page_rect = page.rect
    pw = page_rect.width
    page_area = page_rect.get_area()

    # 1) collect raw image rects
    img_rects_xref = get_image_rects(page)
    text_rects = [r for r, _ in get_text_blocks(page)]

    if not img_rects_xref or not text_rects:
        info["reason"] = "no image or no text blocks"
        return False, info

    # 2) drop only header/footer first (do NOT tiny-filter yet: images may be tiled)
    kept_imgs = []
    for bbox, xref in img_rects_xref:
        if is_header_or_footer(bbox, page_rect):
            continue
        kept_imgs.append((bbox, xref))

    if not kept_imgs:
        info["reason"] = "all images were header/footer"
        return False, info

    # 3) cluster/union all remaining image tiles, then check size once
    img_rects = [bb for bb, _ in kept_imgs]
    img_cluster = cluster_rects(img_rects)
    text_cluster = cluster_rects(text_rects)
    if not img_cluster or not text_cluster:
        info["reason"] = "no clusters"
        return False, info

    # reject only if the combined image area is still truly tiny
    if (img_cluster.get_area() / max(1.0, page_area)) < MIN_COMBINED_IMAGE_AREA_FRAC:
        info["reason"] = "combined-image-area tiny"
        return False, info

    # 4) original left-image / right-text geometry (unchanged)
    img_center_x  = (img_cluster.x0 + img_cluster.x1) / 2.0
    text_center_x = (text_cluster.x0 + text_cluster.x1) / 2.0
    img_left_ok   = img_center_x <= pw * LEFT_COLUMN_MAX_FRAC
    text_right_ok = text_center_x >= pw * RIGHT_TEXT_MIN_FRAC
    overlap_ok    = rects_vertical_overlap(img_cluster, text_cluster) if REQUIRE_OVERLAP_Y else True

    ok = (img_left_ok and text_right_ok and overlap_ok)

    info.update({
        "img_cluster": tuple(img_cluster),
        "text_cluster": tuple(text_cluster),
        "img_left_ok": img_left_ok,
        "text_right_ok": text_right_ok,
        "overlap_ok": overlap_ok,
        "kept_images": len(kept_imgs),
        "all_images": len(img_rects_xref),
    })
    return ok, info

# -------------------- Robust Pixmap saving --------------------

def save_pixmap_as_rgb(doc: fitz.Document, xref: int, filepath: str) -> bool:
    try:
        pix = fitz.Pixmap(doc, xref)
    except Exception:
        return False

    if pix.colorspace is None:
        return False

    try:
        if pix.n in (1, 3) and not pix.alpha:
            pass
        else:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if os.path.exists(filepath):
            os.remove(filepath)
        pix.save(filepath)
        return True
    except Exception:
        try:
            pix = fitz.Pixmap(fitz.csGRAY, pix)
            if os.path.exists(filepath):
                os.remove(filepath)
            pix.save(filepath)
            return True
        except Exception:
            return False
    finally:
        try:
            pix = None
        except Exception:
            pass

# ---------------------- OBS fallback rasterization -------------------

def render_left_column_fallback(page: fitz.Page, pdf_name: str, page_idx: int) -> Optional[Dict]:
    """
    When inside OBS but no extractable XObject images survive our filters,
    render a high-res crop of the left column (excluding header/footer bands).
    """
    try:
        rect = page.rect
        top  = rect.y0 + rect.height * HEADER_BAND_FRAC
        bot  = rect.y1 - rect.height * FOOTER_BAND_FRAC
        left = fitz.Rect(rect.x0, top, rect.x0 + rect.width * 0.62, bot)
        pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=left, alpha=False)
        fname = f"{pdf_name}_page{page_idx+1}_fallback.png"
        fpath = os.path.join(IMAGES_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
        pix.save(fpath)
        if DEBUG_IMAGES:
            print(f"      + fallback render -> {fname} @ {tuple(left)}")
        return {
            "report_id": pdf_name,
            "page": page_idx + 1,
            "image_index": 1,  # fallback uses a single crop
            "image_file": fname,
            "bbox": tuple(left),
            "page_width": rect.width,
            "page_height": rect.height,
            "fallback": True,
        }
    except Exception as e:
        if DEBUG_IMAGES:
            print(f"      - fallback render failed: {e}")
        return None

# ----------------------- Section-aware gating -----------------------

def is_summary_page(raw: str) -> bool:
    return any(re.search(p, raw, flags=re.I) for p in SUMMARY_HEADINGS_STRICT)

def is_block_top(page) -> Tuple[bool, str]:
    raw = page_text_raw(page)
    if is_summary_page(raw):
        return True, "summary heading"

    tops_list = top_heading_lines(page, n=14)
    tops_text = "\n".join(tops_list)

    for p in BLOCK_TOP_STRICT:
        if re.search(p, tops_text, flags=re.I):
            return True, "blocked heading (strict)"

    # STRICT: Any heading line containing "overview" = BLOCK,
    # unless that line also contains obs/def/rec terms.
    for ln in tops_list:
        if re.search(r"(?i)\boverview\b", ln):
            if not re.search(r"(?i)observations?|deficienc(?:y|ies)|recommendations?", ln):
                return True, f"overview line: {ln}"

    return False, ""

def is_obs_top(page) -> Tuple[bool, str]:
    tops = top_heading_text(page, n=14)
    for p in OBS_SECTION_HEADINGS_STRICT:
        m = re.search(p, tops, flags=re.I)
        if m:
            return True, m.group(0).strip()
    for p in OBS_SECTION_HEADINGS_UNANCHORED:
        m = re.search(p, tops, flags=re.I)
        if m:
            return True, m.group(0).strip()
    return False, ""

def classify_top_heading(page) -> Tuple[str, str]:
    is_block, why = is_block_top(page)
    if is_block:
        return 'BLOCK', why
    is_obs, why_obs = is_obs_top(page)
    if is_obs:
        return 'OBS', why_obs
    return 'OTHER', ""

# ----------------------- Domain-aware gating ------------------------

def classify_domain_top_heading(page) -> str:
    tops = top_heading_text(page, n=14)
    if contains_any(ROOF_DOMAIN_HEADINGS, tops):
        return 'ROOF'
    if contains_any(BUILDING_DOMAIN_HEADINGS, tops):
        return 'BUILDING'
    return 'OTHER'

# ------------------- Page decision with state -----------------------

def should_process_page(i: int, n_pages: int, appendix_start: Optional[int],
                        toc_range: Optional[Tuple[int,int]], page, raw: str,
                        img_count: int, section_state: str) -> Tuple[bool, List[str], str]:
    reasons = []
    new_state = section_state  # persist by default

    # Structural skips
    if i == 0:
        reasons.append("title");  return False, reasons, new_state
    if i == n_pages - 1:
        reasons.append("last(summary)");  return False, reasons, new_state
    if toc_range is not None and toc_range[0] <= i <= toc_range[1]:
        reasons.append("toc");  return False, reasons, new_state
    if appendix_start is not None and i >= appendix_start:
        reasons.append("appendix");  return False, reasons, new_state

    text_norm = norm_spaces(raw)
    if looks_like_intro(text_norm):
        reasons.append("intro-like")

    # Top-of-page heading classification (uses geometry-aware headings)
    cls, info = classify_top_heading(page)
    if cls == 'BLOCK':
        if section_state == 'OBS' and DEBUG_SECTION:
            print(f"    >> SECTION EXIT: {info}")
        new_state = 'NONE'
        reasons.append("blocked-heading")
        return False, reasons, new_state
    elif cls == 'OBS':
        if section_state != 'OBS' and DEBUG_SECTION:
            print(f"    >> SECTION ENTER (OBS): {info}")
        new_state = 'OBS'

    # Must have any images to consider (if zero, we may still fallback later)
    if img_count == 0:
        reasons += ["no images"]
        return False, reasons, new_state

    # Geometry check always required for normal extraction
    layout_ok, info_geo = page_has_left_image_right_text_layout(page)
    if not layout_ok:
        reasons.append("layout-not-left-img_right-text")
        if "reason" in info_geo:
            reasons.append(info_geo["reason"])
        return False, reasons, new_state

    # Decision by (persistent) section state
    if new_state == 'OBS':
        reasons.append("in-OBS-section (persistent)")
        reasons.append("eligible")
        return True, reasons, new_state

    # Outside OBS: safety valve
    if has_strong_inline_obs_tokens(raw) and has_weak_photo_hints(raw):
        reasons.append("outside-OBS: strong-inline+weak-hints")
        reasons.append("eligible")
        return True, reasons, new_state

    reasons.append("outside-OBS: no strong obs context")
    return False, reasons, new_state

# ----------------------- Extraction core -----------------------

def bbox_in_keep_window(b: fitz.Rect) -> bool:
    return (KEEP_X0_RANGE[0] < b.x0 < KEEP_X0_RANGE[1] and
            KEEP_X1_RANGE[0] < b.x1 < KEEP_X1_RANGE[1] and
            KEEP_Y0_RANGE[0] < b.y0 < KEEP_Y0_RANGE[1] and
            KEEP_Y1_RANGE[0] < b.y1 < KEEP_Y1_RANGE[1])

def extract_images_on_page(doc, page_idx: int, pdf_name: str) -> List[Dict]:
    records = []
    page = doc[page_idx]
    page_rect = page.rect

    img_rects_xref = get_image_rects(page)
    kept = 0
    for bbox, xref in img_rects_xref:
        skip_reasons = []
        if is_header_or_footer(bbox, page_rect):
            skip_reasons.append("header/footer")
        if is_tiny_logo(bbox, page_rect):
            skip_reasons.append("tiny-logo")
        if skip_reasons:
            if DEBUG_IMAGES:
                print(f"      - skip image xref={xref} @ {tuple(bbox)} | {', '.join(skip_reasons)}")
            continue

        # Hard bbox window filter (must fall within keep ranges)
        if not bbox_in_keep_window(bbox):
            if DEBUG_IMAGES:
                print(f"      - skip image xref={xref} @ {tuple(bbox)} | outside keep-window {tuple(bbox)}")
            continue

        filename = f"{pdf_name}_page{page_idx+1}_img{kept+1}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        ok = save_pixmap_as_rgb(doc, xref, filepath)
        if not ok:
            if DEBUG_IMAGES:
                print(f"      - skip image xref={xref} | could not save (unsupported/mask)")
            continue

        kept += 1
        if DEBUG_IMAGES:
            print(f"      + kept image xref={xref} -> {filename} @ {tuple(bbox)}")

        # ---- NEW: grab the right-column text that belongs to THIS image
        right = right_text_for_image(page, bbox)
        fields = derive_label_and_flags(right.get("text", ""))

        rec = {
            "report_id": pdf_name,
            "page": page_idx + 1,
            "image_index": kept,
            "image_file": filename,
            "bbox": tuple(bbox),
            "page_width": page.rect.width,
            "page_height": page.rect.height,

            # label fields:
            "label": fields["label"],
            "label_source": fields["label_source"],          # 'observation', 'observation=other', or 'none'
            "flagged_for_manual": fields["flagged"],         # True if 'Other' or no Observation found
            "discussion_or_description": fields["discussion_or_description"],  # filled when Other or for context
            "right_text_bbox": tuple(right["bbox"]) if right["bbox"] else None,
            "right_text_blocks_used": right.get("blocks_used", 0),
            "right_text_raw": fields["raw_right_text"],
            "photo_id": fields["photo_id"],
        }
        records.append(rec)

    return records

def extract_images_from_pdf(pdf_path: str) -> Tuple[List[Dict], int, Set[int]]:
    """
    Returns:
      - records: list of image metadata dicts
      - n_pages: total pages in this PDF
      - pages_used: set of 1-based page numbers from which images were extracted (incl. fallback)
    """
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    n_pages = len(doc)

    toc_range = find_toc_range(doc)
    appendix_from_toc = appendix_start_from_toc(doc, toc_range) if toc_range else None
    appendix_start = appendix_from_toc if appendix_from_toc is not None else fallback_find_appendix_start(doc)

    if DEBUG_TOC:
        print(f"\n--- Analyzing: {pdf_name} ({n_pages} pages) ---")
        if toc_range:
            print(f"TOC pages: {toc_range[0]+1} to {toc_range[1]+1}")
        else:
            print("TOC pages: not detected")
        if appendix_start is not None:
            print(f"Appendix start page: {appendix_start+1} (from {'TOC' if appendix_from_toc is not None else 'fallback'})")
        else:
            print("Appendix start page: not detected")

    records: List[Dict] = []
    section_state = 'NONE'   # OBS / NONE
    domain_state  = 'OTHER'  # ROOF / BUILDING / OTHER
    pages_used: Set[int] = set()

    for i in range(n_pages):
        page = doc[i]
        raw = page_text_raw(page)

        # --- DOMAIN: update on top heading
        d = classify_domain_top_heading(page)
        if d in ('ROOF', 'BUILDING') and d != domain_state:
            domain_state = d
            if DEBUG_SECTION:
                print(f"    >> DOMAIN ENTER: {domain_state}")

        img_count = len(get_image_rects(page))

        do_process, reasons, section_state = should_process_page(
            i, n_pages, appendix_start, toc_range, page, raw, img_count, section_state
        )

        # Enforce domain gate: only process while inside ROOF
        if do_process and domain_state != 'ROOF':
            if DEBUG_PAGES:
                msg = "SKIP"
                extras = ", ".join(reasons + [f"domain={domain_state}", "blocked-by-domain"])
                print(f"Page {i+1:>3}: {msg}  | {extras}  | state={section_state}")
            continue

        if DEBUG_PAGES:
            msg = "PROCESS" if do_process else "SKIP"
            extras = ", ".join(reasons) if reasons else "eligible"
            extras = extras + f", domain={domain_state}"
            print(f"Page {i+1:>3}: {msg}  | {extras}  | state={section_state}")

        if do_process and img_count > 0:
            recs = extract_images_on_page(doc, i, pdf_name)
            # If in OBS and nothing kept (e.g., tiny/header-only), try fallback render
            if OBS_FALLBACK and len(recs) == 0 and section_state == 'OBS' and domain_state == 'ROOF':
                fb = render_left_column_fallback(page, pdf_name, i)
                if fb:
                    recs = [fb]
            if recs:
                pages_used.add(i + 1)
            records.extend(recs)

        # OBS fallback even if geometry failed / no XObjects
        if OBS_FALLBACK and (not do_process) and section_state == 'OBS' and domain_state == 'ROOF':
            if "layout-not-left-img_right-text" in reasons or "no images" in reasons:
                fb = render_left_column_fallback(page, pdf_name, i)
                if fb:
                    records.append(fb)
                    pages_used.add(i + 1)

    # Conservative overall fallback window (domain-aware) – disabled when OBS_FALLBACK=False
    if OBS_FALLBACK and len(records) == 0:
        if DEBUG_SUMMARY:
            print(f"⚠️  {pdf_name}: No images after gating. Running fallback window...")
        start_i = (toc_range[1] + 1) if toc_range else 2
        end_limit = appendix_start if appendix_start is not None else (n_pages - 1)
        section_state = 'NONE'
        domain_state = 'OTHER'
        for i in range(start_i, max(start_i, end_limit)):
            page = doc[i]
            raw = page_text_raw(page)

            d = classify_domain_top_heading(page)
            if d in ('ROOF', 'BUILDING'):
                domain_state = d
            if domain_state != 'ROOF':
                continue

            cls, _ = classify_top_heading(page)
            if cls == 'BLOCK':
                section_state = 'NONE'
                continue
            elif cls == 'OBS':
                section_state = 'OBS'

            layout_ok, _ = page_has_left_image_right_text_layout(page)
            if layout_ok and section_state == 'OBS':
                recs = extract_images_on_page(doc, i, pdf_name)
                if not recs:
                    fb = render_left_column_fallback(page, pdf_name, i)
                    if fb:
                        records.append(fb)
                        pages_used.add(i + 1)
                else:
                    records.extend(recs)
                    pages_used.add(i + 1)

    doc.close()
    return records, n_pages, pages_used

# --------------------- CSV safe write with retry ---------------------

def safe_write_csv_with_retry(df: pd.DataFrame, path: str) -> str:
    while True:
        try:
            df.to_csv(path, index=False)
            print(f"Metadata CSV      : {path}")
            return path
        except PermissionError:
            print(f"\n⚠️  Permission denied writing to:\n   {path}")
            print("   The file is likely open (e.g., Excel). Please close it.")
            ans = input("When closed, type 'y' to retry, or press Enter to save to a new file: ").strip().lower()
            if ans == 'y':
                continue
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = os.path.join(os.path.dirname(path), f"image_metadata_{ts}.csv")
            try:
                df.to_csv(alt, index=False)
                print(f"→ Wrote to alternate file: {alt}")
                return alt
            except Exception as e:
                print(f"   Still couldn't write (unexpected error: {e}). Retrying original path...")

# ---------------------- Robust report-id sorting ---------------------

def sort_key_report_id(report_id: str) -> Tuple:
    """
    Custom, format-agnostic sorter for IDs like:
      18-012  < 18-013A < 18-013-2 < 18-013B < 21-054-PH1 < 21-054-PH2
      23-022 < 23-023R1 < 23-023R2 < 23-024
    Strategy:
      - Tokenize into alternating [letters] / [digits] across the whole string
      - Compare lexicographically with letters ranked BEFORE numbers at the same position,
        so '...013A' sorts before '...013-2'
      - Uppercase letters for stable ordering; numbers compared as ints
    """
    tokens = re.findall(r'[A-Za-z]+|\d+', report_id)
    key = []
    for t in tokens:
        if t.isdigit():
            key.append((1, int(t)))       # numbers rank after letters at same depth
        else:
            key.append((0, t.upper()))    # letters first (A < B, PH < R, etc.)
    return tuple(key)

# ---------------------- Run history and diffing ----------------------

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def load_previous_run(path: str) -> Optional[dict]:
    """Return dict with 'reports' and 'totals', or None if not present/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_current_run(path: str, payload: dict) -> None:
    """Overwrite the history file with the current payload."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Could not write run history to {path}: {e}")

def diff_and_print(prev: Optional[dict], curr: dict, sort_key_fn) -> None:
    """
    Print a compact diff between prev and curr.
    - Per report: pages_used, images
    - Totals: total_reports, total_images
    """
    print("\n================ RUN DIFF vs LAST =================")
    if prev is None:
        print("No previous run history found. This will be the new baseline.")
        print("===================================================")
        return

    p_reports = prev.get("reports", {})
    c_reports = curr.get("reports", {})

    # union of report ids, sorted using your custom sort
    all_ids = sorted(set(p_reports) | set(c_reports), key=sort_key_fn)

    any_change = False
    for rid in all_ids:
        p = p_reports.get(rid, {})
        c = c_reports.get(rid, {})
        p_pages = _safe_int(p.get("pages_used", 0))
        c_pages = _safe_int(c.get("pages_used", 0))
        p_imgs  = _safe_int(p.get("images", 0))
        c_imgs  = _safe_int(c.get("images", 0))

        # Detect deltas
        if rid not in p_reports:
            any_change = True
            print(f"[NEW] {rid:<20} pages_used: {c_pages:>3}  images: {c_imgs:>4}")
        elif rid not in c_reports:
            any_change = True
            print(f"[REMOVED] {rid:<20} (was pages_used: {p_pages}, images: {p_imgs})")
        else:
            pages_diff = c_pages - p_pages
            imgs_diff  = c_imgs - p_imgs
            if pages_diff != 0 or imgs_diff != 0:
                any_change = True
                pd = f"{p_pages}→{c_pages}" if pages_diff else f"{c_pages}"
                idf = f"{p_imgs}→{c_imgs}" if imgs_diff else f"{c_imgs}"
                sign_pages = "↑" if pages_diff>0 else ("↓" if pages_diff<0 else " ")
                sign_imgs  = "↑" if imgs_diff>0 else ("↓" if imgs_diff<0 else " ")
                print(f"[CHANGED] {rid:<20} pages_used: {pd:<7} {sign_pages}  images: {idf:<9} {sign_imgs}")

    # Totals
    p_tot = prev.get("totals", {})
    c_tot = curr.get("totals", {})
    pr = _safe_int(p_tot.get("total_reports", 0))
    cr = _safe_int(c_tot.get("total_reports", 0))
    pi = _safe_int(p_tot.get("total_images", 0))
    ci = _safe_int(c_tot.get("total_images", 0))
    dr = cr - pr
    di = ci - pi

    print("---------------------------------------------------")
    if dr or di:
        rr = f"{pr}→{cr}" if dr else f"{cr}"
        ii = f"{pi}→{ci}" if di else f"{ci}"
        srr = "↑" if dr>0 else ("↓" if dr<0 else " ")
        sii = "↑" if di>0 else ("↓" if di<0 else " ")
        print(f"TOTAL REPORTS: {rr:<9} {srr}    TOTAL IMAGES: {ii:<11} {sii}")
        any_change = True
    else:
        print(f"TOTAL REPORTS: {cr}         TOTAL IMAGES: {ci}")

    if not any_change:
        print("No differences vs last run.")
    print("===================================================")

# --- Helpers for run history (place above main or near other helpers) ---

def _history_path() -> str:
    # Save alongside your CSV by default
    return os.path.join(os.path.dirname(CSV_OUTPUT), "run_history.txt")

def _load_last_snapshot() -> Dict[str, Dict[str, int]]:
    """
    Returns:
      {
        "__totals__": {"reports": int, "images": int},
        "<report_id>": {"pages_used": int, "images": int},
        ...
      }
    """
    path = _history_path()
    snap: Dict[str, Dict[str, int]] = {}
    if not os.path.exists(path):
        return snap
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # format: key|pages_used|images
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                key, p_used, imgs = parts
                snap[key] = {"pages_used": int(p_used), "images": int(imgs)}
    except Exception:
        pass
    return snap

def _write_snapshot(per_report_stats: Dict[str, Dict[str, int]], total_docs: int, total_images: int) -> None:
    """
    Writes a compact snapshot of the current run to a .txt file (overwrites each run).
    Only stores: report_id, pages_used, images, plus totals under key "__totals__".
    """
    path = _history_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Extraction run snapshot\n")
            f.write(f"# Saved: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"__totals__|{total_docs}|{total_images}\n")
            for rid, stats in per_report_stats.items():
                f.write(f"{rid}|{stats.get('pages_used', 0)}|{stats.get('images', 0)}\n")
    except Exception as e:
        print(f"⚠️  Could not write run history: {e}")

# ------------------------------- Updated main -------------------------------

def main():
    pdf_files = glob.glob(os.path.join(REPORTS_DIR, "*.pdf"))
    all_records: List[Dict] = []
    total_docs = 0

    # Per-report aggregates
    per_report_stats: Dict[str, Dict[str, int]] = {}

    # Load last snapshot (for diff at the end)
    last = _load_last_snapshot()

    for pdf in pdf_files:
        total_docs += 1
        print(f"\nExtracting from {pdf} ...")
        recs, n_pages, pages_used = extract_images_from_pdf(pdf)
        all_records.extend(recs)

        report_id = os.path.splitext(os.path.basename(pdf))[0]
        per_report_stats.setdefault(report_id, {"total_pages": n_pages, "pages_used": 0, "images": 0})
        per_report_stats[report_id]["total_pages"] = n_pages
        per_report_stats[report_id]["pages_used"]  = len(pages_used)
        per_report_stats[report_id]["images"]      = len(recs)

        if DEBUG_SUMMARY:
            print(f"• Extracted {len(recs)} image(s) from {os.path.basename(pdf)}")

    df = pd.DataFrame(all_records)

    if "bbox" in df.columns:
        # Expand bbox into individual columns
        df[["x0", "y0", "x1", "y1"]] = pd.DataFrame(df["bbox"].tolist(), index=df.index)

        # Reorder into desired order: x0, x1, y0, y1
        df = df.drop(columns=["bbox"])
        df = df[["report_id", "page", "image_index", "image_file", 
                "x0", "x1", "y0", "y1", 
                "page_width", "page_height"] + 
                [c for c in df.columns if c not in ["report_id","page","image_index","image_file",
                                                    "x0","x1","y0","y1","page_width","page_height"]]]

        # Add delta columns
        df["dx"] = df["x1"] - df["x0"]
        df["dy"] = df["y1"] - df["y0"]

    # --- Flatten RIGHT-TEXT bbox (if present) ---
    if "right_text_bbox" in df.columns:
        # Expand to rtx0, rty0, rtx1, rty1 (allow None -> NaN)
        rt_expanded = df["right_text_bbox"].apply(
            lambda v: pd.Series(v) if isinstance(v, (list, tuple)) and len(v) == 4
            else pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        )
        rt_expanded.columns = ["rtx0", "rty0", "rtx1", "rty1"]
        df = pd.concat([df.drop(columns=["right_text_bbox"]), rt_expanded], axis=1)

        # Deltas (absolute size in PDF points)
        df["rtdx"] = df["rtx1"] - df["rtx0"]
        df["rtdy"] = df["rty1"] - df["rty0"]

        # (Optional) normalized fractions of page, uncomment if you want them:
        # df["rtdx_frac"] = df["rtdx"] / df["page_width"]
        # df["rtdy_frac"] = df["rtdy"] / df["page_height"]

        # (Optional) move these near your main bbox fields for readability
        base_order = ["report_id", "page", "image_index", "image_file",
                    "x0", "x1", "y0", "y1", "dx", "dy",
                    "rtx0", "rtx1", "rty0", "rty1", "rtdx", "rtdy",
                    "page_width", "page_height"]
        df = df[base_order + [c for c in df.columns if c not in base_order]]

    # After bbox expansion
    for col in ["x0", "x1", "y0", "y1", "dx", "dy",
                "rtx0", "rtx1", "rty0", "rty1", "rtdx", "rtdy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Flatten IMAGE bbox ---
    if "bbox" in df.columns:
        df[["x0", "y0", "x1", "y1"]] = pd.DataFrame(df["bbox"].tolist(), index=df.index)
        df = df.drop(columns=["bbox"])
        df["dx"] = df["x1"] - df["x0"]
        df["dy"] = df["y1"] - df["y0"]

    # --- Flatten RIGHT-TEXT bbox ---
    if "right_text_bbox" in df.columns:
        rt_expanded = df["right_text_bbox"].apply(
            lambda v: pd.Series(v) if isinstance(v, (list, tuple)) and len(v) == 4
            else pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        )
        rt_expanded.columns = ["rtx0", "rty0", "rtx1", "rty1"]
        df = pd.concat([df.drop(columns=["right_text_bbox"]), rt_expanded], axis=1)
        df["rtdx"] = df["rtx1"] - df["rtx0"]
        df["rtdy"] = df["rty1"] - df["rty0"]

    # --- Force numeric + round to 1 decimal place ---
    num_cols = ["x0","x1","y0","y1","dx","dy",
                "rtx0","rtx1","rty0","rty1","rtdx","rtdy",
                "page_width","page_height"]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)

    # --- Clean up raw right text so it's easier to read and Excel-friendly ---
    if "right_text_raw" in df.columns:
        # --- Version 1: preserve sentence breaks ---
        df["right_text_raw"] = (
            df["right_text_raw"]
            .astype(str)
            # Replace newlines that are NOT after ., !, or ? with a space
            .str.replace(r"(?<=[^\.\!\?])\n", " ", regex=True)
            # Keep real sentence-ending newlines
            .str.replace(r"(?<=[\.\!\?])\n", "\n", regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)  # tidy spaces
            .str.strip()
        )

    written_csv = safe_write_csv_with_retry(df, CSV_OUTPUT)

    print("\n================ SUMMARY ================")
    print(f"Processed reports : {total_docs}")
    print(f"Total images      : {len(all_records)}")
    print(f"Images folder     : {IMAGES_DIR}")
    print(f"Metadata CSV      : {written_csv}")
    print("=========================================")

    # --- Per-report table (Total pages, Pages with images, Total images) ---
    if per_report_stats:
        # Build a DataFrame to leverage sorting
        rows = []
        for rid, stats in per_report_stats.items():
            rows.append({
                "report_id": rid,
                "total_pages": stats["total_pages"],
                "pages_with_images": stats["pages_used"],
                "image_count": stats["images"],
            })
        tdf = pd.DataFrame(rows)

        # Custom sort using robust report-id key
        tdf = tdf.sort_values(by="report_id", key=lambda col: col.map(sort_key_report_id))

        # Print table
        print("\n============= PER-REPORT EXTRACTION SUMMARY =============")
        print(f"{'Report ID':<20} {'Pages':>7} {'Pages Used':>12} {'Images':>8}")
        print("-" * 55)
        for _, r in tdf.iterrows():
            print(f"{r['report_id']:<20} {int(r['total_pages']):>7} {int(r['pages_with_images']):>12} | {int(r['image_count']):>8}")
        print("=========================================================")
    else:
        print("\n(no reports processed)")

    # --- Diff vs last run (only pages_used and images per report, plus totals) ---
    print("\n================ CHANGES VS LAST RUN =================")
    changes = 0

    # Totals diff
    last_tot = last.get("__totals__")
    if last_tot:
        last_reports = last_tot.get("reports", 0)
        last_images  = last_tot.get("images", 0)
        if last_reports != total_docs:
            print(f"• Total reports changed: {last_reports} → {total_docs}")
            changes += 1
        if last_images != len(all_records):
            print(f"• Total images changed : {last_images} → {len(all_records)}")
            changes += 1

    # Per-report diffs
    current_simple = {rid: {"pages_used": s["pages_used"], "images": s["images"]}
                      for rid, s in per_report_stats.items()}

    all_keys = set(last.keys()) | set(current_simple.keys())
    all_keys.discard("__totals__")

    if all_keys:
        # Sort keys using your report-id sort
        sorted_keys = sorted(all_keys, key=sort_key_report_id)
        for rid in sorted_keys:
            prev = last.get(rid)
            cur  = current_simple.get(rid)

            if prev and not cur:
                print(f"• {rid}: removed (was pages_used={prev['pages_used']}, images={prev['images']})")
                changes += 1
            elif not prev and cur:
                print(f"• {rid}: added (pages_used={cur['pages_used']}, images={cur['images']})")
                changes += 1
            elif prev and cur:
                deltas = []
                if prev["pages_used"] != cur["pages_used"]:
                    deltas.append(f"pages_used {prev['pages_used']}→{cur['pages_used']}")
                if prev["images"] != cur["images"]:
                    deltas.append(f"images {prev['images']}→{cur['images']}")
                if deltas:
                    print(f"• {rid}: " + ", ".join(deltas))
                    changes += 1

    if changes == 0:
        print("No differences from the last run.")

    print("=======================================================")

    # Save the snapshot for next run
    _write_snapshot(per_report_stats, total_docs, len(all_records))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Stopping script (Ctrl+C pressed).")
