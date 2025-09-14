# ------------------------------------------------------------
# Image & text-region extraction for FORSMITH roof reports.
# - Handles domain/section gating, geometry checks
# - Extracts kept images and captures right-column text
# - Delegates label logic to a LabelMatcher provided by caller
# ------------------------------------------------------------

from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict
import fitz  # PyMuPDF
import os, re

# ---------- HEADER / FOOTER + SIZE FILTERS (unchanged) ----------
HEADER_BAND_FRAC = 0.10
FOOTER_BAND_FRAC = 0.10
MIN_IMAGE_AREA_FRAC = 0.03
LEFT_COLUMN_MAX_FRAC = 0.60
RIGHT_TEXT_MIN_FRAC  = 0.38
MIN_COMBINED_IMAGE_AREA_FRAC = 0.04
REQUIRE_OVERLAP_Y = True

# === HEADING GEOMETRY FILTERS ===
HEADING_TOP_BAND_FRAC   = 0.30
HEADING_LEFT_MAX_FRAC   = 0.45
HEADING_WIDE_MIN_FRAC   = 0.60

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

BLOCK_TOP_STRICT = [
    r"(?mi)^\s*condition\s+assessment\b",
    r"(?mi)^\s*roof\s+condition\s+assessment\b",
    r"(?mi)^\s*existing\s+conditions\b",
    r"(?mi)^\s*roof\s+system\s+description\b",
    r"(?mi)^\s*roof\s+composition\b",
]

OBS_SECTION_HEADINGS_STRICT = [
    r"(?mi)^\s*(?:\d+(?:\.\d+)*)?\s*observations?\s*(?:&|and)\s*recommendations?\b",
    r"(?mi)^\s*deficienc(?:y|ies)\s*(?:&|and)?\s*recommendations?\b",
    r"(?mi)^\s*observations?\b",
    r"(?mi)^\s*deficienc(?:y|ies)\b",
    r"(?mi)^\s*recommendations?\b",
]
OBS_SECTION_HEADINGS_UNANCHORED = [
    r"(?mi)\bobservations?\s*(?:&|and)\s*recommendations?\b",
    r"(?mi)\bdeficienc(?:y|ies)\s*(?:&|and)?\s*recommendations?\b",
]

WEAK_PHOTO_HINTS = [
    r"\bphotograph[s]?\b",
    r"\bphoto[s]?\b",
    r"\bfigure[s]?\b",
    r"(?mi)^\s*item\s+\d+\b",
    r"(?mi)^\s*section\s+\d+\b",
    r"\broof\s*(area|section)\s+\d+\b",
]

STRONG_INLINE_TOKENS = [
    r"\bobservation[s]?\b",
    r"\bdeficienc(?:y|ies)\b",
    r"\brecommendation[s]?\b",
    r"\bdeterioration\b",
    r"\bleak\b",
]

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

TOC_LINE_PATTERN = r".{3,}(\.{2,}|\s{2,})\s+(\d{1,4})\s*$"
TOC_APPENDIX_PATTERN = r"\bappendix\b"
APPENDIX_HEADING_START = r"(?mi)^\s*appendix(?:\s+[A-Z0-9]+)?\b"

SEC_REGEX  = re.compile(r"^\s*([1-7])\.(\d+)\s+(.+)$")   # e.g., 3.0 Roof Penetrations

# ---------- Debug toggles (kept for parity; caller can set) ----------
DEBUG_TOC     = True
DEBUG_PAGES   = True
DEBUG_SUMMARY = True
DEBUG_IMAGES  = True
DEBUG_SECTION = True

# ------------------------ Small helpers ------------------------
def norm_spaces(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s or "").strip().lower()

def contains_any(patterns, text: str) -> bool:
    import re
    return any(re.search(p, text or "", flags=re.I) for p in patterns)

def page_text_raw(page) -> str:
    return page.get_text("text") or ""

def get_text_blocks(page):
    blocks = []
    try:
        for b in page.get_text("blocks") or []:
            if len(b) >= 7 and b[6] == 0 and isinstance(b[4], str):
                import fitz
                blocks.append((fitz.Rect(b[0], b[1], b[2], b[3]), b[4]))
            elif len(b) >= 5 and isinstance(b[4], str):
                import fitz
                blocks.append((fitz.Rect(b[0], b[1], b[2], b[3]), b[4]))
    except Exception:
        pass
    return blocks

# ---------------------- Image helpers ----------------------
def get_image_rects(page):
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

# ---------------- Layout check: left image / right text -------------
def page_has_left_image_right_text_layout(page) -> Tuple[bool, Dict]:
    info = {}
    page_rect = page.rect
    pw = page_rect.width
    page_area = page_rect.get_area()

    img_rects_xref = get_image_rects(page)
    text_rects = [r for r, _ in get_text_blocks(page)]
    if not img_rects_xref or not text_rects:
        info["reason"] = "no image or no text blocks"
        return False, info

    kept_imgs = []
    for bbox, xref in img_rects_xref:
        if is_header_or_footer(bbox, page_rect):
            continue
        kept_imgs.append((bbox, xref))
    if not kept_imgs:
        info["reason"] = "all images were header/footer"
        return False, info

    img_rects = [bb for bb, _ in kept_imgs]
    img_cluster = cluster_rects(img_rects)
    text_cluster = cluster_rects(text_rects)
    if not img_cluster or not text_cluster:
        info["reason"] = "no clusters"
        return False, info

    if (img_cluster.get_area() / max(1.0, page_area)) < MIN_COMBINED_IMAGE_AREA_FRAC:
        info["reason"] = "combined-image-area tiny"
        return False, info

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

# ---------------- Right-column capture (anchored to image) ----------------
def collect_right_column_blocks(page) -> List[Tuple[fitz.Rect, str]]:
    blocks = []
    pw, ph = page.rect.width, page.rect.height
    x_cut = page.rect.x0 + pw * RIGHT_TEXT_MIN_FRAC
    y_top = page.rect.y0 + ph * HEADER_BAND_FRAC
    y_bot = page.rect.y1 - ph * FOOTER_BAND_FRAC
    for rect, text in get_text_blocks(page):
        if rect.x0 >= x_cut and rect.y1 > y_top and rect.y0 < y_bot and (text or "").strip():
            blocks.append((rect, text))
    blocks.sort(key=lambda rt: (round(rt[0].y0, 2), round(rt[0].x0, 2)))
    return blocks

def merge_rect(a: fitz.Rect, b: fitz.Rect) -> fitz.Rect:
    r = fitz.Rect(a); r |= b; return r

def right_text_for_image(page, img_bbox: fitz.Rect, y_pad: float = 6.0, x_pad: float = 6.0) -> Dict:
    blocks = collect_right_column_blocks(page)
    if not blocks:
        return {"text": "", "bbox": None, "blocks_used": 0}

    used = []
    img_y0 = img_bbox.y0 - y_pad
    img_y1 = img_bbox.y1 + y_pad
    for rect, _ in blocks:
        top = max(rect.y0, img_y0)
        bot = min(rect.y1, img_y1)
        if (bot - top) > 0:
            used.append((rect, ""))

    if not used:
        cy = (img_bbox.y0 + img_bbox.y1) / 2.0
        rect, _ = min(blocks, key=lambda rt: abs(((rt[0].y0 + rt[0].y1)/2.0) - cy))
        r = fitz.Rect(rect.x0 - x_pad, rect.y0 - y_pad, rect.x1 + x_pad, rect.y1 + y_pad)
        full = page.get_text("text", clip=r) or ""
        return {"text": full.strip(), "bbox": r, "blocks_used": 1}

    merged = used[0][0]
    for rect, _ in used[1:]:
        merged = merge_rect(merged, rect)
    r = fitz.Rect(merged.x0 - x_pad, merged.y0 - y_pad, merged.x1 + x_pad, merged.y1 + y_pad)
    full = page.get_text("text", clip=r) or ""
    return {"text": full.strip(), "bbox": r, "blocks_used": len(used)}

# -------------------- TOC / Appendix detection --------------------
def is_toc_like_page(raw: str) -> bool:
    if not raw:
        return False
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return False
    if contains_any(TOC_HEADER_PATTERNS, raw):
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

# ---------- GEOMETRY-AWARE heading capture (ignores right column) ----
def top_heading_lines(page, n: int = 14):
    pw = page.rect.width
    ph = page.rect.height
    top_cut = page.rect.y0 + ph * HEADING_TOP_BAND_FRAC
    left_max_x0 = page.rect.x0 + pw * HEADING_LEFT_MAX_FRAC
    wide_min_w  = pw * HEADING_WIDE_MIN_FRAC

    lines = []
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
    for ln in tops_list:
        if re.search(r"(?i)\boverview\b", ln) and not re.search(r"(?i)observations?|deficienc(?:y|ies)|recommendations?", ln):
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

def classify_domain_top_heading(page) -> str:
    tops = top_heading_text(page, n=14)
    if contains_any(ROOF_DOMAIN_HEADINGS, tops):
        return 'ROOF'
    if contains_any(BUILDING_DOMAIN_HEADINGS, tops):
        return 'BUILDING'
    return 'OTHER'

# -------------------- Page gating with state --------------------
def should_process_page(i: int, n_pages: int, appendix_start: Optional[int],
                        toc_range: Optional[Tuple[int,int]], page, raw: str,
                        img_count: int, section_state: str) -> Tuple[bool, List[str], str]:
    reasons = []
    new_state = section_state

    if i == 0:
        reasons.append("title");  return False, reasons, new_state
    if i == n_pages - 1:
        reasons.append("last(summary)");  return False, reasons, new_state
    if toc_range is not None and toc_range[0] <= i <= toc_range[1]:
        reasons.append("toc");  return False, reasons, new_state
    if appendix_start is not None and i >= appendix_start:
        reasons.append("appendix");  return False, reasons, new_state

    if any(re.search(p, norm_spaces(raw or ""), flags=re.I) for p in INTRO_PATTERNS):
        reasons.append("intro-like")

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

    if img_count == 0:
        reasons += ["no images"]
        return False, reasons, new_state

    layout_ok, info_geo = page_has_left_image_right_text_layout(page)
    if not layout_ok:
        reasons.append("layout-not-left-img_right-text")
        if "reason" in info_geo:
            reasons.append(info_geo["reason"])
        return False, reasons, new_state

    if new_state == 'OBS':
        reasons.append("in-OBS-section (persistent)")
        reasons.append("eligible")
        return True, reasons, new_state

    if contains_any(STRONG_INLINE_TOKENS, raw) and contains_any(WEAK_PHOTO_HINTS, raw):
        reasons.append("outside-OBS: strong-inline+weak-hints")
        reasons.append("eligible")
        return True, reasons, new_state

    reasons.append("outside-OBS: no strong obs context")
    return False, reasons, new_state

# ----------------------- Keep-window & helpers -----------------------
def bbox_in_keep_window(b: fitz.Rect) -> bool:
    return (KEEP_X0_RANGE[0] < b.x0 < KEEP_X0_RANGE[1] and
            KEEP_X1_RANGE[0] < b.x1 < KEEP_X1_RANGE[1] and
            KEEP_Y0_RANGE[0] < b.y0 < KEEP_Y0_RANGE[1] and
            KEEP_Y1_RANGE[0] < b.y1 < KEEP_Y1_RANGE[1])

# ----------------------- Core per-page extraction -----------------------
def extract_images_on_page(doc, page_idx: int, pdf_name: str, images_dir: str, matcher) -> List[Dict]:
    """
    Extract kept images and capture right text; delegate label computation to `matcher`.
    Returns a list of record dicts (same schema as before).
    """
    import os
    records = []
    page = doc[page_idx]
    page_rect = page.rect
    page_w, page_h = page_rect.width, page_rect.height

    img_rects_xref = get_image_rects(page)
    kept = 0

    for bbox, xref in img_rects_xref:
        skip_reasons = []
        if is_header_or_footer(bbox, page_rect):
            skip_reasons.append("header/footer")
        if is_tiny_logo(bbox, page_rect):
            skip_reasons.append("tiny-logo")
        if not bbox_in_keep_window(bbox):
            skip_reasons.append("outside keep-window")
        if skip_reasons:
            if DEBUG_IMAGES:
                print(f"      - skip image xref={xref} @ {tuple(bbox)} | {', '.join(skip_reasons)}")
            continue

        filename = f"{pdf_name}_page{page_idx+1}_img{kept+1}.png"
        filepath = os.path.join(images_dir, filename)
        ok = save_pixmap_as_rgb(doc, xref, filepath)
        if not ok:
            if DEBUG_IMAGES:
                print(f"      - skip image xref={xref} | could not save (unsupported/mask)")
            continue

        kept += 1
        if DEBUG_IMAGES:
            print(f"      + kept image xref={xref} -> {filename} @ {tuple(bbox)}")

        right = right_text_for_image(page, bbox)
        right_text = right.get("text", "") or ""
        right_bbox = right.get("bbox")
        right_blocks_used = right.get("blocks_used", 0)

        # current 1.0–7.0 section from page text (for priors)
        raw_page = page_text_raw(page)
        current_section = None
        for ln in raw_page.splitlines():
            m = SEC_REGEX.match(ln.strip())
            if m:
                current_section = f"{m.group(1)}.0"

        # Delegate full labeling (observation/discussion parsing, taxonomy match, linkage)
        label_info = matcher.compute_label(right_text, raw_page, current_section, bbox, right_bbox, page_h)

        rec = {
            "report_id": pdf_name,
            "page": page_idx + 1,
            "image_index": kept,
            "image_file": filename,

            "bbox": tuple(bbox),
            "page_width": page_w,
            "page_height": page_h,

            "right_text_bbox": tuple(right_bbox) if right_bbox else None,
            "right_text_blocks_used": right_blocks_used,
            "right_text_raw": label_info["raw_right_text"],

            "observation_raw": label_info["observation_raw"],
            "discussion_or_description": label_info["discussion_or_description"],
            "label_source": label_info["label_source"],
            "photo_id": label_info["photo_id"],

            "observation_label": label_info["observation_label"],
            "observation_category": label_info["observation_category"],
            "confidence_label": label_info["confidence_label"],
            "confidence_linkage": label_info["confidence_linkage"],

            "flag_review": label_info["flag_review"],
            "flag_reason": label_info["flag_reason"],
            "section": current_section,
        }
        records.append(rec)
    return records

# ----------------------- Whole-PDF extraction -----------------------
def extract_images_from_pdf(pdf_path: str, images_dir: str, matcher) -> Tuple[List[Dict], int, Set[int]]:
    """
    Drives page gating + per-page extraction. Returns:
      - records (list[dict])
      - n_pages (int)
      - pages_used (set[int], 1-based)
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
    section_state = 'NONE'
    domain_state  = 'OTHER'
    pages_used: Set[int] = set()

    for i in range(n_pages):
        page = doc[i]
        raw  = page_text_raw(page)

        # domain update
        d = classify_domain_top_heading(page)
        if d in ('ROOF', 'BUILDING') and d != domain_state:
            domain_state = d
            if DEBUG_SECTION:
                print(f"    >> DOMAIN ENTER: {domain_state}")

        img_count = len(get_image_rects(page))
        do_process, reasons, section_state = should_process_page(
            i, n_pages, appendix_start, toc_range, page, raw, img_count, section_state
        )

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
            recs = extract_images_on_page(doc, i, pdf_name, images_dir, matcher)
            if recs:
                pages_used.add(i + 1)
                records.extend(recs)

    doc.close()
    return records, n_pages, pages_used