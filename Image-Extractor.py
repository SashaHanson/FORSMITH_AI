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
# ------------------------------------------------------------

import fitz  # PyMuPDF
import os, glob, re
import pandas as pd
import datetime
from typing import List, Optional, Dict, Tuple

# === CONFIG: YOUR FOLDERS ===
REPORTS_DIR = r"D:\FORSMITH - AI\Dataset\Reports"
IMAGES_DIR  = r"D:\FORSMITH - AI\Dataset\Images"
CSV_OUTPUT  = r"D:\FORSMITH - AI\Dataset\image_metadata.csv"

# === LOGGING SWITCHES ===
DEBUG_TOC     = True
DEBUG_PAGES   = True
DEBUG_SUMMARY = True
DEBUG_IMAGES  = True
DEBUG_SECTION = True   # prints ENTER/EXIT for sections and domain

# === HEADER / FOOTER + SIZE FILTERS ===
HEADER_BAND_FRAC = 0.10
FOOTER_BAND_FRAC = 0.10
MIN_IMAGE_AREA_FRAC = 0.01   # relaxed from 0.02 to catch smaller embedded photos
LEFT_COLUMN_MAX_FRAC = 0.60  # relaxed from 0.55
RIGHT_TEXT_MIN_FRAC  = 0.40  # relaxed from 0.45
REQUIRE_OVERLAP_Y    = True

# === HEADING GEOMETRY FILTERS ===
HEADING_TOP_BAND_FRAC   = 0.20  # consider headings only in top 20% of page
HEADING_LEFT_MAX_FRAC   = 0.45  # consider left/center blocks only (x0 <= 45% pw)
HEADING_WIDE_MIN_FRAC   = 0.60  # or very wide blocks (>=60% pw)

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

    img_rects_xref = get_image_rects(page)
    text_rects = [r for r, _ in get_text_blocks(page)]

    if not img_rects_xref or not text_rects:
        info["reason"] = "no image or no text blocks"
        return False, info

    filtered_imgs = []
    for bbox, xref in img_rects_xref:
        if is_header_or_footer(bbox, page_rect):
            continue
        if is_tiny_logo(bbox, page_rect):
            continue
        filtered_imgs.append((bbox, xref))

    if not filtered_imgs:
        info["reason"] = "all images were header/footer/tiny"
        return False, info

    img_rects = [bb for bb, _ in filtered_imgs]
    img_cluster = cluster_rects(img_rects)
    text_cluster = cluster_rects(text_rects)
    if not img_cluster or not text_cluster:
        info["reason"] = "no clusters"
        return False, info

    # Primary left-right geometry
    img_center_x = (img_cluster.x0 + img_cluster.x1) / 2.0
    text_center_x = (text_cluster.x0 + text_cluster.x1) / 2.0
    img_left_ok  = img_center_x <= pw * LEFT_COLUMN_MAX_FRAC
    text_right_ok = text_center_x >= pw * RIGHT_TEXT_MIN_FRAC
    overlap_ok   = rects_vertical_overlap(img_cluster, text_cluster) if REQUIRE_OVERLAP_Y else True

    # Secondary heuristic: stacked small images on left half
    left_half = page_rect.x0 + 0.55 * pw
    stacked_left = all(bb.x1 <= left_half for bb in img_rects) and len(img_rects) >= 2

    ok = (img_left_ok and text_right_ok and overlap_ok) or stacked_left

    info.update({
        "img_cluster": tuple(img_cluster),
        "text_cluster": tuple(text_cluster),
        "img_left_ok": img_left_ok,
        "text_right_ok": text_right_ok,
        "overlap_ok": overlap_ok,
        "stacked_left": stacked_left,
        "kept_images": len(filtered_imgs),
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
        # widen a bit to ensure we cover photos that are slightly more centered
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

    # Get only geometry-aware top lines (left column / wide blocks near top)
    tops_list = top_heading_lines(page, n=14)
    tops_text = "\n".join(tops_list)

    for p in BLOCK_TOP_STRICT:
        if re.search(p, tops_text, flags=re.I):
            return True, "blocked heading (strict)"

    # Catch '... Overview', but ignore photo captions or canopy/roof/photo refs
    for ln in tops_list:
        if re.search(r"(?i)\boverview\b", ln):
            # skip if likely caption / not a true heading
            if re.search(r"(?i)\b(canopy|roof|parapet|photograph|photo)\b", ln):
                continue
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
        # don't return yet; let OBS fallback decide downstream
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

        records.append({
            "report_id": pdf_name,
            "page": page_idx + 1,
            "image_index": kept,
            "image_file": filename,
            "bbox": tuple(bbox),
            "page_width": page.rect.width,
            "page_height": page.rect.height,
        })

    return records

def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
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
            if len(recs) == 0 and section_state == 'OBS' and domain_state == 'ROOF':
                fb = render_left_column_fallback(page, pdf_name, i)
                if fb:
                    recs = [fb]
            records.extend(recs)

        # If geometry prevented PROCESS but we are in OBS and domain=ROOF and there are *no* usable XObjects,
        # attempt a fallback render anyway (covers vector-only photos / unusual embedding).
        if (not do_process) and section_state == 'OBS' and domain_state == 'ROOF':
            # Only when the reason indicates no usable images or layout mismatch:
            if "layout-not-left-img_right-text" in reasons or "no images" in reasons:
                fb = render_left_column_fallback(page, pdf_name, i)
                if fb:
                    records.append(fb)

    # Conservative overall fallback window (domain-aware)
    if len(records) == 0:
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
                else:
                    records.extend(recs)

    doc.close()
    return records

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

# --------------------------- Main ------------------------------

def main():
    pdf_files = glob.glob(os.path.join(REPORTS_DIR, "*.pdf"))
    all_records: List[Dict] = []
    total_docs = 0

    for pdf in pdf_files:
        total_docs += 1
        print(f"\nExtracting from {pdf} ...")
        recs = extract_images_from_pdf(pdf)
        all_records.extend(recs)

        if DEBUG_SUMMARY:
            print(f"• Extracted {len(recs)} image(s) from {os.path.basename(pdf)}")

    df = pd.DataFrame(all_records)

    written_csv = safe_write_csv_with_retry(df, CSV_OUTPUT)

    print("\n================ SUMMARY ================")
    print(f"Processed reports : {total_docs}")
    print(f"Total images      : {len(all_records)}")
    print(f"Images folder     : {IMAGES_DIR}")
    print(f"Metadata CSV      : {written_csv}")
    print("=========================================")

    # --- Summary table by report ---
    if not df.empty and "report_id" in df.columns:
        counts = df.groupby("report_id").size().reset_index(name="image_count")

        # Robust sort for aa-bbb-ccc, aa-bbb-ccc-d, aa-bbb-cc-d, alphanumeric suffixes
        def sort_key(report_id: str):
            parts = report_id.split("-")
            key = []
            for part in parts:
                m = re.match(r"(\d+)([A-Za-z].*)?", part)
                if m:
                    num = int(m.group(1))
                    suffix = m.group(2) or ""
                    key.append(num)
                    key.append(suffix)
                else:
                    key.append(part)
            return tuple(key)

        counts = counts.sort_values(by="report_id", key=lambda col: col.map(sort_key))

        print("\n============= IMAGE COUNT PER REPORT =============")
        print(f"{'Report ID':<20} {'Images':>8}")
        print("-" * 34)
        for _, row in counts.iterrows():
            print(f"{row['report_id']:<20} {row['image_count']:>8}")
        print("==================================================")
    else:
        print("\n(no extractions to summarize by report)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Stopping script (Ctrl+C pressed).")
