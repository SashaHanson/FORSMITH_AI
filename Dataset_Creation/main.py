# ------------------------------------------------------------
# Orchestrates extraction & labeling:
# - Loads taxonomy
# - Iterates PDFs, calls image_extraction with LabelMatcher
# - Builds DataFrame and writes CSV
# - Saves run snapshot & prints diff vs previous run
# - Prints end-of-run SUMMARY and per-report table
# ------------------------------------------------------------

import os, glob, re, json, datetime
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import pandas as pd

from label_matching import load_labels_from_json, LabelMatcher
from image_extraction import (
    extract_images_from_pdf,
)

# === CONFIG: YOUR FOLDERS (unchanged paths; adjust as needed) ===
REPORTS_DIR = r"D:\FORSMITH - AI\Dataset\Reports"
IMAGES_DIR  = r"D:\FORSMITH - AI\Dataset\Images"
CSV_OUTPUT  = r"D:\FORSMITH - AI\Dataset\image_metadata.csv"
RUN_HISTORY_TXT = r"D:\FORSMITH - AI\Dataset\run_summary.txt"
LABELS_JSON = r"D:\FORSMITH - AI\Code\Label_Extraction\forsmith_roof_labels.json"

os.makedirs(IMAGES_DIR, exist_ok=True)

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

# ---------------------- Robust report-id sorting ----------------------
def sort_key_report_id(report_id: str) -> Tuple:
    tokens = re.findall(r'[A-Za-z]+|\d+', report_id)
    key = []
    for t in tokens:
        if t.isdigit():
            key.append((1, int(t)))
        else:
            key.append((0, t.upper()))
    return tuple(key)

# ---------------------- Run history (compact snapshot) ----------------------
def _history_path() -> str:
    return os.path.join(os.path.dirname(CSV_OUTPUT), "run_history.txt")

def _load_last_snapshot() -> Dict[str, Dict[str, int]]:
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
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                key, p_used, imgs = parts
                snap[key] = {"pages_used": int(p_used), "images": int(imgs)}
    except Exception:
        pass
    return snap

def _write_snapshot(per_report_stats: Dict[str, Dict[str, int]], total_docs: int, total_images: int) -> None:
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

def diff_and_print(prev: Optional[dict], curr: dict, sort_key_fn) -> None:
    print("\n================ RUN DIFF vs LAST =================")
    if prev is None:
        print("No previous run history found. This will be the new baseline.")
        print("===================================================")
        return

    p_reports = prev.get("reports", {})
    c_reports = curr.get("reports", {})
    all_ids = sorted(set(p_reports) | set(c_reports), key=sort_key_fn)

    any_change = False
    def _safe_int(x):
        try: return int(x)
        except: return 0

    for rid in all_ids:
        p = p_reports.get(rid, {})
        c = c_reports.get(rid, {})
        p_pages = _safe_int(p.get("pages_used", 0))
        c_pages = _safe_int(c.get("pages_used", 0))
        p_imgs  = _safe_int(p.get("images", 0))
        c_imgs  = _safe_int(c.get("images", 0))
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

# ------------------------------- Main -------------------------------
def main():
    pdf_files = glob.glob(os.path.join(REPORTS_DIR, "*.pdf"))
    all_records: List[Dict] = []
    total_docs = 0
    per_report_stats: Dict[str, Dict[str, int]] = {}

    # ---- Load taxonomy and init matcher ----
    taxonomy_by_canonical, labels_by_category, all_labels_canonical = load_labels_from_json(LABELS_JSON)
    print(f"Loaded {len(all_labels_canonical)} labels from JSON: {LABELS_JSON}")

    matcher = LabelMatcher(
        taxonomy_by_canonical=taxonomy_by_canonical,
        labels_by_category=labels_by_category,
        all_labels_canonical=all_labels_canonical,
        cfg={
            "fuzzy_strict": 0.90,
            "fuzzy_loose" : 0.80,
            "header_ymax": 90,
            "footer_ymin_from_bottom": 80,
            "min_panel_width": 80,
        }
    )

    # Load prior compact snapshot (optional pretty diff at end)
    prior_snapshot_txt = _history_path()
    last_snapshot_compact = {}
    if os.path.exists(prior_snapshot_txt):
        last_lines = open(prior_snapshot_txt, "r", encoding="utf-8").read().splitlines()
        prev_reports = {}
        totals = {}
        for ln in last_lines:
            if not ln or ln.startswith("#"): continue
            key, a, b = ln.split("|")
            if key == "__totals__":
                totals = {"total_reports": int(a), "total_images": int(b)}
            else:
                prev_reports[key] = {"pages_used": int(a), "images": int(b)}
        last_snapshot_compact = {"reports": prev_reports, "totals": totals}

    for pdf in pdf_files:
        total_docs += 1
        print(f"\nExtracting from {pdf} ...")
        recs, n_pages, pages_used = extract_images_from_pdf(pdf, IMAGES_DIR, matcher)
        all_records.extend(recs)

        report_id = os.path.splitext(os.path.basename(pdf))[0]
        per_report_stats.setdefault(report_id, {"total_pages": n_pages, "pages_used": 0, "images": 0})
        per_report_stats[report_id]["total_pages"] = n_pages
        per_report_stats[report_id]["pages_used"]  = len(pages_used)
        per_report_stats[report_id]["images"]      = len(recs)

        print(f"• Extracted {len(recs)} image(s) from {os.path.basename(pdf)}")

    df = pd.DataFrame(all_records)

    if not df.empty and "bbox" in df.columns:
        df[["x0", "y0", "x1", "y1"]] = pd.DataFrame(df["bbox"].tolist(), index=df.index)
        df = df.drop(columns=["bbox"])
        df = df[["report_id", "page", "image_index", "image_file",
                 "x0", "x1", "y0", "y1",
                 "page_width", "page_height"] +
                 [c for c in df.columns if c not in ["report_id","page","image_index","image_file",
                                                     "x0","x1","y0","y1","page_width","page_height"]]]
        df["dx"] = df["x1"] - df["x0"]
        df["dy"] = df["y1"] - df["y0"]

    if not df.empty and "right_text_bbox" in df.columns:
        rt_expanded = df["right_text_bbox"].apply(
            lambda v: pd.Series(v) if isinstance(v, (list, tuple)) and len(v) == 4
            else pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        )
        rt_expanded.columns = ["rtx0", "rty0", "rtx1", "rty1"]
        df = pd.concat([df.drop(columns=["right_text_bbox"]), rt_expanded], axis=1)
        df["rtdx"] = df["rtx1"] - df["rtx0"]
        df["rtdy"] = df["rty1"] - df["rty0"]

    csv_path = safe_write_csv_with_retry(df, CSV_OUTPUT) # Write CSV

    # ======== YOUR REQUESTED SUMMARY OUTPUT ========
    print("\n================ SUMMARY ================")
    print(f"Processed reports : {total_docs}")
    print(f"Total images      : {len(all_records)}")
    print(f"Images folder     : {IMAGES_DIR}")
    print(f"Metadata CSV      : {csv_path}")
    print("=========================================")

    # --- Per-report table (Total pages, Pages with images, Total images) ---
    if per_report_stats:
        rows = []
        for rid, stats in per_report_stats.items():
            rows.append({
                "report_id": rid,
                "total_pages": stats["total_pages"],
                "pages_with_images": stats["pages_used"],
                "image_count": stats["images"],
            })
        tdf = pd.DataFrame(rows)
        tdf = tdf.sort_values(by="report_id", key=lambda col: col.map(sort_key_report_id))

        print("\n============= PER-REPORT EXTRACTION SUMMARY =============")
        print(f"{'Report ID':<20} {'Pages':>7} {'Pages Used':>12} {'Images':>8}")
        print("-" * 55)
        for _, r in tdf.iterrows():
            print(f"{r['report_id']:<20} {int(r['total_pages']):>7} {int(r['pages_with_images']):>12} | {int(r['image_count']):>8}")
        print("=========================================================")
    else:
        print("\n(no reports processed)")
    # ======== END SUMMARY OUTPUT ========

    # Build a compact JSON summary (for human diff)
    curr_summary = {
        "reports": {rid: {"pages_used": v.get("pages_used", 0), "images": v.get("images", 0)}
                    for rid, v in per_report_stats.items()},
        "totals": {
            "total_reports": len(per_report_stats),
            "total_images": int(df.shape[0]) if not df.empty else 0
        }
    }

    # Pretty diff vs last text snapshot (if we have prev)
    diff_and_print(last_snapshot_compact if last_snapshot_compact else None, curr_summary, sort_key_report_id)

    # Write a fresh compact text snapshot for next run
    _write_snapshot(
        per_report_stats={rid: {"pages_used": v.get("pages_used", 0), "images": v.get("images", 0)}
                          for rid, v in per_report_stats.items()},
        total_docs=len(per_report_stats),
        total_images=int(df.shape[0]) if not df.empty else 0
    )
    print("\nDone.")

if __name__ == "__main__":
    main()
