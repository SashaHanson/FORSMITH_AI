# FORSMITH_AI
Report generation AI for FORSMITH Building Science Consultants

## Environment Setup

The dataset creation workflow pulls together PDF parsing, traditional IR, and neural reranking. To keep dependencies isolated, work inside a virtual environment (recommended).

1. **Create the virtual environment**
   ```powershell
   python -m venv .venv
   ```

2. **Activate it (PowerShell)**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   You should see `(.venv)` prefixed to your shell prompt. Leave the environment active while installing packages and running the pipeline.

3. **Install Python dependencies**
   
   Core packages used across the codebase:
   ```powershell
   python -m pip install --upgrade pip
   pip install pandas scikit-learn "pymupdf==1.26.5" rapidfuzz
   ```

   Neural retrieval / reranking (E5 bi-encoder + DeBERTa cross-encoder):
   ```powershell
   pip install torch torchvision torchaudio
   pip install "sentence-transformers>=2.2.2" "transformers>=4.35" accelerate
   ```

   Optional extras:
   ```powershell
   pip install pillow pytesseract   # OCR fallback when PDFs lack text
   ```

   These packages cover:
   - `pymupdf` (`fitz`) for PDF parsing and image extraction.
   - `pandas` for CSV/report generation.
   - `scikit-learn` for TF-IDF cause/effect ranking.
   - `rapidfuzz` for high-performance fuzzy matching (falls back to a pure-Python implementation if missing).
   - `torch`, `sentence-transformers`, `transformers`, `accelerate` for the semantic label retriever/reranker.
   - `pillow`, `pytesseract` (optional) to enable OCR when the PDF has no text layer (requires the Tesseract binary on PATH).

   If you prefer global installation, run the same commands outside the venv (add `--user` when installing without admin rights on Windows).

4. **Verify the install (optional)**
   ```powershell
   python - <<'PY'
import torch, transformers, sentence_transformers, accelerate
import pandas, sklearn, fitz, rapidfuzz
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("accelerate:", accelerate.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
fitz_doc = getattr(fitz, "__doc__", "") or ""
print("PyMuPDF:", fitz_doc.splitlines()[0] if fitz_doc else "unknown")
print("rapidfuzz:", rapidfuzz.__version__)
PY
   ```

5. **Run the dataset pipeline**
   ```powershell
   python Dataset_Creation\main.py
   ```

When you finish working, exit the virtual environment with `deactivate`. Delete the `.venv` folder if you ever need to rebuild it from scratch.
