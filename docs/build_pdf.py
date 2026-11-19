"""Generate docs/LOKI_Remake_Guide.pdf from docs/project_guide.html via headless Chrome."""

from __future__ import annotations

import subprocess
from pathlib import Path

CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DOCS_DIR = Path(__file__).resolve().parent
HTML_PATH = DOCS_DIR / "project_guide.html"
PDF_PATH = DOCS_DIR / "LOKI_Remake_Guide.pdf"


def build_pdf(html_path: Path = HTML_PATH, pdf_path: Path = PDF_PATH) -> Path:
    """Render HTML guide into a clean A4 PDF using headless Chrome."""
    if not CHROME_BIN.exists():
        raise FileNotFoundError(f"Google Chrome not found at {CHROME_BIN}")
    if not html_path.exists():
        raise FileNotFoundError(f"HTML guide not found at {html_path}")

    cmd = [
        str(CHROME_BIN),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"PDF was not generated at {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    out = build_pdf()
    print(f"Generated {out} ({out.stat().st_size} bytes)")
