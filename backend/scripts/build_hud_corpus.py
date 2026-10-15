"""Build the guideline corpus (JSONL, one record per handbook page) from HUD Handbook 4000.1.

Source (U.S. government work, public domain):
https://www.hud.gov/sites/dfiles/OCHCO/documents/4000.1hsgh.pdf
Usage: python scripts/build_hud_corpus.py <path-to-4000.1.pdf> [first_pdf_page] [last_pdf_page]
Default range = Section II.A.4 "Underwriting the Borrower Using TOTAL Mortgage Scorecard".
"""
import json
import re
import sys
from pathlib import Path

from PyPDF2 import PdfReader

OUT = Path(__file__).resolve().parents[1] / "data" / "hud_4000_1_borrower_underwriting.jsonl"
NOISE = re.compile(
    r"^(II\. ORIGINATION|A\. Title II|4\. Underwriting the Borrower|Handbook 4000\.1\s+\d+|"
    r"Effective Date:|\*Refer to the online|FHA Single Family Housing Policy)", re.I)
HEADING = re.compile(r"^\s*(?:[a-z]|[ivx]+)\.\s+([A-Z][^.,:]{3,80})$")
PAGE_NO = re.compile(r"Handbook 4000\.1\s+(\d+)")


def build(pdf: str, first: int, last: int) -> int:
    reader = PdfReader(pdf)
    section, n = "II.A.4 Underwriting the Borrower Using TOTAL", 0
    with OUT.open("w", encoding="utf-8") as f:
        for i in range(first - 1, last):
            raw = reader.pages[i].extract_text() or ""
            m = PAGE_NO.search(raw)
            page = int(m.group(1)) if m else None
            lines = []
            for line in raw.splitlines():
                s = line.strip()
                if not s or NOISE.match(s):
                    continue
                h = HEADING.match(line)
                if h:
                    section = re.sub(r"\s+", " ", h.group(1)).strip()
                lines.append(s)
            text = re.sub(r"[ \t]+", " ", "\n".join(lines))
            if page and len(text) > 200:
                f.write(json.dumps({"page": page, "section": section, "text": text}) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    a = sys.argv
    print("pages written:", build(a[1], int(a[2]) if len(a) > 2 else 199, int(a[3]) if len(a) > 3 else 290))
