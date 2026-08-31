import re

import pymupdf

SECTION_HINTS = ("abstract", "introduction", "methods", "methodology", "results", "discussion", "conclusion")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    return _CONTROL_CHARS.sub("", text)


def parse_pdf(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        text = clean_text(page.get_text("text")).strip()
        if text:
            pages.append({"page": page.number + 1, "text": text})
    doc.close()
    return pages


def pages_to_sections(pages: list[dict]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = [("body", "")]
    for page in pages:
        for line in page["text"].split("\n"):
            clean = line.strip().lower().rstrip(".:")
            if any(clean.startswith(h) or clean == h for h in SECTION_HINTS):
                sections.append((line.strip()[:100], ""))
            else:
                name, buf = sections[-1]
                sections[-1] = (name, f"{buf}\n{line}")
    return [(name, buf.strip()) for name, buf in sections if buf.strip()]
