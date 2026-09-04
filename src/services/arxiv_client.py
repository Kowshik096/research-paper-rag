import re
from pathlib import Path

import arxiv
import httpx

DATA_DIR = Path("data/pdfs")


def sanitize_arxiv_id(arxiv_id: str) -> str:
    """Sanitize arXiv ID to a safe base filename without directory traversal."""
    if not arxiv_id or not arxiv_id.strip():
        raise ValueError("Empty arxiv_id provided")
    cleaned = re.sub(r"[/\\]+", "_", arxiv_id)
    cleaned = re.sub(r"\.{2,}", "_", cleaned)
    cleaned = re.sub(r"[^\w\.-]", "_", cleaned)
    base_name = Path(cleaned).name.lstrip(".")
    if not base_name:
        raise ValueError(f"Invalid arxiv_id: {arxiv_id}")
    return base_name


def get_pdf_path(arxiv_id: str) -> Path:
    """Get the safe resolved path for an arXiv PDF, asserting it remains within DATA_DIR."""
    safe_name = sanitize_arxiv_id(arxiv_id)
    target_dir = DATA_DIR.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = (target_dir / f"{safe_name}.pdf").resolve()
    if target_dir not in resolved_path.parents and resolved_path != target_dir:
        raise ValueError(f"Path traversal detected for arxiv_id: {arxiv_id}")
    return resolved_path


def fetch_papers(query: str, max_results: int) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for result in client.results(search):
        short_id = result.get_short_id()
        papers.append(
            {
                "arxiv_id": short_id,
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors),
                "abstract": result.summary.replace("\n", " "),
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": f"https://arxiv.org/pdf/{short_id}",
            }
        )
    return papers


def download_pdf(arxiv_id: str, pdf_url: str) -> Path:
    path = get_pdf_path(arxiv_id)
    if path.exists() and path.stat().st_size > 0:
        return path
    with httpx.stream("GET", pdf_url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_bytes(65536):
                f.write(chunk)
    return path
