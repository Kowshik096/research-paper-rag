from pathlib import Path

import arxiv
import httpx

DATA_DIR = Path("data/pdfs")


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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = arxiv_id.replace("/", "_")
    path = DATA_DIR / f"{safe_name}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return path
    with httpx.stream("GET", pdf_url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_bytes(65536):
                f.write(chunk)
    return path
