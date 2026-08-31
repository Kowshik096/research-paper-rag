import json
import random
import sys

sys.path.insert(0, ".")

from src.config import get_settings
from src.services import pdf_parser
from src.services.embeddings import get_client
from src.services.experiments import load_pdf_texts


def make_question(client, model: str, content: str) -> str:
    prompt = (
        "Read this excerpt from a research paper and write ONE specific research question "
        "that can be answered using ONLY information in this excerpt. "
        "Do not mention 'the excerpt' or 'the paper'. Reply with only the question.\n\n"
        f"EXCERPT:\n{content[:3000]}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip().strip('"')


def main() -> None:
    settings = get_settings()
    client = get_client()
    texts = load_pdf_texts()
    golden = []
    for stem, text in sorted(texts.items()):
        pages = pdf_parser.parse_pdf(f"data/pdfs/{stem}.pdf")
        middle = pages[len(pages) // 2]["text"] if len(pages) > 2 else text
        question = make_question(client, settings.gemini_chat_model, middle)
        if not question or "?" not in question:
            continue
        golden.append({"question": question, "arxiv_id": stem})
        print(f"[{stem}] {question}")

    random.shuffle(golden)
    with open("evals/golden_retrieval.jsonl", "w", encoding="utf-8") as f:
        for row in golden:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved {len(golden)} pairs to evals/golden_retrieval.jsonl")


if __name__ == "__main__":
    main()
