import argparse
import json
import sys

sys.path.insert(0, ".")

from src.services.ingest import ingest_papers


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the RAG system")
    parser.add_argument("--query", required=True, help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()
    results = ingest_papers(query=args.query, max_results=args.max_results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
