"""Ingest aapl_10k.json (or any compatible JSON) into Milvus.

Usage:
    python -m agent.ingest [--data PATH] [--collection NAME]

Expected JSON format — a dict with one key whose value is a list of rows
with fields: symbol, file_fiscal_year, section_title, section_id, section_text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

from agent.vectorstore import COLLECTION_NAME, split_and_add

BATCH_SIZE = 16


def json_to_documents(path: Path) -> list[Document]:
    """Convert a SEC 10-K JSON file to LangChain Documents."""
    with open(path) as f:
        raw = json.load(f)

    # Top-level key is the SQL query string; value is the rows list
    rows: list[dict] = raw[next(iter(raw))]

    return [
        Document(
            page_content=row["section_text"],
            metadata={
                "symbol": row["symbol"],
                "fiscal_year": int(row["file_fiscal_year"]),
                "section_title": row["section_title"],
                "section_id": int(row["section_id"]),
                "form_type": row.get("form_type", "10-K"),
            },
        )
        for row in rows
        if row.get("section_text", "").strip()
    ]


def run(data_path: Path, collection: str) -> None:
    print(f"Loading {data_path} …")
    docs = json_to_documents(data_path)
    print(f"  {len(docs)} sections found")

    print(f"Splitting + embedding into '{collection}' (batch_size={BATCH_SIZE}) …")
    total = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        count = split_and_add(batch, collection_name=collection)
        total += count
        end = min(i + BATCH_SIZE, len(docs))
        print(f"  [{end}/{len(docs)}] → {count} chunks inserted (total {total})")

    print(f"\nDone. {total} chunks stored in collection '{collection}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 10-K JSON into Milvus")
    parser.add_argument("--data", default="./aapl_10k.json")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()
    run(Path(args.data), args.collection)
