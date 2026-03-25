"""FastAPI service — upload SEC 10-K JSON files into the Milvus vector store.

Start:
    uvicorn api.main:app --reload --port 8080

Endpoints
---------
GET  /health          liveness check
POST /upload          upload aapl_10k.json-format file
POST /upload/raw      upload generic JSON array with a configurable text field
"""

from __future__ import annotations

import json
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.documents import Document

from agent.vectorstore import COLLECTION_NAME, split_and_add

app = FastAPI(
    title="10-K Knowledge Base API",
    description="Upload SEC 10-K JSON files and ingest them into the Milvus vector store.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_10k_json(content: bytes) -> list[Document]:
    """Parse the standard aapl_10k.json format into Documents.

    Expected structure:
        { "<sql_query>": [ { symbol, file_fiscal_year, section_title,
                              section_id, section_text, ... }, ... ] }
    """
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise HTTPException(status_code=400, detail="JSON must be a non-empty object")

    rows = raw[next(iter(raw))]
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="JSON value must be a list of row objects")

    docs: list[Document] = []
    for i, row in enumerate(rows):
        text = str(row.get("section_text", "")).strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "symbol": str(row.get("symbol", "")),
                    "fiscal_year": int(row.get("file_fiscal_year", 0)),
                    "section_title": str(row.get("section_title", "")),
                    "section_id": int(row.get("section_id", i)),
                    "form_type": str(row.get("form_type", "10-K")),
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/upload", tags=["ingest"])
async def upload_10k(
    file: Annotated[UploadFile, File(description="10-K JSON (aapl_10k.json format)")],
    collection: Annotated[str, Form()] = COLLECTION_NAME,
) -> JSONResponse:
    """Upload a SEC 10-K JSON file and ingest all sections into Milvus.

    The file must follow the same schema as ``aapl_10k.json``:
    one top-level key whose value is a list of rows containing
    ``symbol``, ``file_fiscal_year``, ``section_title``, ``section_id``,
    ``section_text``, and optionally ``form_type``.

    Long sections are automatically chunked with ``RecursiveCharacterTextSplitter``
    (chunk_size=1500, overlap=150) before embedding.
    """
    if not (file.filename or "").endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted")

    content = await file.read()
    docs = _parse_10k_json(content)

    if not docs:
        raise HTTPException(status_code=422, detail="No non-empty sections found in the file")

    try:
        chunks_inserted = split_and_add(docs, collection_name=collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Milvus insert failed: {exc}") from exc

    return JSONResponse(
        {
            "status": "ok",
            "file": file.filename,
            "collection": collection,
            "sections_parsed": len(docs),
            "chunks_inserted": chunks_inserted,
        }
    )


@app.post("/upload/raw", tags=["ingest"])
async def upload_raw(
    file: Annotated[UploadFile, File(description="JSON array of {text_field, **metadata}")],
    text_field: Annotated[str, Form(description="Key to use as page_content")] = "text",
    collection: Annotated[str, Form()] = COLLECTION_NAME,
) -> JSONResponse:
    """Upload a generic JSON array and map any field to ``Document.page_content``.

    Useful for ingesting non-10-K documents. All other fields become metadata.

    Example row::

        {"text": "Apple revenue grew 6%…", "source": "earnings_call", "year": 2025}
    """
    content = await file.read()
    try:
        rows = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="JSON must be an array of objects")

    docs = [
        Document(
            page_content=str(row.get(text_field, "")).strip(),
            metadata={k: v for k, v in row.items() if k != text_field},
        )
        for row in rows
        if str(row.get(text_field, "")).strip()
    ]

    if not docs:
        raise HTTPException(status_code=422, detail=f"No rows with '{text_field}' field found")

    try:
        chunks_inserted = split_and_add(docs, collection_name=collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Milvus insert failed: {exc}") from exc

    return JSONResponse(
        {
            "status": "ok",
            "file": file.filename,
            "collection": collection,
            "docs_parsed": len(docs),
            "chunks_inserted": chunks_inserted,
        }
    )
