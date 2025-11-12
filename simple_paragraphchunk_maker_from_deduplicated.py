#!/usr/bin/env python3
"""
Simple Paragraph Chunk Maker (from deduplicated agentic chunks)

- Reads deduplicated agent-generated chunk files
  (e.g., *_chunks_agent*_deduplicated.json) from RAG_DATA_DIR
- Groups chunks by source_document (normalized to ignore leading numeric prefixes)
- Concatenates texts per document in source order into paragraph-level chunks
- Saves new JSON and builds a FAISS index for retrieval
"""

import os
import re
import glob
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_DATA_DIR = "/Users/choemanseung/789/hft/RAGdatav2"


def _normalize_source_document(source_document: str) -> str:
    """Normalize source document names to ignore leading numeric prefixes.

    Examples:
    - "0023_Filename.txt" -> "filename"
    - "12-File-Name.TXT" -> "file-name"
    - "00045 File name.txt" -> "file name"
    """
    if not source_document:
        return ""

    base_name = os.path.basename(source_document).lower()
    stem, _ = os.path.splitext(base_name)
    stem = re.sub(r"^\s*\d+[\s_\-–—\.\)]*", "", stem)
    return stem.strip()


def _generate_embeddings(texts: List[str]) -> np.ndarray:
    print(f"Generating embeddings for {len(texts)} paragraphs...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.asarray(embeddings, dtype="float32")


def _build_faiss_index(embeddings: np.ndarray, index_path: str) -> None:
    print(f"Building FAISS index with {embeddings.shape[0]} vectors...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _combine_chunks_to_paragraphs(chunks: List[Dict]) -> List[Dict]:
    """Group chunks by normalized source_document and combine their texts.

    - Sort chunks within each document by source_char_start when available
    - Join texts with double newlines to preserve readability
    - Compute overall source_char_start (min) and source_char_end (max)
    - Keep a representative raw source_document (first encountered)
    """
    # Group by normalized source document
    doc_to_chunks: Dict[str, List[Dict]] = defaultdict(list)
    doc_rep_name: Dict[str, str] = {}

    for ch in chunks:
        raw_doc = ch.get("source_document", "")
        norm_doc = _normalize_source_document(raw_doc)
        if norm_doc not in doc_rep_name and raw_doc:
            doc_rep_name[norm_doc] = raw_doc
        doc_to_chunks[norm_doc].append(ch)

    paragraph_chunks: List[Dict] = []

    for norm_doc, doc_chunks in doc_to_chunks.items():
        # Sort by source_char_start if present
        doc_chunks_sorted = sorted(
            doc_chunks,
            key=lambda c: _safe_int(c.get("source_char_start", 0), 0),
        )

        texts_in_order: List[str] = []
        for c in doc_chunks_sorted:
            t = (c.get("text", "") or "").strip()
            if not t:
                continue
            # Avoid immediate duplicates while preserving order
            if not texts_in_order or t != texts_in_order[-1]:
                texts_in_order.append(t)

        combined_text = "\n\n".join(texts_in_order)

        # Aggregate character boundaries when available
        starts = [c.get("source_char_start") for c in doc_chunks_sorted if isinstance(c.get("source_char_start"), int)]
        ends = [c.get("source_char_end") for c in doc_chunks_sorted if isinstance(c.get("source_char_end"), int)]
        overall_start = min(starts) if starts else 0
        overall_end = max(ends) if ends else 0

        paragraph_chunks.append(
            {
                # chunk_id will be assigned later
                "text": combined_text,
                "source_document": doc_rep_name.get(norm_doc, norm_doc),
                "source_char_start": overall_start,
                "source_char_end": overall_end,
            }
        )

    return paragraph_chunks


def _assign_chunk_ids(chunks: List[Dict], base_name: str) -> None:
    for i, ch in enumerate(chunks):
        ch["chunk_id"] = f"{base_name}_chunk_{i}"


def _derive_output_paths(input_json_path: str) -> Tuple[str, str, str]:
    base_name = os.path.basename(input_json_path)
    name_wo_ext = os.path.splitext(base_name)[0]

    # Derive chunk_id base from the filename when possible
    # Prefer replacing "_chunks_agent_" with "_chunks_paragraph_"
    chunk_id_base = name_wo_ext
    if "_chunks_agent_" in chunk_id_base:
        chunk_id_base = chunk_id_base.replace("_chunks_agent_", "_chunks_paragraph_")
    else:
        chunk_id_base = f"{chunk_id_base}_paragraph"

    output_json = os.path.join(RAG_DATA_DIR, f"{chunk_id_base}.json")
    output_index = os.path.join(
        RAG_DATA_DIR, chunk_id_base.replace("_chunks_", "_vector_db_") + ".index"
    )

    return chunk_id_base, output_json, output_index


def process_file(input_json_path: str) -> None:
    print(f"\nProcessing (paragraph build): {os.path.basename(input_json_path)}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Input chunks (deduplicated): {len(chunks)}")

    paragraph_chunks = _combine_chunks_to_paragraphs(chunks)
    print(f"Paragraph chunks (grouped by document): {len(paragraph_chunks)}")

    # Derive outputs and assign chunk IDs
    chunk_id_base, output_json, output_index = _derive_output_paths(input_json_path)
    _assign_chunk_ids(paragraph_chunks, chunk_id_base)

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(paragraph_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {os.path.basename(output_json)}")

    # Build vector index
    embeddings = _generate_embeddings([c.get("text", "") for c in paragraph_chunks])
    _build_faiss_index(embeddings, output_index)
    print(f"Saved index: {os.path.basename(output_index)}")


def main() -> None:
    print("=" * 60)
    print("SIMPLE PARAGRAPH CHUNK MAKER (from deduplicated agentic chunks)")
    print("- Groups by source_document and concatenates texts per document")
    print("=" * 60)

    # Look specifically for deduplicated agent chunk files
    pattern = os.path.join(RAG_DATA_DIR, "*_chunks_agent*_deduplicated.json")
    files = sorted(set(glob.glob(pattern)))

    if not files:
        print("No deduplicated agent chunk files found!")
        return

    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    for file_path in files:
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    print("\n" + "=" * 60)
    print("PARAGRAPH BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


