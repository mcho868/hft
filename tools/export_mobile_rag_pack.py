#!/usr/bin/env python3
"""
Export a compact offline RAG pack for the iOS POC.

- Reads one or more chunk JSON files (as generated in your RAG pipeline)
- Selects up to N chunks per source
- Writes:
  - chunks.jsonl  (one JSON object per line: {id, source, doc_id, text})
  - manifest.json (counts and basic stats)

Notes
- This POC targets BM25 on-device to keep the app small and fully offline.
- If you later add an on-device embedding model (MLX Swift), you can extend
  this exporter to also emit precomputed embeddings for the subset.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize into dicts with a text field
    out = []
    for item in data:
        if isinstance(item, str):
            out.append({
                "text": item,
                "source_document": "unknown"
            })
        else:
            out.append(item)
    return out


def sanitize_text(text: str) -> str:
    # Lightweight cleanup for mobile
    text = text.replace("\u0000", " ").strip()
    # Collapse excessive whitespace
    return " ".join(text.split())


def main():
    p = argparse.ArgumentParser(description="Export a mobile RAG pack (BM25-ready)")
    p.add_argument("--chunks-paths", nargs="+", required=True, help="Paths to chunk JSON files")
    p.add_argument("--max-per-source", type=int, default=2000, help="Max chunks per source file")
    p.add_argument("--output-dir", default="mobile_poc_ios/Resources/mobile_rag_pack", help="Output directory")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    manifest = {
        "sources": [],
        "total_chunks": 0,
        "notes": "BM25-only RAG pack for iOS POC. Extend with embeddings if needed."
    }

    idx = 0
    for pth in args.chunks_paths:
        src_path = Path(pth)
        src_name = src_path.stem
        try:
            chunks = load_chunks(src_path)
        except Exception as e:
            print(f"⚠️  Failed to load {src_path}: {e}")
            continue

        selected = []
        for c in chunks:
            text = sanitize_text(c.get("text") or c.get("original_text") or "")
            if len(text) < 60:
                continue
            selected.append({
                "doc_id": str(c.get("source_document") or c.get("doc_id") or "unknown"),
                "source": src_name,
                "text": text
            })
            if len(selected) >= args.max_per_source:
                break

        for rec in selected:
            rec_out = {
                "id": idx,
                "source": rec["source"],
                "doc_id": rec["doc_id"],
                "text": rec["text"],
            }
            all_records.append(rec_out)
            idx += 1

        manifest["sources"].append({
            "file": str(src_path),
            "source_name": src_name,
            "selected": len(selected)
        })

    manifest["total_chunks"] = len(all_records)

    # Write files
    chunks_jsonl = out_dir / "chunks.jsonl"
    manifest_json = out_dir / "manifest.json"

    with open(chunks_jsonl, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(all_records)} chunks → {chunks_jsonl}")
    print(f"🧾 Manifest → {manifest_json}")


if __name__ == "__main__":
    main()

