"""Inspect the built index (plan Phase 2): filter chunks, or run a raw dense / BM25 query.

    .venv/Scripts/python scripts/inspect_index.py --modality row_fact --page 141
    .venv/Scripts/python scripts/inspect_index.py --bm25 "inventories" -k 5
    .venv/Scripts/python scripts/inspect_index.py --dense "inventories in fiscal 2026" -k 5
    .venv/Scripts/python scripts/inspect_index.py --index data/index_bge_base --dense "goodwill"

Runs in the serving environment (chromadb, fastembed, rank-bm25); no Docling needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.core.settings import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--index", type=Path, default=PROJECT_ROOT / "data" / "index")
    ap.add_argument("--modality", choices=["text", "table", "row_fact", "figure"])
    ap.add_argument("--page", type=int)
    ap.add_argument("--section")
    ap.add_argument("--subsection")
    ap.add_argument("--dense", metavar="QUERY", help="cosine search through Chroma")
    ap.add_argument("--bm25", metavar="QUERY", help="lexical search through the pickled BM25 index")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="print whole documents")
    args = ap.parse_args(argv)

    import chromadb

    from rag.ingest.index import COLLECTION_NAME, load_manifest

    manifest = load_manifest(args.index)
    print(
        f"index={args.index}  corpus_version={manifest.corpus_version}  "
        f"embedder={manifest.embedder} ({manifest.embedding_dim}-d)  chunks={manifest.chunk_counts}"
    )
    client = chromadb.PersistentClient(path=str(args.index / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)

    where_clauses = []
    if args.modality:
        where_clauses.append({"modality": args.modality})
    if args.page is not None:
        where_clauses.append({"page": args.page})
    if args.section:
        where_clauses.append({"section": args.section})
    if args.subsection:
        where_clauses.append({"subsection": args.subsection})
    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif where_clauses:
        where = {"$and": where_clauses}

    def show(id_: str, doc: str, meta: dict, score: str = "") -> None:
        head = (
            f"{id_}  [{meta.get('modality')} p{meta.get('page')} {meta.get('subsection')}] {score}"
        )
        print(head)
        body = doc if args.full else doc.replace("\n", " ⏎ ")[:220]
        print("   ", body)

    if args.dense:
        from rag.query.store import IndexStore

        store = IndexStore(args.index, figures_root=PROJECT_ROOT / "data" / "index")
        print(f"\nDENSE  {args.dense!r}  (exact cosine, D-54)")
        for id_, score in store.dense_search(args.dense, args.k, where=where):
            chunk = store.get(id_)
            show(id_, chunk.document, chunk.chroma_metadata(), f"cos={score:.3f}")

    if args.bm25:
        from rag.core.bm25 import BM25Index

        bm25 = BM25Index.load(args.index / "bm25.pkl")
        hits = bm25.search(args.bm25, k=args.k * 4 if where else args.k)
        ids = [h[0] for h in hits]
        got = (
            collection.get(ids=ids, include=["documents", "metadatas"], where=where)
            if ids
            else {"ids": []}
        )
        by_id = {
            i: (d, m)
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        }
        print(f"\nBM25  {args.bm25!r}")
        shown = 0
        for id_, score in hits:
            if id_ in by_id:
                show(id_, by_id[id_][0], by_id[id_][1], f"bm25={score:.2f}")
                shown += 1
                if shown >= args.k:
                    break

    if not args.dense and not args.bm25:
        got = collection.get(where=where, limit=args.k, include=["documents", "metadatas"])
        print(f"\n{len(got['ids'])} chunk(s) (filters: {where})")
        for id_, doc, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=True):
            show(id_, doc, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
