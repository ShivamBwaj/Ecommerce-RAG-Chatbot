"""One-time migration: copy local Qdrant collections to a Qdrant Cloud cluster.

Unlike reindex_openai_embeddings.py (which re-embeds because it switches embedding
providers), this script copies vectors directly - the embedding model isn't
changing, only which Qdrant instance holds the data.

Usage:
    export QDRANT_CLOUD_URL=https://xxxxx.cloud.qdrant.io:6333
    export QDRANT_CLOUD_API_KEY=...
    uv run --package api python apps/api/scripts/migrate_to_qdrant_cloud.py
"""
import argparse
import os

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

DEFAULT_COLLECTIONS = [
    "amazon-items-collection-01-hybrid-search",
    "amazon-items-collection-02-openai-small",
    "amazon-items-collection-02-openai-small-reviews",
    "amazon-items-collection-03-hf-reviews",
]


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        client.get_collection(name)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return False
        raise
    return True


def migrate_collection(source: QdrantClient, target: QdrantClient, name: str, batch_size: int, recreate: bool) -> int:
    if not collection_exists(source, name):
        print(f"Skipping {name!r}: not found on source, nothing to migrate.")
        return 0

    source_info = source.get_collection(name)

    if collection_exists(target, name):
        if not recreate:
            info = target.get_collection(name)
            print(f"Skipping {name!r}: already exists on target ({info.points_count} points). Pass --recreate to rebuild it.")
            return 0
        target.delete_collection(name)

    target.create_collection(
        collection_name=name,
        vectors_config=source_info.config.params.vectors,
        sparse_vectors_config=source_info.config.params.sparse_vectors,
    )

    offset = None
    migrated = 0
    while True:
        records, offset = source.scroll(
            collection_name=name,
            offset=offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        target.upsert(
            collection_name=name,
            points=records,
            wait=True,
        )
        migrated += len(records)
        print(f"  [{name}] migrated {migrated} points...")

        if offset is None:
            break

    return migrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy local Qdrant collections to Qdrant Cloud.")
    parser.add_argument("--source-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--target-url", default=os.getenv("QDRANT_CLOUD_URL"))
    parser.add_argument("--target-api-key", default=os.getenv("QDRANT_CLOUD_API_KEY"))
    parser.add_argument("--collections", nargs="*", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.target_url or not args.target_api_key:
        raise SystemExit(
            "Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY (or pass --target-url/--target-api-key)."
        )

    source = QdrantClient(url=args.source_url)
    target = QdrantClient(url=args.target_url, api_key=args.target_api_key)

    for name in args.collections:
        migrated = migrate_collection(source, target, name, args.batch_size, args.recreate)
        if migrated:
            info = target.get_collection(name)
            print(f"Done with {name!r}: migrated {migrated} points, target now reports {info.points_count}.")


if __name__ == "__main__":
    main()
