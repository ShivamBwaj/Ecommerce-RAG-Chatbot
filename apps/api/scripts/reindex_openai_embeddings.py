import argparse

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Document,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from api.core.config import config
from api.core.embeddings import get_embeddings


SOURCE_COLLECTION = "amazon-items-collection-01-hybrid-search"


def collection_exists(qdrant_client: QdrantClient, collection_name: str) -> bool:
    try:
        qdrant_client.get_collection(collection_name)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return False
        raise
    return True


def create_target_collection(
    qdrant_client: QdrantClient,
    target_collection: str,
    recreate: bool,
) -> None:
    if collection_exists(qdrant_client, target_collection):
        if not recreate:
            info = qdrant_client.get_collection(target_collection)
            raise RuntimeError(
                f"Target collection {target_collection!r} already exists with "
                f"{info.points_count} points. Pass --recreate to rebuild it."
            )
        qdrant_client.delete_collection(target_collection)

    qdrant_client.create_collection(
        collection_name=target_collection,
        vectors_config={
            config.qdrant_dense_vector_name: VectorParams(
                size=config.embedding_dimensions,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            config.QDRANT_SPARSE_VECTOR_NAME: SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def migrate_collection(
    qdrant_client: QdrantClient,
    source_collection: str,
    target_collection: str,
    batch_size: int,
) -> int:
    offset = None
    migrated = 0

    while True:
        records, offset = qdrant_client.scroll(
            collection_name=source_collection,
            offset=offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            break

        descriptions = [
            str(record.payload.get("description") or "")
            for record in records
        ]
        embeddings = get_embeddings(descriptions)

        points = []
        for record, description, embedding in zip(records, descriptions, embeddings):
            points.append(
                PointStruct(
                    id=record.id,
                    vector={
                        config.qdrant_dense_vector_name: embedding,
                        config.QDRANT_SPARSE_VECTOR_NAME: Document(
                            text=description,
                            model="qdrant/bm25",
                        ),
                    },
                    payload=record.payload,
                )
            )

        qdrant_client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )
        migrated += len(points)
        print(f"Migrated {migrated} points...")

        if offset is None:
            break

    return migrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reindex the Amazon items Qdrant collection with OpenAI embeddings."
    )
    parser.add_argument(
        "--source-collection",
        default=SOURCE_COLLECTION,
        help="Existing collection to copy payloads from.",
    )
    parser.add_argument(
        "--target-collection",
        default=config.QDRANT_COLLECTION,
        help="New OpenAI-backed collection to create.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of points to embed/upsert per batch.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the target collection if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    if config.embedding_provider != "openai":
        raise RuntimeError("Set EMBEDDING_PROVIDER=openai before running this migration.")

    args = parse_args()
    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    create_target_collection(
        qdrant_client=qdrant_client,
        target_collection=args.target_collection,
        recreate=args.recreate,
    )
    migrated = migrate_collection(
        qdrant_client=qdrant_client,
        source_collection=args.source_collection,
        target_collection=args.target_collection,
        batch_size=args.batch_size,
    )

    info = qdrant_client.get_collection(args.target_collection)
    print(
        f"Done. Migrated {migrated} points into {args.target_collection!r}. "
        f"Collection now reports {info.points_count} points."
    )


if __name__ == "__main__":
    main()
