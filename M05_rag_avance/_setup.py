"""
Setup partagé entre les démos M5.

Charge le corpus du M4 et l'indexe dans Qdrant si pas déjà fait.
"""
from __future__ import annotations

import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer

# Réutilise le corpus du M4
M4_DEMOS = Path(__file__).resolve().parent.parent.parent / "M04_vector_search" / "demos"
sys.path.insert(0, str(M4_DEMOS))
from _corpus import get_corpus  # noqa: E402

COLLECTION = "m5_rag_demos"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"


def get_or_create_index() -> tuple[QdrantClient, SentenceTransformer, list[dict]]:
    """Indexe le corpus partagé si nécessaire et retourne (client, model, corpus)."""
    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    corpus = get_corpus()

    if not client.collection_exists(COLLECTION):
        dim = model.get_sentence_embedding_dimension()
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field in ["product", "version", "category"]:
            client.create_payload_index(COLLECTION, field, PayloadSchemaType.KEYWORD)

        texts = [f"passage: {c['content']}" for c in corpus]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        points = [
            PointStruct(id=c["id"], vector=embs[i].tolist(), payload=c)
            for i, c in enumerate(corpus)
        ]
        client.upsert(COLLECTION, points=points, wait=True)

    return client, model, corpus


def search_dense(client: QdrantClient, model: SentenceTransformer, query: str,
                 top_k: int = 5, query_filter=None) -> list:
    """Helper de recherche dense."""
    emb = model.encode(f"query: {query}", normalize_embeddings=True)
    response = client.query_points(
        COLLECTION,
        query=emb.tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return response.points
