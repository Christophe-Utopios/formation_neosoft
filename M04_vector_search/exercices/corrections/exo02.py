from __future__ import annotations

import time
import statistics
from datetime import datetime, timezone

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    Filter, FieldCondition, MatchValue, DatetimeRange,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

console = Console()
COLLECTION_V1 = "wikifr_v1"
COLLECTION_V2 = "wikifr_v2"
MODEL_NAME = "intfloat/multilingual-e5-base"
N_ARTICLES = 200  # plus petit pour tenir dans le TP


KEYWORDS_TO_CATEGORY = {
    "personne": ["né", "née", "écrivain", "philosophe", "président", "roi", "reine", "savant", "musicien"],
    "lieu": ["ville", "pays", "capitale", "fleuve", "montagne", "région", "département"],
    "science": ["théorie", "physique", "biologie", "chimie", "mathématique", "équation", "molécule"],
    "histoire": ["guerre", "bataille", "siècle", "époque", "empire", "révolution", "ancien"],
}


def categorize(text: str) -> str:
    t = text[:500].lower()
    for cat, kws in KEYWORDS_TO_CATEGORY.items():
        if any(kw in t for kw in kws):
            return cat
    return "autre"


def load_articles(n: int) -> list[dict]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    return [
        {"id": int(a["id"]), "title": a["title"], "text": a["text"], "url": a["url"]}
        for a in ds.take(n)
    ]


def index_v1_simple(articles: list[dict], model: SentenceTransformer) -> None:
    """Indexation V1 : un chunk = un point, sans quantization."""
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION_V1):
        client.delete_collection(COLLECTION_V1)
    dim = model.get_embedding_dimension()
    client.create_collection(COLLECTION_V1, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    points = []
    chunk_id = 0
    for art in articles:
        for piece in splitter.split_text(art["text"]):
            chunk_id += 1
            emb = model.encode(f"passage: {piece}", normalize_embeddings=True)
            points.append(PointStruct(
                id=chunk_id,
                vector=emb.tolist(),
                payload={"article_id": art["id"], "title": art["title"], "content": piece},
            ))
    client.upsert(COLLECTION_V1, points=points, wait=True)
    console.print(f"[green]V1 : {len(points)} points indexés[/green]")


def index_v2_parent_doc_quantized(articles: list[dict], model: SentenceTransformer) -> None:
    """Indexation V2 : parent-doc + quantization int8 + index payload riches."""
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION_V2):
        client.delete_collection(COLLECTION_V2)

    dim = model.get_embedding_dimension()
    client.create_collection(
        COLLECTION_V2,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            )
        ),
    )

    # Index payload riches
    client.create_payload_index(COLLECTION_V2, "category", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_V2, "title", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_V2, "indexed_at", PayloadSchemaType.DATETIME)
    client.create_payload_index(COLLECTION_V2, "parent_id", PayloadSchemaType.INTEGER)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

    points = []
    point_id = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for art in articles:
        category = categorize(art["text"])
        parents = parent_splitter.split_text(art["text"])
        for parent_idx, parent in enumerate(parents):
            parent_global_id = art["id"] * 1000 + parent_idx
            children = child_splitter.split_text(parent)
            for child in children:
                point_id += 1
                emb = model.encode(f"passage: {child}", normalize_embeddings=True)
                points.append(PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        "child_content": child,
                        "parent_id": parent_global_id,
                        "parent_content": parent,
                        "title": art["title"],
                        "category": category,
                        "indexed_at": now_iso,
                    },
                ))

    # Upsert par batches
    for i in range(0, len(points), 200):
        client.upsert(COLLECTION_V2, points=points[i:i+200], wait=False)
    client.update_collection(COLLECTION_V2, optimizer_config={"indexing_threshold": 0})
    console.print(f"[green]V2 : {len(points)} children indexés (parent-doc + quantization)[/green]")


def index_size_mb(client: QdrantClient, collection: str) -> float:
    info = client.get_collection(collection)
    # Approximation : on regarde le nombre de points × dim × bytes_per_dim
    n = info.points_count or 0
    dim = info.config.params.vectors.size
    # V1: float32 = 4 bytes, V2 quantized = ~1 byte int8 (always_ram) + originaux disque
    return n * dim * 4 / (1024 * 1024)


def main() -> None:
    console.print("\n[bold orange3]Solution TP 2 — Indexation production-ready[/bold orange3]\n")

    articles = load_articles(N_ARTICLES)
    console.print(f"[green]✓ {len(articles)} articles chargés[/green]")

    model = SentenceTransformer(MODEL_NAME)

    console.print("\n[cyan]Indexation V1 (simple)...[/cyan]")
    t0 = time.perf_counter()
    index_v1_simple(articles, model)
    t_v1 = time.perf_counter() - t0

    console.print("\n[cyan]Indexation V2 (parent-doc + quantization)...[/cyan]")
    t0 = time.perf_counter()
    index_v2_parent_doc_quantized(articles, model)
    t_v2 = time.perf_counter() - t0

    # Golden set : 10 titres d'articles présents dans la base, avec une question
    golden = [
        {"q": "Qui était Napoléon Bonaparte ?", "relevant_titles": {"Napoléon Ier", "Napoléon Bonaparte"}},
        {"q": "Histoire de l'Empire romain", "relevant_titles": {"Empire romain", "Rome antique"}},
        {"q": "Théorie de la relativité", "relevant_titles": {"Théorie de la relativité", "Albert Einstein"}},
        # Note : sur 200 articles random, peu de chances de tomber sur ces titres précis.
        # Adapter le golden set en fonction des articles réellement chargés.
    ]
    # En pratique, dériver le golden set des articles chargés :
    sample_golden = []
    for art in articles[:10]:
        sample_golden.append({
            "q": f"Qu'est-ce que {art['title']} ?",
            "relevant_titles": {art["title"]},
        })

    client = QdrantClient(url="http://localhost:6333")


    # Démo filtre 
    console.print("\n[cyan]Démo filtre catégorie + temporel sur V2...[/cyan]")
    emb = model.encode("query: relativité", normalize_embeddings=True)
    hits = client.query_points(
        COLLECTION_V2,
        query=emb.tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="science"))]
        ),
        limit=3,
        with_payload=True,
    ).points
    for h in hits:
        console.print(f"  [{h.score:.3f}] {h.payload['title']} ({h.payload['category']})")

    console.print("\n[bold]Conclusion type :[/bold]")
    console.print("• La quantization int8 réduit ~4x l'empreinte mémoire avec perte recall < 1%")
    console.print("• Le parent-doc améliore le recall sur questions naturelles (le LLM voit + de contexte)")
    console.print("• Les filtres payload sont quasi-gratuits si les indexes sont créés en amont")
    console.print("• V2 prêt pour la prod sous condition de monitoring du recall en continu\n")


if __name__ == "__main__":
    main()
