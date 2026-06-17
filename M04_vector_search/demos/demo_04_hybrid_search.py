from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
import re

from _corpus import get_corpus

console = Console()
COLLECTION = "demo_04_hybrid"
MODEL_NAME = "intfloat/multilingual-e5-small"


def tokenize(text: str) -> list[str]:
    """Tokenisation simple pour BM25 : minuscule + split sur non-alphanumérique."""
    return [t for t in re.split(r"\W+", text.lower()) if t]


def rrf_fusion(rankings: list[list[int]], k: int = 60, top_n: int = 5) -> list[tuple[int, float]]:
    """Fusion par Reciprocal Rank Fusion : combine plusieurs rankings sans scores normalisés."""
    scores: dict[int, float] = {}
    # Pour chaque ranking (dense, sparse, etc.), ajoute un score inversement proportionnel au rang
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]


def main() -> None:
    console.print("\n[bold orange3]Demo 04 — Recherche hybride dense + sparse + RRF[/bold orange3]\n")

    model = SentenceTransformer(MODEL_NAME)
    corpus = get_corpus()

    # --- Setup index dense (embeddings vectoriels avec Qdrant) ---
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    dim = model.get_embedding_dimension()
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    texts_passage = [f"passage: {c['content']}" for c in corpus]
    embeddings = model.encode(texts_passage, normalize_embeddings=True, show_progress_bar=False)

    points = [PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload=c) for i, c in enumerate(corpus)]
    client.upsert(COLLECTION, points=points, wait=True)
    console.print(f"[green]✓ Index dense Qdrant : {len(points)} points[/green]")

    # --- Setup index sparse (BM25 pour la correspondance lexicale) ---
    # BM25 : algorithme de ranking basé sur la fréquence des mots (TF-IDF amélioré)
    tokenized = [tokenize(c["content"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    id_by_idx = [c["id"] for c in corpus]
    console.print(f"[green]✓ Index BM25 in-memory : {len(tokenized)} docs[/green]\n")

    # --- Cas de test choisis pour montrer les forces/faiblesses ---
    queries = [
        ("Cas 1 — Paraphrase sémantique (avantage dense)", "Comment réinitialiser un mot de passe ou récupérer un accès ?"),
        ("Cas 2 — Référence technique exacte (avantage sparse)", "Erreur 503 sur /api/reports en version 3.2"),
        ("Cas 3 — Mixte (l'hybride doit gagner)", "Comment configurer SAML 2.0 sur la version 3.2 ?"),
    ]

    for label, query in queries:
        console.print(f"\n[bold cyan]{label}[/bold cyan]")
        console.print(f"[dim]{query}[/dim]\n")

        # Recherche dense (similarité sémantique vectorielle)
        query_emb = model.encode(f"query: {query}", normalize_embeddings=True)
        dense_hits = client.query_points(COLLECTION, query=query_emb.tolist(), limit=10, with_payload=True).points
        dense_ranking = [hit.id for hit in dense_hits]

        # Recherche sparse (correspondance lexicale BM25)
        scores = bm25.get_scores(tokenize(query))
        sparse_idx_ranked = sorted(range(len(corpus)), key=lambda i: -scores[i])[:10]
        sparse_ranking = [id_by_idx[i] for i in sparse_idx_ranked]

        # Recherche hybride : fusion RRF des deux rankings
        fused = rrf_fusion([dense_ranking, sparse_ranking], k=60, top_n=5)
        fused_ids = [doc_id for doc_id, _ in fused]

        # Top-5 de chaque
        title_by_id = {c["id"]: c["title"] for c in corpus}

        table = Table(show_lines=False)
        table.add_column("Rang", justify="center")
        table.add_column("Dense (cosine)", overflow="fold")
        table.add_column("Sparse (BM25)", overflow="fold")
        table.add_column("Hybrid (RRF)", overflow="fold", style="bold orange3")

        for rank in range(5):
            d_id = dense_ranking[rank] if rank < len(dense_ranking) else None
            s_id = sparse_ranking[rank] if rank < len(sparse_ranking) else None
            h_id = fused_ids[rank] if rank < len(fused_ids) else None
            table.add_row(
                str(rank + 1),
                f"#{d_id} {title_by_id.get(d_id, '')}" if d_id else "",
                f"#{s_id} {title_by_id.get(s_id, '')}" if s_id else "",
                f"#{h_id} {title_by_id.get(h_id, '')}" if h_id else "",
            )
        console.print(table)

    console.print("\n[bold]Observations attendues :[/bold]")
    console.print("• Cas 1 : le dense trouve via paraphrase, le sparse passe à côté")
    console.print("• Cas 2 : le sparse trouve la doc exacte 503/api/reports/3.2 que le dense rate")
    console.print("• Cas 3 : la fusion RRF combine les deux et retrouve les meilleures sources\n")


if __name__ == "__main__":
    main()
