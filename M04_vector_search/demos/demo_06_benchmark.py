from __future__ import annotations

import re
import statistics
import time

import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from _corpus import get_corpus

console = Console()
COLLECTION = "demo_06_bench"
MODEL_NAME = "intfloat/multilingual-e5-small"


# Golden set : 10 questions avec leurs IDs de chunks pertinents
GOLDEN_SET = [
    {"q": "Comment configurer SAML 2.0 ?", "relevant": [1, 5]},
    {"q": "Quelles sont les méthodes d'authentification supportées ?", "relevant": [1, 2, 3, 4, 5]},
    {"q": "Erreur 503 sur les rapports", "relevant": [8]},
    {"q": "Limite de taux par défaut sur l'API", "relevant": [9, 15]},
    {"q": "Comment exporter les données en bulk", "relevant": [12]},
    {"q": "Politique de sauvegarde et restauration", "relevant": [32]},
    {"q": "Comment changer de plan tarifaire ?", "relevant": [24]},
    {"q": "RGPD et conformité européenne", "relevant": [29, 31, 50]},
    {"q": "Migration depuis la version 2", "relevant": [26]},
    {"q": "Intégration avec Slack", "relevant": [38]},
]


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def rrf_fusion(rankings: list[list[int]], k: int = 60, top_n: int = 10) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Recall@k : proportion de documents pertinents retrouvés dans les k premiers résultats."""
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """MRR : 1/rang du premier document pertinent trouvé (mesure de précision)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def main() -> None:
    console.print("\n[bold orange3]Demo 06 — Benchmark dense vs sparse vs hybride[/bold orange3]\n")

    model = SentenceTransformer(MODEL_NAME)
    corpus = get_corpus()

    # Setup index dense (Qdrant)
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = model.get_embedding_dimension()
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    embeddings = model.encode([f"passage: {c['content']}" for c in corpus], normalize_embeddings=True, show_progress_bar=False)
    client.upsert(COLLECTION, points=[PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload=c) for i, c in enumerate(corpus)], wait=True)

    # Setup index sparse (BM25)
    tokenized = [tokenize(c["content"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    id_by_idx = [c["id"] for c in corpus]
    
    # Warmup : pour stabiliser les performances
    for _ in range(5):
      model.encode("query: test", normalize_embeddings=True)

    # Benchmark : évaluation de chaque mode sur le golden set
    metrics = {"dense": {}, "sparse": {}, "hybrid": {}}

    for mode in ["dense", "sparse", "hybrid"]:
        recalls5: list[float] = []
        recalls10: list[float] = []
        rrs: list[float] = []
        latencies: list[float] = []

        for case in GOLDEN_SET:
            q = case["q"]
            relevant = set(case["relevant"])

            # Mesure de la latence pour chaque mode
            t0 = time.perf_counter()
            if mode == "dense":
                emb = model.encode(f"query: {q}", normalize_embeddings=True)
                hits = client.query_points(COLLECTION, query=emb.tolist(), limit=10).points
                retrieved = [hit.id for hit in hits]
            elif mode == "sparse":
                scores = bm25.get_scores(tokenize(q))
                top_idx = sorted(range(len(corpus)), key=lambda i: -scores[i])[:10]
                retrieved = [id_by_idx[i] for i in top_idx]
            else:  # hybrid
                emb = model.encode(f"query: {q}", normalize_embeddings=True)
                dense_hits = client.query_points(COLLECTION, query=emb.tolist(), limit=20).points
                dense_ranking = [hit.id for hit in dense_hits]

                scores = bm25.get_scores(tokenize(q))
                top_idx = sorted(range(len(corpus)), key=lambda i: -scores[i])[:20]
                sparse_ranking = [id_by_idx[i] for i in top_idx]

                retrieved = rrf_fusion([dense_ranking, sparse_ranking], top_n=10)

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
                

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            # Calcul des métriques de qualité
            recalls5.append(recall_at_k(retrieved, relevant, k=5))
            recalls10.append(recall_at_k(retrieved, relevant, k=10))
            rrs.append(reciprocal_rank(retrieved, relevant))

        # Agrégation des métriques : moyennes pour qualité, percentiles pour latence
        metrics[mode] = {
            "recall@5": statistics.mean(recalls5),
            "recall@10": statistics.mean(recalls10),
            "mrr": statistics.mean(rrs),
            "p50_ms": statistics.median(latencies),  # Médiane (latence typique)
            "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],  # 95e percentile
        }

    # Tableau
    table = Table(title="Benchmark final", show_lines=True)
    table.add_column("Mode", style="cyan")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")

    for mode, m in metrics.items():
        style = "bold orange3" if mode == "hybrid" else ""
        table.add_row(
            mode,
            f"{m['recall@5']:.3f}",
            f"{m['recall@10']:.3f}",
            f"{m['mrr']:.3f}",
            f"{m['p50_ms']:.1f}",
            f"{m['p95_ms']:.1f}",
            style=style,
        )
    console.print(table)

    # Génération du graphique comparatif
    modes = list(metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Graphique 1 : Qualité (recall, MRR)
    metrics_to_plot = ["recall@5", "recall@10", "mrr"]
    x = range(len(metrics_to_plot))
    width = 0.25
    colors = ["#FF6B35", "#0066FF", "#00A86B"]
    for i, mode in enumerate(modes):
        values = [metrics[mode][m] for m in metrics_to_plot]
        axes[0].bar([xi + i * width for xi in x], values, width, label=mode, color=colors[i])
    axes[0].set_xticks([xi + width for xi in x])
    axes[0].set_xticklabels(metrics_to_plot)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Qualité du retrieval")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Graphique 2 : Latence (p50 et p95)
    p50s = [metrics[m]["p50_ms"] for m in modes]
    p95s = [metrics[m]["p95_ms"] for m in modes]
    x2 = range(len(modes))
    axes[1].bar([xi - 0.2 for xi in x2], p50s, 0.4, label="p50", color="#FF6B35")
    axes[1].bar([xi + 0.2 for xi in x2], p95s, 0.4, label="p95", color="#1A1A1A")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(modes)
    axes[1].set_ylabel("Latence (ms)")
    axes[1].set_title("Latence")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output = "demo_06_benchmark.png"
    plt.savefig(output, dpi=120)
    console.print(f"\n[green]Graphique sauvé : {output}[/green]")

    console.print("\n[bold]Lecture pédagogique :[/bold]")
    console.print("• Sur ce corpus, l'hybride domine généralement sur recall et MRR")
    console.print("• La latence hybride est légèrement supérieure (2 recherches + fusion)")
    console.print("• Sur des corpus plus complexes (techniques), l'écart se creuse encore plus en faveur de l'hybride\n")


if __name__ == "__main__":
    main()
