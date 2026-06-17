from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "demos"))
from _corpus import get_corpus

console = Console()
COLLECTION = "tp3_bench"
MODEL_NAME = "intfloat/multilingual-e5-small"


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def setup(corpus, model):
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = model.get_sentence_embedding_dimension()
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    embeddings = model.encode([f"passage: {c['content']}" for c in corpus],
                              normalize_embeddings=True, show_progress_bar=False)
    client.upsert(
        COLLECTION,
        points=[PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload=c) for i, c in enumerate(corpus)],
        wait=True,
    )
    bm25 = BM25Okapi([tokenize(c["content"]) for c in corpus])
    id_by_idx = [c["id"] for c in corpus]
    return client, bm25, id_by_idx


def search_dense(query, model, client, top_k=10):
    emb = model.encode(f"query: {query}", normalize_embeddings=True)
    hits = client.query_points(COLLECTION, query=emb.tolist(), limit=top_k).points
    return [h.id for h in hits]


def search_sparse(query, bm25, id_by_idx, top_k=10):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(id_by_idx)), key=lambda i: -scores[i])[:top_k]
    return [id_by_idx[i] for i in top]


def search_hybrid_rrf(query, model, client, bm25, id_by_idx, top_k=10, k=60):
    dense_top = search_dense(query, model, client, top_k=20)
    sparse_top = search_sparse(query, bm25, id_by_idx, top_k=20)
    scores: dict[int, float] = {}
    for ranking in [dense_top, sparse_top]:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved, relevant):
    for rank, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def benchmark(name, search_fn, golden):
    by_cat: dict[str, list] = {}
    latencies = []
    for case in golden:
        relevant = set(case["relevant"])
        cat = case["category"]
        t0 = time.perf_counter()
        retrieved = search_fn(case["q"])
        latencies.append((time.perf_counter() - t0) * 1000)
        by_cat.setdefault(cat, []).append({
            "r5": recall_at_k(retrieved, relevant, 5),
            "r10": recall_at_k(retrieved, relevant, 10),
            "rr": reciprocal_rank(retrieved, relevant),
        })

    overall_r5 = statistics.mean(v["r5"] for vals in by_cat.values() for v in vals)
    overall_r10 = statistics.mean(v["r10"] for vals in by_cat.values() for v in vals)
    overall_mrr = statistics.mean(v["rr"] for vals in by_cat.values() for v in vals)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]

    result = {
        "mode": name,
        "recall@5": round(overall_r5, 3),
        "recall@10": round(overall_r10, 3),
        "mrr": round(overall_mrr, 3),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
    }
    for cat, vals in by_cat.items():
        result[f"r5_{cat}"] = round(statistics.mean(v["r5"] for v in vals), 3)
    return result


def build_golden(corpus) -> list[dict]:
    """Golden set construit pour ce corpus support technique."""
    return [
        # Paraphrases (10) — avantage dense
        {"category": "paraphrase", "q": "Comment se connecter avec un mot de passe oublié ?", "relevant": [3, 4]},
        {"category": "paraphrase", "q": "Comment activer l'authentification renforcée ?", "relevant": [3]},
        {"category": "paraphrase", "q": "Politique relative aux fichiers de sauvegarde", "relevant": [32]},
        {"category": "paraphrase", "q": "Quelle est la stratégie d'archivage ?", "relevant": [31, 32]},
        {"category": "paraphrase", "q": "Combien de temps les logs sont-ils gardés ?", "relevant": [31]},
        {"category": "paraphrase", "q": "Comment je modifie mon plan d'abonnement ?", "relevant": [24]},
        {"category": "paraphrase", "q": "Mon entreprise est-elle conforme aux règles européennes ?", "relevant": [29]},
        {"category": "paraphrase", "q": "Qu'est-ce que la fédération d'identité ?", "relevant": [1, 5]},
        {"category": "paraphrase", "q": "Comment me débrouiller avec une mauvaise réception en mobilité ?", "relevant": [36]},
        {"category": "paraphrase", "q": "Apparence personnalisée de l'application", "relevant": [47]},

        # Références techniques (10) — avantage sparse
        {"category": "ref_tech", "q": "Erreur 503 sur /api/reports", "relevant": [8]},
        {"category": "ref_tech", "q": "Code HTTP 401 token expiré", "relevant": [6]},
        {"category": "ref_tech", "q": "X-Request-Id 502", "relevant": [10]},
        {"category": "ref_tech", "q": "Limite 100 req/min", "relevant": [9, 15]},
        {"category": "ref_tech", "q": "OAuth 2.0 authorization code flow", "relevant": [4]},
        {"category": "ref_tech", "q": "TLS 1.3 AES-256", "relevant": [30]},
        {"category": "ref_tech", "q": "Bind DN CN=svc_account", "relevant": [2]},
        {"category": "ref_tech", "q": "endpoint /metrics Prometheus", "relevant": [46]},
        {"category": "ref_tech", "q": "novacloud-cli ldap import-ca", "relevant": [41]},
        {"category": "ref_tech", "q": "vector cosine_ops HNSW", "relevant": []},  # absent : test négatif

        # Mixtes (10)
        {"category": "mixte", "q": "Configurer SAML 2.0 sur la version 3.2", "relevant": [1]},
        {"category": "mixte", "q": "Comment faire un export CSV via l'API ?", "relevant": [12]},
        {"category": "mixte", "q": "Activer les notifications Slack pour les alertes", "relevant": [38]},
        {"category": "mixte", "q": "Migration depuis NovaCloud 2.x vers 3.x", "relevant": [26]},
        {"category": "mixte", "q": "Récupérer mes factures PDF mensuelles", "relevant": [23]},
        {"category": "mixte", "q": "Suppression compte RGPD article 17", "relevant": [49, 50]},
        {"category": "mixte", "q": "Pagination cursor-based via paramètre after", "relevant": [14, 35]},
        {"category": "mixte", "q": "Webhook signature HMAC SHA-256", "relevant": [13]},
        {"category": "mixte", "q": "Chiffrement Customer-Managed Keys CMK", "relevant": [30]},
        {"category": "mixte", "q": "Notifications push iOS Android", "relevant": [37]},
    ]


def plot_results(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cats = ["paraphrase", "ref_tech", "mixte"]
    modes = df["mode"].tolist()
    width = 0.25
    x = range(len(cats))
    colors = ["#FF6B35", "#0066FF", "#1A1A1A"]

    for i, mode in enumerate(modes):
        row = df[df["mode"] == mode].iloc[0]
        values = [row.get(f"r5_{c}", 0) for c in cats]
        axes[0].bar([xi + i * width for xi in x], values, width, label=mode, color=colors[i])
    axes[0].set_xticks([xi + width for xi in x])
    axes[0].set_xticklabels(cats)
    axes[0].set_ylabel("Recall@5")
    axes[0].set_title("Recall@5 par catégorie")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    axes[1].bar(modes, df["p95_ms"], color=colors)
    axes[1].set_ylabel("Latence p95 (ms)")
    axes[1].set_title("Latence p95")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_plots.png", dpi=120)
    console.print("[green]Graphique sauvé : benchmark_plots.png[/green]")


def main() -> None:
    console.print("\n[bold orange3]Solution TP 3 — Benchmark dense vs sparse vs hybride[/bold orange3]\n")

    corpus = get_corpus()
    model = SentenceTransformer(MODEL_NAME)
    client, bm25, id_by_idx = setup(corpus, model)

    golden = build_golden(corpus)
    Path("golden_set.json").write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓ Golden set sauvé ({len(golden)} cas)[/green]\n")

    # Warmup
    for _ in range(3):
        model.encode("query: warmup", normalize_embeddings=True)

    results = [
        benchmark("dense", lambda q: search_dense(q, model, client), golden),
        benchmark("sparse", lambda q: search_sparse(q, bm25, id_by_idx), golden),
        benchmark("hybrid_rrf", lambda q: search_hybrid_rrf(q, model, client, bm25, id_by_idx), golden),
    ]
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)

    table = Table(title="Résultats benchmark", show_lines=True)
    for col in df.columns:
        table.add_column(col, justify="right" if col != "mode" else "left",
                         style="bold orange3" if col == "mode" else "")
    for _, row in df.iterrows():
        table.add_row(*[str(row[c]) for c in df.columns])
    console.print(table)

    plot_results(df)

    # Note de recommandation
    console.print("\n[bold]Note de recommandation type :[/bold]")
    console.print("• Sur paraphrases : dense > sparse, hybride ≈ dense")
    console.print("• Sur réf techniques : sparse > dense, hybride ≈ sparse (ou +)")
    console.print("• Sur mixtes : hybride > dense ET > sparse")
    console.print("• Latence : hybride ~+10-20% vs dense seul, acceptable")
    console.print("• → RECOMMANDATION : hybride RRF en production, monitoring continu via re-run hebdo\n")


if __name__ == "__main__":
    main()
