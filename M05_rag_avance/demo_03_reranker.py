from __future__ import annotations

import time

from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.table import Table

from _setup import get_or_create_index, search_dense

console = Console()
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    console.print("\n[bold orange3]Demo 03 — Re-ranking cross-encoder[/bold orange3]\n")
    client, model, _ = get_or_create_index()

    console.print(f"[dim]Chargement du re-ranker {RERANKER_NAME}...[/dim]")
    reranker = CrossEncoder(RERANKER_NAME, max_length=512)

    queries = [
        "Comment configurer SAML 2.0 ?",
        "Erreur 503 endpoint /api/reports",
        "Procédure d'export RGPD pour un utilisateur",
    ]

    for query in queries:
        console.print(f"\n[bold cyan]Requête : {query}[/bold cyan]")

        # 1. Première passe : retrieval vectoriel rapide (bi-encoder)
        # Bi-encoder : encode query et docs séparément, rapide mais moins précis
        t0 = time.perf_counter()
        candidates = search_dense(client, model, query, top_k=20)
        t_retrieve = (time.perf_counter() - t0) * 1000

        # 2. Deuxième passe : re-ranking précis (cross-encoder)
        # Cross-encoder : évalue chaque paire (query, document) conjointement
        # Plus lent (doit traiter chaque paire) mais beaucoup plus précis
        pairs = [(query, c.payload["content"]) for c in candidates]
        t0 = time.perf_counter()
        rerank_scores = reranker.predict(pairs, show_progress_bar=False)
        t_rerank = (time.perf_counter() - t0) * 1000

        # Tri par score de reranking et sélection du top-5 final
        # Les documents peuvent changer de position significativement
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: -x[1])
        top5_after = ranked[:5]

        # Affichage comparatif
        table = Table(title=f"Avant vs Après re-ranking", show_lines=False)
        table.add_column("Rang")
        table.add_column("Avant (cosine)", overflow="fold")
        table.add_column("Après (rerank)", overflow="fold", style="bold orange3")

        for rank in range(5):
            before = candidates[rank]
            after, score = top5_after[rank]
            same = "  " if before.id == after.id else "→ "
            table.add_row(
                str(rank + 1),
                f"[{before.score:.3f}] {before.payload['title']}",
                f"{same}[{score:.3f}] {after.payload['title']}",
            )
        console.print(table)
        console.print(f"[dim]Retrieval : {t_retrieve:.1f} ms · Rerank top-20 : {t_rerank:.1f} ms[/dim]")

    console.print("\n[bold]Observation :[/bold]")
    console.print("• Le rerank repositionne souvent un chunk plus précis en tête")
    console.print("• La latence ajoutée est de 100-300 ms selon CPU/GPU")
    console.print("• En prod : top-20 retrieval → top-5 rerank est le pattern recommandé\n")


if __name__ == "__main__":
    main()
