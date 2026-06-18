from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from _setup import get_or_create_index

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

REWRITE_PROMPT = """Reformule cette question utilisateur en UNE requête de recherche claire,
incluant termes techniques pertinents. Réponds uniquement avec la question reformulée.

Question : {question}"""

GENERATE_PROMPT = """Tu es un assistant support technique pour NovaCloud.

Réponds en t'appuyant UNIQUEMENT sur le contexte ci-dessous.
Pour chaque affirmation, cite la source entre crochets : [chunk_id].
Si l'info n'est pas dans le contexte, dis-le explicitement.

Contexte :
{context}

Question : {question}

Réponse (avec citations [chunk_id]) :"""


@dataclass
class StageTiming:
    name: str
    duration_ms: float


def tokenize(text: str) -> list[str]:
    """Tokenisation simple pour BM25 : split sur non-alphanumériques."""
    return [t for t in re.split(r"\W+", text.lower()) if t]


def rrf_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion : fusionne plusieurs rankings en un seul.

    Formule RRF : score(doc) = Σ 1/(k + rank_i) pour chaque ranking i
    k=60 est une constante de lissage standard recommandée par l'article RRF original.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def full_pipeline(question: str, client, model, reranker, bm25, id_by_idx,
                  corpus, llm: Anthropic) -> tuple[dict, list[StageTiming]]:
    """Pipeline RAG complet : rewriting → hybrid → rerank → génération."""
    timings: list[StageTiming] = []

    # 1. Query rewriting : reformulation pour améliorer le retrieval
    # Transforme une question vague en requête technique précise
    t0 = time.perf_counter()
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=120, temperature=0,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
    )
    rewritten = msg.content[0].text.strip()
    timings.append(StageTiming("1. Query rewriting", (time.perf_counter() - t0) * 1000))

    # 2. Hybrid retrieval : dense + sparse avec fusion RRF (top-20 candidats)
    # Dense (vectoriel) : capture la similarité sémantique
    # Sparse (BM25) : capture les mots-clés exacts
    # RRF fusionne les deux rankings pour robustesse
    t0 = time.perf_counter()
    emb = model.encode(f"query: {rewritten}", normalize_embeddings=True)
    dense_hits = client.query_points(
        "m5_rag_demos", query=emb.tolist(), limit=20, with_payload=True,
    ).points
    dense_ranking = [h.id for h in dense_hits]
    sparse_scores = bm25.get_scores(tokenize(rewritten))
    sparse_top = sorted(range(len(id_by_idx)), key=lambda i: -sparse_scores[i])[:20]
    sparse_ranking = [id_by_idx[i] for i in sparse_top]
    fused_ids = rrf_fusion([dense_ranking, sparse_ranking])[:20]
    fused_chunks = [c for cid in fused_ids for c in corpus if c["id"] == cid][:20]
    timings.append(StageTiming("2. Hybrid retrieval (top-20)", (time.perf_counter() - t0) * 1000))

    # 3. Re-ranking : affine la sélection avec cross-encoder → top-5 finaux
    # Cross-encoder évalue précisément chaque paire (query, chunk)
    t0 = time.perf_counter()
    pairs = [(rewritten, c["content"]) for c in fused_chunks]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(fused_chunks, rerank_scores), key=lambda x: -x[1])
    top5 = [c for c, _ in ranked[:5]]
    timings.append(StageTiming("3. Re-ranking", (time.perf_counter() - t0) * 1000))

    # 4. Génération : LLM produit la réponse avec citations des sources
    # Format [chunk_id] permet la traçabilité et vérification des sources
    t0 = time.perf_counter()
    context = "\n\n".join(f"[chunk_{c['id']}] {c['content']}" for c in top5)
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": GENERATE_PROMPT.format(context=context, question=question)}],
    )
    answer = msg.content[0].text.strip()
    timings.append(StageTiming("4. Génération", (time.perf_counter() - t0) * 1000))

    return {
        "question_original": question,
        "question_rewritten": rewritten,
        "top5_chunks": top5,
        "answer": answer,
    }, timings


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant.[/red]")
        return

    console.print("\n[bold orange3]Demo 05 — Pipeline RAG complet[/bold orange3]\n")
    client, model, corpus = get_or_create_index()
    llm = Anthropic()

    console.print(f"[dim]Chargement du re-ranker {RERANKER_NAME}...[/dim]")
    reranker = CrossEncoder(RERANKER_NAME, max_length=512)
    bm25 = BM25Okapi([tokenize(c["content"]) for c in corpus])
    id_by_idx = [c["id"] for c in corpus]

    questions = [
        "ça marche pas, j'ai changé de tel",
        "Comment configurer SAML 2.0 sur la version 3.2 ?",
    ]

    for question in questions:
        console.print(f"\n[bold cyan]Question : {question}[/bold cyan]")
        result, timings = full_pipeline(question, client, model, reranker, bm25, id_by_idx, corpus, llm)

        if result["question_rewritten"] != question:
            console.print(f"[orange3]→ Reformulée : {result['question_rewritten']}[/orange3]")

        console.print("\n[bold]Top-5 chunks après pipeline complet :[/bold]")
        for c in result["top5_chunks"]:
            console.print(f"  [chunk_{c['id']}] {c['title']} (cat={c['category']})")

        console.print(Panel(result["answer"], title="Réponse finale", border_style="orange3"))

        # Tableau timings
        table = Table(title="Latence par étape", show_lines=False)
        table.add_column("Étape", style="cyan")
        table.add_column("Latence (ms)", justify="right")
        total = 0.0
        for t in timings:
            table.add_row(t.name, f"{t.duration_ms:.0f}")
            total += t.duration_ms
        table.add_row("[bold]TOTAL[/bold]", f"[bold orange3]{total:.0f}[/bold orange3]")
        console.print(table)

    console.print("\n[bold]Observation :[/bold]")
    console.print("• Pipeline complet ~1.5-3s sur Haiku, dominé par le LLM (rewriting + génération)")
    console.print("• Le retrieval+rerank reste sous 200 ms")
    console.print("• Les citations [chunk_id] permettent l'audit et la confiance utilisateur\n")


if __name__ == "__main__":
    main()
