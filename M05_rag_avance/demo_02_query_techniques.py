from __future__ import annotations

import os
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from _setup import get_or_create_index, search_dense

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"

REWRITE_PROMPT = """Tu es un assistant qui reformule des questions utilisateurs
en requêtes de recherche claires et complètes.

Question utilisateur : {question}

Reformule en UNE question explicite incluant :
- L'intention claire
- Les termes techniques pertinents
- Le contexte implicite (produit NovaCloud) s'il peut être déduit

Réponds UNIQUEMENT avec la question reformulée, sans préambule."""

MULTI_QUERY_PROMPT = """Génère exactement 3 reformulations DIFFÉRENTES de cette question.
Chaque reformulation doit utiliser un vocabulaire différent mais garder le sens.
Une reformulation par ligne, sans numérotation, sans préambule.

Question : {question}"""

HYDE_PROMPT = """Réponds en 3-4 phrases à cette question, comme si tu connaissais
déjà la réponse. Ton factuel, style documentation technique. N'invente pas de chiffres.

Question : {question}

Réponse :"""


def rewrite_query(question: str, llm: Anthropic) -> str:
    """Query rewriting : reformulation de la question en termes clairs et techniques."""
    # Utilise le LLM pour reformuler en vocabulaire technique adapté au corpus
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=120, temperature=0,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
    )
    return msg.content[0].text.strip()


def multi_query(question: str, llm: Anthropic) -> list[str]:
    """Multi-query : génération de plusieurs paraphrases pour élargir la recherche."""
    # Génère 3 reformulations différentes pour couvrir plusieurs angles sémantiques
    # Temperature > 0 pour plus de diversité dans les paraphrases
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0.3,
        messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=question)}],
    )
    return [line.strip(" -•") for line in msg.content[0].text.strip().split("\n") if line.strip()]


def hyde(question: str, llm: Anthropic) -> str:
    """HyDE : génération d'une réponse hypothétique pour rechercher par similarité."""
    # Hypothetical Document Embedding : génère une fausse réponse et cherche par similarité
    # Plus efficace que chercher avec la question, car sémantiquement proche des docs
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
    )
    return msg.content[0].text.strip()


def show_hits(label: str, hits) -> None:
    table = Table(title=label, show_lines=False)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Catégorie", width=14)
    table.add_column("Titre", overflow="fold")
    for hit in hits[:5]:
        table.add_row(f"{hit.score:.3f}", hit.payload["category"], hit.payload["title"])
    console.print(table)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant.[/red]")
        return

    console.print("\n[bold orange3]Demo 02 — Techniques de transformation de requête[/bold orange3]\n")
    client, model, _ = get_or_create_index()
    llm = Anthropic()

    question = "ça marche pas, j'ai changé de tel"
    console.print(Panel(question, title="Question utilisateur originale", border_style="cyan"))

    # 1. Naïve (baseline)
    # Recherche directe sans transformation, pour comparaison
    hits_naive = search_dense(client, model, question, top_k=5)
    show_hits("Top-5 SANS transformation (baseline)", hits_naive)

    # 2. Query rewriting
    # Reformule la question ambiguë en requête technique claire
    rewritten = rewrite_query(question, llm)
    console.print(Panel(rewritten, title="Question reformulée (rewriting)", border_style="orange3"))
    hits_rw = search_dense(client, model, rewritten, top_k=5)
    show_hits("Top-5 avec query rewriting", hits_rw)

    # 3. Multi-query : génère plusieurs paraphrases et fusionne les résultats
    paraphrases = multi_query(question, llm)
    console.print("\n[bold]Paraphrases générées (multi-query) :[/bold]")
    for p in paraphrases:
        console.print(f"  • {p}")
    # Recherche avec chaque paraphrase, puis fusion par RRF (Reciprocal Rank Fusion)
    rankings: list[list[int]] = []
    for p in [question] + paraphrases:
        hits = search_dense(client, model, p, top_k=10)
        rankings.append([h.id for h in hits])
    # Fusion RRF : combine les rankings en un seul score
    # Formule : score(doc) = Σ 1/(k + rank), k=60 constante de lissage standard
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])[:5]
    title_by_id = {c["id"]: c for c in get_or_create_index()[2]}
    table = Table(title="Top-5 fusionné (multi-query + RRF)", show_lines=False)
    table.add_column("RRF", justify="right", width=8)
    table.add_column("Catégorie", width=14)
    table.add_column("Titre", overflow="fold")
    for doc_id, score in fused:
        c = title_by_id[doc_id]
        table.add_row(f"{score:.4f}", c["category"], c["title"])
    console.print(table)

    # 4. HyDE
    # Génère une réponse hypothétique, puis cherche des docs similaires à cette réponse
    # Principe : une réponse est sémantiquement plus proche des docs que la question
    hypothetical = hyde(question, llm)
    console.print(Panel(hypothetical, title="Réponse hypothétique (HyDE)", border_style="orange3"))
    hits_hyde = search_dense(client, model, hypothetical, top_k=5)
    show_hits("Top-5 avec HyDE (recherche par fausse réponse)", hits_hyde)

    console.print("\n[bold]Observation :[/bold]")
    console.print("• Sans transformation : retrieval pollué par des chunks hors-sujet")
    console.print("• Rewriting : remet la question dans le bon vocabulaire technique")
    console.print("• Multi-query : robuste si les utilisateurs varient leur formulation")
    console.print("• HyDE : très efficace pour les corpus techniques précis\n")


if __name__ == "__main__":
    main()
