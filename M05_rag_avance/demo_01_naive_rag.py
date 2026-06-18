from __future__ import annotations

import os
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from _setup import get_or_create_index, search_dense

load_dotenv()
console = Console()

LLM_MODEL = "claude-haiku-4-5-20251001"

NAIVE_PROMPT = """Tu es un assistant support. Réponds à la question de l'utilisateur
en t'appuyant sur le contexte fourni.

Contexte :
{context}

Question : {question}

Réponse :"""


def naive_rag(client, model, llm: Anthropic, question: str, top_k: int = 5) -> dict:
    """Pipeline naïve : retrieval direct → LLM (sans optimisation)."""
    # 1. Recherche vectorielle simple sur la question brute
    # Aucune transformation de la requête, recherche directe
    hits = search_dense(client, model, question, top_k=top_k)

    # 2. Concaténation du contexte récupéré
    # Tous les chunks sont concaténés tels quels, sans filtrage ni réorganisation
    context = "\n\n---\n\n".join(h.payload["content"] for h in hits)

    # 3. Construction du prompt et appel au LLM
    # Prompt minimal : contexte + question, sans instruction avancée
    prompt = NAIVE_PROMPT.format(context=context, question=question)
    msg = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=400,
        temperature=0,  # Déterministe pour reproductibilité
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "question": question,
        "answer": msg.content[0].text.strip(),
        "context": context,
        "hits": hits,
    }


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant.[/red]")
        return

    console.print("\n[bold orange3]Demo 01 — Naïve RAG[/bold orange3]\n")
    # Setup : récupère le client Qdrant, le modèle d'embedding et le corpus
    client, model, _ = get_or_create_index()
    llm = Anthropic()

    # Deux cas tests pour montrer les limites du RAG naïf
    cases = [
        ("Cas A — Question explicite (devrait bien marcher)",
         "Comment configurer SAML 2.0 sur la version 3.2 de NovaCloud ?"),
        ("Cas B — Question ambiguë (le naïve RAG va galérer)",
         "ça marche pas, j'ai changé de tel"),  # Contexte manquant, vocabulaire vague
    ]

    for label, question in cases:
        console.print(f"\n[bold cyan]{label}[/bold cyan]")
        # Exécution du pipeline RAG naïf complet
        result = naive_rag(client, model, llm, question, top_k=5)

        console.print(Panel(question, title="Question", border_style="cyan"))

        # Affichage des top-3 chunks récupérés pour observer la qualité du retrieval
        console.print("[bold]Top-3 chunks récupérés :[/bold]")
        for hit in result["hits"][:3]:
            console.print(f"  [{hit.score:.3f}] {hit.payload['title']} (cat={hit.payload['category']})")

        console.print(Panel(result["answer"], title="Réponse", border_style="orange3"))

    console.print("\n[bold]Observation :[/bold]")
    console.print("• Cas A : le top-3 est cohérent, la réponse est précise")
    console.print("• Cas B : le retrieval est aléatoire car la question est trop floue")
    console.print("• → On va voir comment query rewriting transforme le cas B (demo 02)\n")


if __name__ == "__main__":
    main()
