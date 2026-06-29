from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich.console import Console

import _mock_insurance as ins

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


# TODO 1 : implémenter les 4 tools en exposant les fonctions de _mock_insurance.
# Bien rédiger les docstrings avec arguments typés.

@tool
def find_client(query: str) -> dict:
    """TODO."""
    return {}


@tool
def get_client_contracts(client_id: str) -> list:
    """TODO."""
    return []


@tool
def is_product_available(product: str, client_id: str) -> dict:
    """TODO."""
    return {}


@tool
def create_quote(client_id: str, product: str) -> dict:
    """TODO."""
    return {}


SYSTEM_PROMPT = """TODO : décrire le rôle de l'agent (commercial assurance),
ce qu'il peut faire (les 4 tools), et les règles à respecter (ne pas inventer
de tarifs, refuser poliment les demandes hors cadre)."""


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]TP 1 — Agent assurance[/bold orange3]\n")

    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    tools = [find_client, get_client_contracts, is_product_available, create_quote]

    # TODO 2 : passer le SYSTEM_PROMPT à create_react_agent
    agent = create_react_agent(llm, tools=tools)

    questions = [
        "Bonjour, peux-tu me dire qui est Jean Dupont et quels sont ses contrats ?",
        "Pour la cliente carole.dubois@example.com, prépare un devis pour le produit 'Multirisque Habitation'.",
        "Le produit 'Auto Premium' est-il dispo pour tous nos clients ?",
        "Crée un devis pour Maxime Lemoine sur le produit 'Santé Famille'.",
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Q : {q}[/bold cyan]")
        # TODO 3 : streamer les étapes (tool_calls, results, réponse finale)
        # avec config={"recursion_limit": 12}
        ...


if __name__ == "__main__":
    main()
