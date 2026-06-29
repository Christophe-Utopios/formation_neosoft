from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich.console import Console

# Importer le mock depuis exercises/
sys.path.insert(0, str(Path(__file__).parent.parent / "exercises"))
import _mock_insurance as ins

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


@tool
def find_client(query: str) -> dict:
    """Recherche un client par nom (partiel) ou email exact.
    Retourne ses informations (id, name, email, age, city, risk_score) ou un message d'erreur.

    Args:
        query: nom partiel ou email exact du client à chercher
    """
    c = ins.find_client(query)
    return c if c else {"error": f"Client '{query}' introuvable"}


@tool
def get_client_contracts(client_id: str) -> list:
    """Liste les contrats actifs et expirés d'un client par son ID.

    Args:
        client_id: identifiant interne du client (ex: 'c_1')
    """
    return ins.get_client_contracts(client_id)


@tool
def is_product_available(product: str, client_id: str) -> dict:
    """Vérifie si un produit d'assurance est disponible pour un client donné,
    selon ses critères d'âge et de risque.

    Args:
        product: nom exact du produit (ex: 'Auto Premium', 'Multirisque Habitation')
        client_id: identifiant du client (ex: 'c_1')
    """
    return ins.is_product_available(product, client_id)


@tool
def create_quote(client_id: str, product: str) -> dict:
    """Crée un devis pour un client sur un produit donné.
    Vérifie d'abord la disponibilité, retourne le devis ou une raison d'échec.

    Args:
        client_id: identifiant du client
        product: nom exact du produit
    """
    return ins.create_quote(client_id, product)


SYSTEM_PROMPT = """Tu es l'assistant des commerciaux d'un courtier en assurance.

Tu peux utiliser les tools suivants :
- find_client : trouver un client par nom ou email
- get_client_contracts : lister ses contrats
- is_product_available : vérifier qu'un produit est disponible pour ce client
- create_quote : créer un devis (uniquement si disponible)

Règles strictes :
1. N'invente JAMAIS de tarif. Utilise toujours create_quote pour obtenir un montant officiel.
2. Avant de créer un devis, vérifie toujours la disponibilité.
3. Si la demande sort du périmètre (ex : météo, juridique, RH), redirige vers un conseiller humain.
4. Si un client est introuvable, dis-le explicitement et propose les noms similaires.
5. Présente les résultats dans un format clair (tableaux markdown si pertinent)."""


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Solution TP 1 — Agent assurance[/bold orange3]\n")

    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    tools = [find_client, get_client_contracts, is_product_available, create_quote]

    agent = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)

    questions = [
        "Bonjour, peux-tu me dire qui est Jean Dupont et quels sont ses contrats ?",
        "Pour la cliente carole.dubois@example.com, prépare un devis pour le produit 'Multirisque Habitation'.",
        "Le produit 'Auto Premium' est-il dispo pour tous nos clients ?",
        "Crée un devis pour Maxime Lemoine sur le produit 'Santé Famille'.",
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Q : {q}[/bold cyan]")
        for chunk in agent.stream(
            {"messages": [("user", q)]},
            stream_mode="values",
            config={"recursion_limit": 12},
        ):
            msg = chunk["messages"][-1]
            if msg.type == "ai":
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        console.print(f"[yellow]🔧 {call['name']}({call['args']})[/yellow]")
                elif msg.content:
                    console.print(f"[orange3]🤖 {msg.content[:500]}...[/orange3]"
                                  if len(str(msg.content)) > 500 else f"[orange3]🤖 {msg.content}[/orange3]")
            elif msg.type == "tool":
                content_str = str(msg.content)[:150]
                console.print(f"[dim]   ↩ {content_str}[/dim]")


if __name__ == "__main__":
    main()
