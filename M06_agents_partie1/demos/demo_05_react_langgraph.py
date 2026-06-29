from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich.console import Console

import _fake_crm as crm

load_dotenv()
console = Console()
LLM_MODEL = "claude-sonnet-4-6"


# Déclaration des outils disponibles avec le décorateur @tool de LangChain
# Chaque docstring décrit l'outil pour le LLM

@tool
def find_tenant_by_name(name: str) -> dict:
    """Recherche un tenant client par son nom (exact ou contenant)."""
    t = crm.find_tenant_by_name(name)
    return t if t else {"error": f"Tenant '{name}' introuvable"}


@tool
def count_active_users(tenant_id: str) -> dict:
    """Compte les utilisateurs actifs d'un tenant donné par son ID."""
    return {"count": crm.count_active_users(tenant_id)}


@tool
def list_tickets(tenant_id: str, status: str = "open") -> list[dict]:
    """Liste les tickets d'un tenant filtrés par statut (open/closed)."""
    return crm.list_tickets(tenant_id, status)


@tool
def create_ticket(tenant_id: str, subject: str, priority: str = "med") -> dict:
    """Crée un nouveau ticket pour un tenant."""
    return crm.create_ticket(tenant_id, subject, priority)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 05 — Agent ReAct LangGraph[/bold orange3]\n")

    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    tools = [find_tenant_by_name, count_active_users, list_tickets, create_ticket]

    # create_react_agent : helper LangGraph qui crée automatiquement un graphe ReAct complet
    # (boucle raisonnement → action → observation → raisonnement)
    agent = create_react_agent(llm, tools=tools)

    question = ("Pour Acme Corp, donne-moi le plan, le nb d'utilisateurs actifs, "
                "et la liste des tickets ouverts avec leurs priorités. "
                "Ensuite crée un ticket de priorité haute pour 'Audit annuel SAML'.")

    console.print(f"[bold cyan]Question :[/bold cyan]\n{question}\n")

    # Streaming des étapes : permet de suivre en temps réel le raisonnement de l'agent
    console.print("[bold]Streaming des étapes :[/bold]\n")

    for chunk in agent.stream(
        {"messages": [("user", question)]},
        stream_mode="values",  # Retourne le State complet à chaque étape
    ):
        msg = chunk["messages"][-1]
        if msg.type == "ai":
            if msg.tool_calls:
                # L'agent a décidé d'appeler un outil
                for call in msg.tool_calls:
                    console.print(f"[yellow]🔧 {call['name']}({call['args']})[/yellow]")
            elif msg.content:
                # L'agent raisonne ou répond
                console.print(f"[orange3]🤖 {msg.content}[/orange3]")
        elif msg.type == "tool":
            # Résultat de l'exécution de l'outil
            content = str(msg.content)[:200]
            console.print(f"[dim]   ↩ {content}[/dim]")

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• create_react_agent encapsule la boucle ReAct standard")
    console.print("• Le streaming permet d'afficher les étapes en temps réel à l'utilisateur")
    console.print("• Chaque tool reçoit ses arguments validés via Pydantic\n")


if __name__ == "__main__":
    main()