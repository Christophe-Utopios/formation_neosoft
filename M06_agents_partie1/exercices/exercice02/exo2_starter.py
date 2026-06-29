"""
TP 2 — Starter : routing dynamique avec conditional edges.

À COMPLÉTER. Cherchez les TODO.
"""
from __future__ import annotations

import os
from typing import List, Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

import _mock_insurance as ins

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


class State(TypedDict):
    question: str
    intent: Optional[str]   # "info", "quote", "out_of_scope"
    answer: Optional[str]
    needs_validation: bool
    path_taken: List[str]


def classify_intent(state: State) -> dict:
    """TODO 1 : appel LLM pour classifier la question dans 1 des 3 intents."""
    return {"intent": "info", "path_taken": state.get("path_taken", []) + ["classify"]}


def handle_info(state: State) -> dict:
    """TODO 2 : répondre via find_client et get_client_contracts."""
    return {
        "answer": "TODO",
        "path_taken": state["path_taken"] + ["handle_info"],
    }


def handle_quote(state: State) -> dict:
    """TODO 3 : créer un devis si possible, mettre needs_validation=True."""
    return {
        "answer": "TODO",
        "needs_validation": True,
        "path_taken": state["path_taken"] + ["handle_quote"],
    }


def handle_out_of_scope(state: State) -> dict:
    """TODO 4 : message d'escalade."""
    return {
        "answer": "Je n'ai pas l'information demandée. Je vous redirige vers un conseiller humain.",
        "path_taken": state["path_taken"] + ["handle_out_of_scope"],
    }


def route(state: State) -> str:
    """TODO 5 : retourner le nom du node suivant selon state['intent']."""
    return "handle_info"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("classify", classify_intent)
    graph.add_node("handle_info", handle_info)
    graph.add_node("handle_quote", handle_quote)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)

    # TODO 6 : ajouter les edges et conditional_edges
    graph.add_edge(START, "classify")
    # ...

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]TP 2 — Routing avec conditional edges[/bold orange3]\n")
    agent = build_graph()

    # Visualisation ASCII
    try:
        console.print(agent.get_graph().draw_ascii())
    except Exception as e:
        console.print(f"[dim]ASCII rendering: {e}[/dim]")

    questions = [
        "Quels sont les contrats de Jean Dupont ?",
        "Crée-moi un devis pour Sophie Martin sur l'Auto Tous Risques",
        "Vous travaillez le dimanche ?",
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Q : {q}[/bold cyan]")
        result = agent.invoke({
            "question": q, "needs_validation": False, "path_taken": [],
        })
        console.print(Panel(result.get("answer", ""), title="Réponse", border_style="orange3"))
        console.print(f"[dim]Chemin : {' → '.join(result['path_taken'])}[/dim]")
        console.print(f"[dim]needs_validation = {result['needs_validation']}[/dim]")


if __name__ == "__main__":
    main()
