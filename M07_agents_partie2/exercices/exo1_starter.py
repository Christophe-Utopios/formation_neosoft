from __future__ import annotations

import os
import sqlite3
from typing import Annotated, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"
DB_PATH = "conversations.db"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def llm_node(state: State) -> dict:
    llm = Anthropic()
    formatted = [{"role": "user" if isinstance(m, HumanMessage) else "assistant",
                  "content": m.content} for m in state["messages"]]
    msg = llm.messages.create(model=LLM_MODEL, max_tokens=400, temperature=0,
                              system="Tu es un assistant juridique pour cabinet d'avocats.",
                              messages=formatted)
    return {"messages": [AIMessage(content=msg.content[0].text)]}


def build_graph(checkpointer):
    """TODO 1 : construire le graphe et le compiler AVEC le checkpointer."""
    g = StateGraph(State)
    g.add_node("llm", llm_node)
    g.add_edge(START, "llm")
    g.add_edge("llm", END)
    # TODO : g.compile(checkpointer=...)
    return g.compile()


def chat(agent, user_id: str, message: str) -> str:
    """TODO 2 : utiliser thread_id=f'user_{user_id}' dans la config."""
    result = agent.invoke({"messages": [HumanMessage(message)]})  # incomplete
    return result["messages"][-1].content


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    # TODO 3 : créer le SqliteSaver(...) et passer à build_graph
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    agent = build_graph(checkpointer=None)  # TODO

    # Scénario à exécuter (alice + bob alternés)
    turns = [
        ("alice", "Je m'appelle Alice, je travaille sur le dossier Martin."),
        ("bob", "Bonjour, je m'occupe du contentieux Société X."),
        ("alice", "Rappelle-moi le nom du dossier que je traite."),
        ("bob", "Quel client j'évoquais ?"),
        ("alice", "Récupère le 1er paragraphe de notre discussion."),
    ]

    for user, msg in turns:
        console.print(f"\n[bold cyan]{user} ➜ {msg}[/bold cyan]")
        answer = chat(agent, user, msg)
        console.print(Panel(answer, border_style="orange3"))

    # TODO 4 : afficher l'historique des snapshots du thread alice


if __name__ == "__main__":
    main()
