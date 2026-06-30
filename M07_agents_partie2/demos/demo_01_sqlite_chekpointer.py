from __future__ import annotations

import os
import sqlite3
from typing import Annotated, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"
DB_PATH = "demo_01_checkpoints.db"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def llm_node(state: State) -> dict:
    """Appel LLM utilisant tout l'historique de messages."""
    llm = Anthropic()
    formatted = [{"role": "user" if isinstance(m, HumanMessage) else "assistant",
                  "content": m.content} for m in state["messages"]]
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=400, temperature=0,
        messages=formatted,
    )
    return {"messages": [AIMessage(content=msg.content[0].text)]}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("llm", llm_node)
    g.add_edge(START, "llm")
    g.add_edge("llm", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    # Nettoyer la DB si existe
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    console.print("\n[bold orange3]Demo 01 — Checkpointing SQLite[/bold orange3]\n")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    agent = build_graph(checkpointer)

    config_alice = {"configurable": {"thread_id": "user_alice"}}

    # Conversation alice
    console.print("[bold cyan]Alice — Tour 1[/bold cyan]")
    r1 = agent.invoke({"messages": [HumanMessage("Bonjour, je m'appelle Alice et je travaille chez Acme Corp.")]}, config=config_alice)
    console.print(Panel(r1["messages"][-1].content, border_style="orange3"))

    console.print("\n[bold cyan]Alice — Tour 2 (10 min plus tard, MÊME thread_id)[/bold cyan]")
    r2 = agent.invoke({"messages": [HumanMessage("Quel est mon prénom et où je travaille ?")]}, config=config_alice)
    console.print(Panel(r2["messages"][-1].content, border_style="orange3"))

    # Conversation bob (autre thread)
    config_bob = {"configurable": {"thread_id": "user_bob"}}
    console.print("\n[bold cyan]Bob — autre thread, isolation des contextes[/bold cyan]")
    r3 = agent.invoke({"messages": [HumanMessage("Quel est mon prénom ?")]}, config=config_bob)
    console.print(Panel(r3["messages"][-1].content, border_style="orange3"))

    # Lecture de l'historique
    console.print("\n[bold]Historique du thread alice :[/bold]")
    history = list(agent.get_state_history(config_alice))
    for snap in reversed(history):
        step = snap.metadata.get("step", "?")
        n_msgs = len(snap.values.get("messages", []))
        console.print(f"  Step {step} — {n_msgs} messages — next: {snap.next}")

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• Le thread_id isole les conversations entre utilisateurs")
    console.print("• L'historique des snapshots est consultable à tout moment")
    console.print(f"• Les checkpoints sont persistés dans {DB_PATH}\n")
    conn.close()


if __name__ == "__main__":
    main()
