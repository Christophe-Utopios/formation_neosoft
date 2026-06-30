from __future__ import annotations

import os
import sqlite3
from typing import Annotated, Optional, TypedDict

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
DB_PATH = "demo_02_timetravel.db"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]
    answer: Optional[str]


def classify(state: State) -> dict:
    llm = Anthropic()
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=20, temperature=0,
        messages=[{"role": "user", "content": f"Classe en 'tech' ou 'billing' (un mot) : {state['messages'][-1].content}"}],
    )
    intent = msg.content[0].text.strip().lower()
    return {"intent": "tech" if "tech" in intent else "billing"}


def respond(state: State) -> dict:
    llm = Anthropic()
    persona = "ingénieur support technique" if state["intent"] == "tech" else "comptable client"
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Tu es {persona}. Réponds : {state['messages'][-1].content}"}],
    )
    answer = msg.content[0].text.strip()
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    console.print("\n[bold orange3]Demo 02 — Time-travel debugging[/bold orange3]\n")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("respond", respond)
    g.add_edge(START, "classify")
    g.add_edge("classify", "respond")
    g.add_edge("respond", END)
    agent = g.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "tt_demo"}}
    user_question = "Mon dernier prélèvement est en double, c'est inadmissible."

    console.print(f"[bold cyan]Question :[/bold cyan] {user_question}\n")

    # 1. Exécution initiale
    result = agent.invoke({"messages": [HumanMessage(user_question)]}, config=config)
    console.print(f"[orange3]Intent classifié :[/orange3] {result['intent']}")
    console.print(Panel(result["answer"], title="Réponse INITIALE", border_style="orange3"))

    # 2. Inspecter l'historique
    history = list(agent.get_state_history(config))
    console.print(f"\n[bold]Historique : {len(history)} snapshots[/bold]")
    for i, snap in enumerate(history):
        console.print(f"  [{i}] step={snap.metadata.get('step', '?')} next={snap.next} intent={snap.values.get('intent')}")

    # 3. Trouver le snapshot juste avant 'respond' (donc après classify)
    target = None
    for snap in history:
        if snap.next == ("respond",):
            target = snap
            break

    if not target:
        console.print("[red]Snapshot cible introuvable[/red]")
        return

    console.print(f"\n[cyan]→ On modifie le snapshot 'avant respond' : intent {target.values['intent']} → 'billing'[/cyan]")

    # 4. Patcher l'intent (simuler une correction humaine)
    new_config = agent.update_state(target.config, {"intent": "billing"})

    # 5. Rejouer depuis ce point
    console.print("[cyan]→ Replay avec intent corrigé[/cyan]\n")
    replayed = agent.invoke(None, config=new_config)
    console.print(Panel(replayed["answer"], title="Réponse APRÈS correction", border_style="green"))

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• On a corrigé l'intent SANS relancer toute la chaîne")
    console.print("• `update_state(config, patch)` retourne un nouveau config qui pointe le snapshot patché")
    console.print("• Très utile pour le QA et la reproduction de bugs\n")
    conn.close()


if __name__ == "__main__":
    main()
