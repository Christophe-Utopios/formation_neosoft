from __future__ import annotations

import os
import sqlite3
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import _setup  # noqa: F401
from _setup import crm  # noqa

load_dotenv()
console = Console()
DB_PATH = "demo_03_hil.db"


class State(TypedDict):
    user_request: str
    draft_action: Optional[dict]
    result: Optional[str]
    cancelled: bool


def draft_action(state: State) -> dict:
    """Prépare une action sensible à partir de la requête utilisateur."""
    request = state["user_request"]
    # Dans la vraie vie : LLM analyse la requête.
    # Ici on simplifie et on construit une action de suppression user.
    if "supprime" in request.lower() or "delete" in request.lower():
        # Extraire un email simple
        import re
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", request)
        email = m.group(0) if m else None
        target = next((u for u in crm.USERS.values() if u["email"] == email), None)
        if target:
            return {
                "draft_action": {
                    "tool": "deactivate_user",
                    "args": {"user_id": target["id"], "email": target["email"]},
                    "reason": "demande utilisateur (RGPD)",
                },
            }
    return {"draft_action": None}


def execute_action(state: State) -> dict:
    """Exécute l'action après approbation humaine."""
    if state.get("cancelled"):
        return {"result": "Action annulée par l'humain."}
    action = state.get("draft_action")
    if not action:
        return {"result": "Aucune action proposée."}

    if action["tool"] == "deactivate_user":
        uid = action["args"]["user_id"]
        if uid in crm.USERS:
            crm.USERS[uid]["active"] = False
            return {"result": f"User {action['args']['email']} désactivé."}
    return {"result": "Tool inconnu, aucune action."}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("draft", draft_action)
    g.add_node("execute", execute_action)
    g.add_edge(START, "draft")
    g.add_edge("draft", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["execute"])


def main() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    console.print("\n[bold orange3]Demo 03 — Human-in-the-Loop[/bold orange3]\n")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    agent = build_graph(checkpointer)

    config = {"configurable": {"thread_id": "hil_session_1"}}
    request = "Supprime le compte de bob@acme.com, demande RGPD"
    console.print(Panel(request, title="Requête utilisateur", border_style="cyan"))

    # 1. Lancer — s'arrête avant execute
    state_after_draft = agent.invoke({"user_request": request, "cancelled": False}, config=config)

    proposed = state_after_draft.get("draft_action")
    if not proposed:
        console.print("[yellow]Aucune action proposée[/yellow]")
        return

    console.print(Panel(
        f"Tool : {proposed['tool']}\n"
        f"Args : {proposed['args']}\n"
        f"Raison : {proposed['reason']}",
        title="⏸ Action en attente d'approbation",
        border_style="yellow",
    ))

    # 2. Demander approbation (DEMO_AUTO=a|r|m permet de scripter en CI / démo enregistrée)
    auto = os.environ.get("DEMO_AUTO")
    if auto in {"a", "r", "m"}:
        decision = auto
        console.print(f"[dim](mode auto via DEMO_AUTO={auto})[/dim]")
    else:
        decision = Prompt.ask(
            "Approuver (a), Rejeter (r), Modifier la raison (m) ?",
            choices=["a", "r", "m"], default="a",
        )

    if decision == "r":
        agent.update_state(config, {"cancelled": True})
        console.print("[red]Action rejetée par l'humain.[/red]")
    elif decision == "m":
        new_reason = Prompt.ask("Nouvelle raison ?", default="audit conformité")
        modified = {**proposed, "reason": new_reason}
        agent.update_state(config, {"draft_action": modified})
        console.print(f"[blue]Raison modifiée → '{new_reason}'[/blue]")
    else:
        console.print("[green]Action approuvée.[/green]")

    # 3. Reprendre l'exécution
    final = agent.invoke(None, config=config)
    console.print(Panel(final["result"], title="Résultat final", border_style="orange3"))

    console.print(f"\n[bold]État final user bob@acme.com :[/bold] active = {crm.USERS['u_2']['active']}")
    console.print("\n[bold]À retenir :[/bold]")
    console.print("• `interrupt_before=['execute']` suspend avant ce node")
    console.print("• `update_state` modifie l'état avant la reprise")
    console.print("• `invoke(None, config)` reprend l'exécution là où elle s'est arrêtée\n")
    conn.close()


if __name__ == "__main__":
    main()
