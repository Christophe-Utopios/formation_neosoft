from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

sys.path.insert(0, str(Path(__file__).parent.parent / "exercises"))
import _mock_hr as hr

load_dotenv()
console = Console()
DB_PATH = "exo2_hil.db"


class State(TypedDict):
    user_request: str
    employee: Optional[dict]
    pending_action: Optional[dict]
    approval: Optional[str]
    result: Optional[str]


def find_employee(state: State) -> dict:
    # Recherche naïve d'un nom dans la requête (2 mots capitalisés successifs)
    import re
    m = re.search(r"\b([A-ZÀ-Ý][a-zà-ÿ]+\s+[A-ZÀ-Ý][a-zà-ÿ]+)\b", state["user_request"])
    if not m:
        return {"employee": None}
    return {"employee": hr.find_employee(m.group(0))}


def draft_deletion(state: State) -> dict:
    emp = state.get("employee")
    if not emp:
        return {"pending_action": None}
    # Heuristique pour la raison : extraire ce qui suit "," ou "raison" dans la requête
    raw = state["user_request"]
    reason = raw.split(",", 1)[1].strip() if "," in raw else "non précisé"
    return {
        "pending_action": {
            "tool": "delete_account",
            "args": {"employee_id": emp["id"], "reason": reason},
            "target_name": emp["name"],
        },
    }


def execute_deletion(state: State) -> dict:
    if state.get("approval") == "rejected" or not state.get("pending_action"):
        return {"result": "Suppression annulée."}
    action = state["pending_action"]
    res = hr.delete_account(action["args"]["employee_id"], action["args"]["reason"])
    return {"result": str(res)}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("find", find_employee)
    g.add_node("draft", draft_deletion)
    g.add_node("execute", execute_deletion)
    g.add_edge(START, "find")
    g.add_edge("find", "draft")
    g.add_edge("draft", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["execute"])


def run_scenario(request: str, decision: str, modify_reason: str = "") -> None:
    """Lance le scénario complet avec une décision donnée."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    agent = build_graph(SqliteSaver(conn))
    config = {"configurable": {"thread_id": f"hil_{decision}"}}

    state = agent.invoke({
        "user_request": request, "employee": None, "pending_action": None,
        "approval": None, "result": None,
    }, config=config)

    pending = state.get("pending_action")
    if not pending:
        console.print("[yellow]Aucune action proposée.[/yellow]")
        conn.close()
        return

    console.print(Panel(
        f"Tool : {pending['tool']}\nArgs : {pending['args']}\nCible : {pending['target_name']}",
        title=f"⏸ Décision = {decision}", border_style="yellow",
    ))

    if decision == "r":
        agent.update_state(config, {"approval": "rejected"})
    elif decision == "m":
        modified = {**pending, "args": {**pending["args"], "reason": modify_reason}}
        agent.update_state(config, {"pending_action": modified, "approval": "modified"})
    else:
        agent.update_state(config, {"approval": "approved"})

    final = agent.invoke(None, config=config)
    console.print(Panel(final["result"], title=f"Résultat ({decision})", border_style="orange3"))
    # État résiduel
    emp_after = next((e for e in hr.EMPLOYEES.values() if e["name"] == pending["target_name"]), None)
    console.print(f"[dim]État employé : status={emp_after.get('status')}, reason={emp_after.get('deletion_reason', '—')}[/dim]")
    conn.close()


def main() -> None:
    console.print("\n[bold orange3]Solution TP 2 — HIL (3 scénarios)[/bold orange3]\n")

    auto = os.environ.get("DEMO_AUTO")
    request = "Supprime le compte de Maxime Lemoine, départ entreprise convenu"

    if auto in {"a", "r", "m"}:
        run_scenario(request, auto, modify_reason="audit conformité 2026")
        return

    # Scénarios complets (réinitialise hr.EMPLOYEES entre chaque)
    original = dict(hr.EMPLOYEES["e_1"])
    for d, reason in [("a", ""), ("r", ""), ("m", "audit conformité 2026")]:
        hr.EMPLOYEES["e_1"] = dict(original)  # reset
        console.print(f"\n[bold cyan]── Scénario : {d} ──[/bold cyan]")
        run_scenario(request, d, modify_reason=reason)


if __name__ == "__main__":
    main()
