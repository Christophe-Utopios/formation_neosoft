from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

console = Console()


def append_history(left: list, right: list) -> list:
    """Reducer : append à la liste plutôt que replace (utilisé pour history)."""
    return (left or []) + (right or [])


class State(TypedDict):
    iteration: int
    last_action: Optional[str]
    history: Annotated[List[str], append_history]
    escalated: bool
    answer: Optional[str]


MAX_ITERATIONS = 10
LOOP_DETECTION_WINDOW = 3


def agent_step(state: State) -> dict:
    """Simule un agent qui appelle toujours le même tool (cas pathologique)."""
    iteration = state["iteration"] + 1
    # Bug simulé : appel répétitif
    action = "find_user(id=u_42)"
    return {
        "iteration": iteration,
        "last_action": action,
        "history": [action],
    }


def safety_check(state: State) -> dict:
    """Détecte les boucles et la limite d'itérations."""
    if state["iteration"] >= MAX_ITERATIONS:
        return {
            "escalated": True,
            "answer": f"Max iterations ({MAX_ITERATIONS}) atteint. Escalade.",
        }

    history = state.get("history", [])
    if len(history) >= LOOP_DETECTION_WINDOW:
        last_n = history[-LOOP_DETECTION_WINDOW:]
        if all(a == last_n[0] for a in last_n):
            return {
                "escalated": True,
                "answer": f"Boucle détectée : '{last_n[0]}' appelé {LOOP_DETECTION_WINDOW} fois. Escalade.",
            }

    return {}


def should_continue(state: State) -> str:
    if state.get("escalated"):
        return "end"
    return "continue"


def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent_step)
    g.add_node("safety", safety_check)
    g.add_edge(START, "agent")
    g.add_edge("agent", "safety")
    g.add_conditional_edges("safety", should_continue, {"continue": "agent", "end": END})
    return g.compile()


def main() -> None:
    console.print("\n[bold orange3]Demo 04 — Détection de boucle[/bold orange3]\n")
    agent = build_graph()

    initial = {
        "iteration": 0, "last_action": None, "history": [],
        "escalated": False, "answer": None,
    }

    # Limite de récursion généreuse pour tester la safety
    result = agent.invoke(initial, config={"recursion_limit": 50})

    console.print(f"[bold cyan]Itérations exécutées :[/bold cyan] {result['iteration']}")
    console.print(f"[bold cyan]Historique :[/bold cyan] {len(result['history'])} actions")
    if result.get("escalated"):
        console.print(Panel(result["answer"], title="🚨 Escalade", border_style="red"))
    else:
        console.print(Panel("Pas d'escalade", title="État final", border_style="orange3"))

    console.print("\n[bold]À retenir :[/bold]")
    console.print(f"• Boucle détectée après {LOOP_DETECTION_WINDOW} appels identiques d'affilée")
    console.print(f"• Max iterations = {MAX_ITERATIONS} en garde-fou supplémentaire")
    console.print("• L'escalade est logguée dans le state, pas une exception → pattern testable\n")


if __name__ == "__main__":
    main()
