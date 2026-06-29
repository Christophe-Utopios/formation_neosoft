import os
from typing import Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

import _fake_crm as crm

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


# State partagé : structure de données passée entre tous les nodes du graphe
# NB : Optional[X] (et non `X | None`) pour compat Python 3.9 lors de
# l'évaluation des type hints par LangGraph.
class AgentState(TypedDict):
    question: str
    tenant_name: Optional[str]
    tenant_data: Optional[dict]
    n_users: Optional[int]
    answer: Optional[str]


def extract_tenant_name(state: AgentState) -> dict:
    """
    Node 1 : extraire le nom du tenant depuis la question.
    Retourne un dict partiel qui met à jour le State.
    """
    llm = Anthropic()
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=50, temperature=0,
        messages=[{"role": "user", "content": f"Extrais le nom du tenant. Réponds avec uniquement le nom.\n\n{state['question']}"}],
    )
    name = msg.content[0].text.strip()
    console.print(f"[dim]→ extract_tenant_name : '{name}'[/dim]")
    return {"tenant_name": name}  # Mise à jour partielle du State


def fetch_tenant_data(state: AgentState) -> dict:
    """
    Node 2 : appeler le CRM pour récupérer données + nb users.
    Utilise tenant_name du State pour faire les appels API.
    """
    if not state.get("tenant_name"):
        return {"tenant_data": None, "n_users": 0}
    tenant = crm.find_tenant_by_name(state["tenant_name"])
    if not tenant:
        console.print(f"[dim]→ fetch_tenant_data : tenant introuvable[/dim]")
        return {"tenant_data": None, "n_users": 0}
    n = crm.count_active_users(tenant["id"])
    console.print(f"[dim]→ fetch_tenant_data : {tenant['name']}, {n} users actifs[/dim]")
    return {"tenant_data": tenant, "n_users": n}


def generate_answer(state: AgentState) -> dict:
    """
    Node 3 : générer la réponse finale.
    Synthétise toutes les données du State en réponse utilisateur.
    """
    llm = Anthropic()
    if not state.get("tenant_data"):
        return {"answer": f"Je n'ai pas trouvé le tenant '{state.get('tenant_name', 'inconnu')}'."}

    context = f"Tenant : {state['tenant_data']}\nUtilisateurs actifs : {state['n_users']}"
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Question : {state['question']}\nContexte : {context}\n\nRéponse concise :"}],
    )
    answer = msg.content[0].text.strip()
    console.print(f"[dim]→ generate_answer : {len(answer)} chars[/dim]")
    return {"answer": answer}


def build_graph():
    """
    Construction du graphe LangGraph : définit les nodes et les edges (flux linéaire).
    """
    graph = StateGraph(AgentState)
    # Ajout des 3 nodes de traitement
    graph.add_node("extract", extract_tenant_name)
    graph.add_node("fetch", fetch_tenant_data)
    graph.add_node("generate", generate_answer)

    # Définition du flux linéaire : START → extract → fetch → generate → END
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "fetch")
    graph.add_edge("fetch", "generate")
    graph.add_edge("generate", END)

    return graph.compile()  # Compilation du graphe pour exécution


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 04 — Premier graphe LangGraph[/bold orange3]\n")
    agent = build_graph()

    # Visualisation ASCII du graphe pour comprendre la structure
    console.print("[bold]Graphe ASCII :[/bold]")
    try:
        console.print(agent.get_graph().draw_ascii())
    except Exception as e:
        console.print(f"[dim]ASCII rendering: {e}[/dim]")
        # Fallback : afficher structure
        for node in agent.get_graph().nodes:
            console.print(f"  • {node}")

    # Exécution
    questions = [
        "Combien d'utilisateurs actifs sur Acme ?",
        "Et sur Globex ?",
        "Et sur ZenithCorp ?",  # n'existe pas
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Question : {q}[/bold cyan]")
        # invoke() : exécute le graphe avec un State initial et retourne le State final
        result = agent.invoke({"question": q})
        console.print(Panel(result["answer"], title="Réponse", border_style="orange3"))

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• Le State est passé entre les nodes, chacun renvoie un dict de mises à jour")
    console.print("• Le graphe est compilé une fois, exécuté plusieurs fois (.invoke)")
    console.print("• La trace dans le state permet le debugging\n")


if __name__ == "__main__":
    main()
