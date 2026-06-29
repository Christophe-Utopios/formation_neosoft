import os
from typing import List, Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

import _fake_crm as crm

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


# State : contient la question, l'intent classé, la réponse et le chemin suivi
# Optional[X] et List[X] pour compat Python 3.9 (LangGraph évalue les hints).
class State(TypedDict):
    question: str
    intent: Optional[str]
    answer: Optional[str]
    path_taken: List[str]  # Trace du chemin pris dans le graphe (à but pédagogique)


CLASSIFY_PROMPT = """Classe la question utilisateur en EXACTEMENT UNE des catégories :
- "factual_crm" : la question concerne des données du CRM (tenants, users, tickets)
- "chitchat" : politesse, salutation, question hors-sujet
- "escalate" : demande d'action sensible (créer/supprimer un compte, changer un plan)

Réponds UNIQUEMENT avec la catégorie, sans explication.

Question : {question}"""


def classify_intent(state: State) -> dict:
    """
    Node de classification : détermine l'intention de la question utilisateur.
    Retourne l'intent qui sera utilisé pour le routage conditionnel.
    """
    llm = Anthropic()
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=20, temperature=0,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(question=state["question"])}],
    )
    intent = msg.content[0].text.strip().lower()
    # Nettoyer et normaliser l'intent
    for known in ["factual_crm", "chitchat", "escalate"]:
        if known in intent:
            intent = known
            break
    else:
        intent = "factual_crm"  # défaut
    console.print(f"[dim]→ classify_intent : '{intent}'[/dim]")
    return {"intent": intent, "path_taken": state.get("path_taken", []) + ["classify"]}


def handle_factual(state: State) -> dict:
    """
    Branche 1 : Questions factuelles CRM.
    Recherche les données dans le CRM et génère une réponse basée sur les faits.
    """
    llm = Anthropic()
    # Lookup naïf : chercher des noms de tenants connus dans la question
    found_tenant = None
    for t in crm.TENANTS.values():
        if t["name"].lower() in state["question"].lower():
            found_tenant = t
            break
    if found_tenant:
        n = crm.count_active_users(found_tenant["id"])
        ctx = f"Tenant {found_tenant['name']} (plan {found_tenant['plan']}) : {n} utilisateurs actifs."
    else:
        ctx = "Aucun tenant identifié dans la question."

    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Q : {state['question']}\nDonnées : {ctx}\nRéponds :"}],
    )
    return {"answer": msg.content[0].text.strip(), "path_taken": state["path_taken"] + ["handle_factual"]}


def handle_chitchat(state: State) -> dict:
    """
    Branche 2 : Conversation informelle (politesse, hors-sujet).
    Génère une réponse conversationnelle sans accéder au CRM.
    """
    llm = Anthropic()
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=100, temperature=0.3,
        messages=[{"role": "user", "content": f"Réponds brièvement à cette question conversationnelle.\n\n{state['question']}"}],
    )
    return {"answer": msg.content[0].text.strip(), "path_taken": state["path_taken"] + ["handle_chitchat"]}


def escalate_to_human(state: State) -> dict:
    """
    Branche 3 : Actions sensibles nécessitant validation humaine.
    L'agent ne traite pas la demande directement (sécurité).
    """
    return {
        "answer": (
            "Cette demande nécessite une validation humaine. "
            "Un ticket a été créé pour notre équipe support qui vous recontactera."
        ),
        "path_taken": state["path_taken"] + ["escalate_to_human"],
    }


def route_by_intent(state: State) -> str:
    """
    Fonction de routage : décide du prochain node en fonction de l'intent.
    Fonction Python pure (déterministe) sans appel LLM.
    """
    return {
        "factual_crm": "handle_factual",
        "chitchat": "handle_chitchat",
        "escalate": "escalate_to_human",
    }.get(state["intent"], "handle_factual")


def build_graph():
    """
    Construction du graphe avec routage conditionnel.
    Structure : classify → route_by_intent → [factual | chitchat | escalate] → END
    """
    graph = StateGraph(State)
    # Ajout des nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("handle_factual", handle_factual)
    graph.add_node("handle_chitchat", handle_chitchat)
    graph.add_node("escalate_to_human", escalate_to_human)

    # Edge simple : toujours vers classify
    graph.add_edge(START, "classify")

    # Edge conditionnelle : classify → fonction de routage → node approprié
    graph.add_conditional_edges(
        "classify",  # Node source
        route_by_intent,  # Fonction qui retourne le nom du node cible
        {
            # Mapping : valeur retournée → node cible
            "handle_factual": "handle_factual",
            "handle_chitchat": "handle_chitchat",
            "escalate_to_human": "escalate_to_human",
        },
    )

    # Toutes les branches mènent à END
    for node in ["handle_factual", "handle_chitchat", "escalate_to_human"]:
        graph.add_edge(node, END)

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 06 — Conditional edges[/bold orange3]\n")
    agent = build_graph()

    questions = [
        "Bonjour, comment ça va ?",
        "Combien d'utilisateurs actifs sur Globex ?",
        "Supprime définitivement le compte de bob@acme.com",
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Q : {q}[/bold cyan]")
        result = agent.invoke({"question": q, "path_taken": []})
        console.print(Panel(result["answer"], title="Réponse", border_style="orange3"))
        console.print(f"[dim]Chemin : {' → '.join(result['path_taken'])}[/dim]")

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• Le node `classify` décide du routage en sortie via une fonction Python pure")
    console.print("• La table `route_by_intent` mappe l'intent vers le nom du node suivant")
    console.print("• Aucun appel LLM dans la décision de routage → déterministe et rapide\n")


if __name__ == "__main__":
    main()
