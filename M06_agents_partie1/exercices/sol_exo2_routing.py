from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

# Importer le mock depuis exercises/
sys.path.insert(0, str(Path(__file__).parent.parent / "exercises"))
import _mock_insurance as ins

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


class State(TypedDict):
    question: str
    intent: Optional[str]
    answer: Optional[str]
    needs_validation: bool
    path_taken: List[str]


CLASSIFY_PROMPT = """Classe cette question dans EXACTEMENT UNE catégorie :
- "info" : demande d'information sur un client ou ses contrats existants
- "quote" : demande de création d'un nouveau devis
- "out_of_scope" : tout ce qui sort du métier de courtier en assurance (météo, RH, juridique, salutations seules)

Réponds UNIQUEMENT avec la catégorie, sans explication.

Question : {question}"""


def _llm_call(prompt: str, max_tokens: int = 200, temperature: float = 0) -> str:
    llm = Anthropic()
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def classify_intent(state: State) -> dict:
    raw = _llm_call(CLASSIFY_PROMPT.format(question=state["question"]), max_tokens=20).lower()
    intent = "info"
    for known in ["out_of_scope", "quote", "info"]:
        if known in raw:
            intent = known
            break
    console.print(f"[dim]→ classify_intent : '{intent}'[/dim]")
    return {"intent": intent, "path_taken": state.get("path_taken", []) + ["classify"]}


def _extract_client_query(question: str) -> str | None:
    """Heuristique : trouve un nom complet ou email dans la question."""
    # email
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", question)
    if m:
        return m.group(0)
    # nom : 2 mots capitalisés consécutifs
    m = re.search(r"\b([A-ZÀ-Ý][a-zà-ÿ]+\s+[A-ZÀ-Ý][a-zà-ÿ]+)\b", question)
    if m:
        return m.group(0)
    return None


def handle_info(state: State) -> dict:
    query = _extract_client_query(state["question"])
    if not query:
        return {
            "answer": "Je n'ai pas identifié le client dans votre question. Pouvez-vous préciser un nom ou un email ?",
            "path_taken": state["path_taken"] + ["handle_info"],
        }
    client = ins.find_client(query)
    if not client:
        return {
            "answer": f"Aucun client trouvé pour '{query}'.",
            "path_taken": state["path_taken"] + ["handle_info"],
        }
    contracts = ins.get_client_contracts(client["id"])

    summary_prompt = f"""Tu es un assistant courtier. Synthétise ces données client en 2-3 phrases.

Client : {client}
Contrats : {contracts}

Question initiale : {state['question']}

Synthèse :"""
    answer = _llm_call(summary_prompt, max_tokens=300)
    return {"answer": answer, "path_taken": state["path_taken"] + ["handle_info"]}


def handle_quote(state: State) -> dict:
    query = _extract_client_query(state["question"])
    client = ins.find_client(query) if query else None
    if not client:
        return {
            "answer": f"Client introuvable pour la demande de devis ('{query}').",
            "needs_validation": False,
            "path_taken": state["path_taken"] + ["handle_quote"],
        }

    # Détecter le produit demandé (matching simple)
    product_found = None
    for p in ins.PRODUCTS:
        if p.lower() in state["question"].lower():
            product_found = p
            break
    if not product_found:
        return {
            "answer": (f"Client {client['name']} identifié, mais je n'ai pas reconnu le produit. "
                       f"Produits disponibles : {', '.join(ins.PRODUCTS.keys())}."),
            "needs_validation": False,
            "path_taken": state["path_taken"] + ["handle_quote"],
        }

    quote_result = ins.create_quote(client["id"], product_found)
    if not quote_result["created"]:
        return {
            "answer": f"Devis impossible : {quote_result['reason']}.",
            "needs_validation": False,
            "path_taken": state["path_taken"] + ["handle_quote"],
        }
    q = quote_result["quote"]
    return {
        "answer": (f"Devis #{q['id']} créé pour {client['name']} sur {q['product']} : "
                   f"{q['premium_eur']} € (statut : draft, **validation requise**)."),
        "needs_validation": True,  # nécessite validation humaine avant envoi
        "path_taken": state["path_taken"] + ["handle_quote"],
    }


def handle_out_of_scope(state: State) -> dict:
    return {
        "answer": (
            "Cette demande sort du périmètre du courtage en assurance. "
            "Je vous redirige vers un conseiller humain."
        ),
        "path_taken": state["path_taken"] + ["handle_out_of_scope"],
    }


def route(state: State) -> str:
    return {
        "info": "handle_info",
        "quote": "handle_quote",
        "out_of_scope": "handle_out_of_scope",
    }.get(state["intent"], "handle_info")


def build_graph():
    graph = StateGraph(State)
    graph.add_node("classify", classify_intent)
    graph.add_node("handle_info", handle_info)
    graph.add_node("handle_quote", handle_quote)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {
            "handle_info": "handle_info",
            "handle_quote": "handle_quote",
            "handle_out_of_scope": "handle_out_of_scope",
        },
    )
    for node in ["handle_info", "handle_quote", "handle_out_of_scope"]:
        graph.add_edge(node, END)

    return graph.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Solution TP 2 — Routing dynamique[/bold orange3]\n")
    agent = build_graph()

    try:
        ascii_graph = agent.get_graph().draw_ascii()
        console.print(ascii_graph)
    except Exception as e:
        console.print(f"[dim]ASCII rendering: {e}[/dim]")
        for node in agent.get_graph().nodes:
            console.print(f"  • {node}")

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
