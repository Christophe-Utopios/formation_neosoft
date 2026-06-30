from __future__ import annotations

import os
import time
from typing import Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


class State(TypedDict):
    product: str
    competitors: Optional[str]
    report: Optional[str]


def find_competitors(state: State) -> dict:
    llm = Anthropic()
    prompt = (
        f"Identifie 3 concurrents principaux du produit '{state['product']}'. "
        "Pour chacun : nom, 2 forces, 2 limites. Format markdown structuré."
    )
    msg = llm.messages.create(model=LLM_MODEL, max_tokens=600, temperature=0.2,
                              messages=[{"role": "user", "content": prompt}])
    return {"competitors": msg.content[0].text.strip()}


def synthesize_report(state: State) -> dict:
    llm = Anthropic()
    prompt = (
        f"À partir de cette analyse :\n\n{state['competitors']}\n\n"
        "Produis un rapport executive markdown (≤ 300 mots) avec : "
        "(1) Synthèse exécutive 3 phrases, (2) Tableau comparatif, (3) Recommandation."
    )
    msg = llm.messages.create(model=LLM_MODEL, max_tokens=800, temperature=0,
                              messages=[{"role": "user", "content": prompt}])
    return {"report": msg.content[0].text.strip()}


def build_graph():
    g = StateGraph(State)
    g.add_node("find", find_competitors)
    g.add_node("synthesize", synthesize_report)
    g.add_edge(START, "find")
    g.add_edge("find", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Solution TP 3 LangGraph — Veille concurrentielle[/bold orange3]\n")
    agent = build_graph()

    product = "plateforme SaaS de gestion documentaire pour PME"
    console.print(f"[cyan]Produit : {product}[/cyan]\n")

    t0 = time.perf_counter()
    result = agent.invoke({"product": product, "competitors": None, "report": None})
    elapsed = time.perf_counter() - t0

    console.print(Panel(result["competitors"], title="Étape 1 — Concurrents", border_style="orange3"))
    console.print(Panel(result["report"], title="Étape 2 — Rapport final", border_style="green"))
    console.print(f"\n[bold]Durée totale : {elapsed:.1f}s[/bold]")
    console.print("[bold]LOC implementation : ~30 (graph + 2 nodes)[/bold]")


if __name__ == "__main__":
    main()
