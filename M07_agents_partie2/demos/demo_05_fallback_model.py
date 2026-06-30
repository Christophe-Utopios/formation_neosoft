from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 05 — Fallback de modèle[/bold orange3]\n")

    # Simulons un primaire défaillant : un Runnable qui lève toujours
    def broken_primary(_input):
        raise RuntimeError("503 Service Unavailable (simulated)")

    primary = RunnableLambda(broken_primary)

    # Fallback réel : Claude Haiku
    fallback = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    robust = primary.with_fallbacks([fallback])

    console.print("[cyan]Test 1 — Primaire en panne, fallback Haiku attendu[/cyan]")
    result = robust.invoke([HumanMessage("Donne un seul mot : 'OK' ou 'KO'.")])
    console.print(Panel(str(result.content), title="Réponse via fallback", border_style="green"))

    # Test 2 : avec un primaire qui marche
    console.print("\n[cyan]Test 2 — Les deux dispos, primaire utilisé[/cyan]")
    primary_ok = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    robust_ok = primary_ok.with_fallbacks([fallback])
    r2 = robust_ok.invoke([HumanMessage("Réponds en un mot : 'fallback' ou 'primaire' ?")])
    console.print(Panel(str(r2.content), title="Réponse via primaire", border_style="orange3"))

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• `with_fallbacks([...])` ajoute une chaîne de fallbacks")
    console.print("• Bascule sur exception : timeout, 503, rate limit, etc.")
    console.print("• En production : monitorer le taux de bascule pour détecter les pannes silencieuses\n")


if __name__ == "__main__":
    main()
