from __future__ import annotations

import os
import json

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

import _fake_crm as crm

load_dotenv()
console = Console()
LLM_MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "find_tenant_by_name",
        "description": "Recherche un tenant client par nom",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    {
        "name": "count_active_users",
        "description": "Compte les utilisateurs actifs d'un tenant donné",
        "input_schema": {"type": "object", "properties": {"tenant_id": {"type": "string"}}, "required": ["tenant_id"]},
    },
    {
        "name": "list_tickets",
        "description": "Liste les tickets d'un tenant par statut (open/closed)",
        "input_schema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}, "status": {"type": "string"}},
            "required": ["tenant_id"],
        },
    },
]

TOOL_FNS = {
    "find_tenant_by_name": lambda **kw: crm.find_tenant_by_name(kw["name"]),
    "count_active_users": lambda **kw: crm.count_active_users(kw["tenant_id"]),
    "list_tickets": lambda **kw: crm.list_tickets(kw["tenant_id"], kw.get("status", "open")),
}


def run_agent(question: str, llm: Anthropic, max_iter: int = 6) -> str:
    """Agent ReAct natif : boucle Thought → Action → Observation."""
    messages = [{"role": "user", "content": question}]
    iteration = 0

    while iteration < max_iter:
        iteration += 1
        console.print(f"\n[bold cyan]── Itération {iteration} ──[/bold cyan]")

        # Appel au LLM : raisonnement (Thought) + décision d'action (Action)
        response = llm.messages.create(
            model=LLM_MODEL, max_tokens=1024, temperature=0,
            tools=TOOLS, messages=messages,
        )

        # Affichage du raisonnement et des actions demandées
        for block in response.content:
            if hasattr(block, "text") and block.text:
                console.print(Panel(block.text, title="Thought / Réponse", border_style="orange3"))
            elif block.type == "tool_use":
                console.print(f"[bold yellow]🔧 Action : {block.name}({json.dumps(block.input)})[/bold yellow]")

        # Si réponse finale, on termine
        if response.stop_reason == "end_turn":
            console.print("[green]✓ Final answer atteint[/green]")
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            return text.strip()

        # Exécution des outils (Observation) et préparation du prochain tour
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FNS.get(block.name)
                result = fn(**block.input) if fn else {"error": f"unknown tool {block.name}"}
                console.print(f"[dim]   Observation : {result}[/dim]")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        # Injection des observations pour le prochain raisonnement
        messages.append({"role": "user", "content": tool_results})

    return "Max iterations atteint sans final answer"


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 02 — ReAct natif Anthropic[/bold orange3]\n")
    llm = Anthropic()

    question = "Pour le tenant Globex, donne-moi le plan, le nb d'utilisateurs actifs, et les tickets ouverts."
    console.print(Panel(question, title="Question utilisateur", border_style="cyan"))

    final = run_agent(question, llm)
    console.print(Panel(final, title="🎯 Réponse finale", border_style="green"))


if __name__ == "__main__":
    main()
