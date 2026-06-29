from __future__ import annotations

import json
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import _fake_crm as crm

load_dotenv()
console = Console()
PLANNER_MODEL = "claude-sonnet-4-6"
EXECUTOR_MODEL = "claude-haiku-4-5-20251001"

PLAN_PROMPT_TEMPLATE = """Tu es un planificateur. Décompose la tâche en 2 à 5 étapes ordonnées,
chacune appelant exactement UN tool.

Tools disponibles :
- find_tenant_by_name(name) : trouve un tenant par nom
- count_active_users(tenant_id) : compte les utilisateurs actifs
- list_tickets(tenant_id, status) : liste les tickets

Réponds en JSON STRICT (et rien d'autre) avec le schéma :
{{"plan": [{{"step": 1, "description": "...", "tool": "tool_name", "args": {{...}}}}]}}

Note : si une étape dépend du résultat d'une étape précédente, tu peux mettre
la valeur "$prev" comme placeholder pour l'argument concerné.

Tâche : {task}"""


SYNTHESIZE_PROMPT = """Tu as exécuté ce plan et obtenu ces résultats. Réponds maintenant à la question initiale.

Question : {question}

Plan exécuté :
{trace}

Réponse synthétique :"""


# Mapping des outils disponibles pour l'exécution des étapes du plan
TOOL_FNS = {
    "find_tenant_by_name": lambda args: crm.find_tenant_by_name(args["name"]),
    "count_active_users": lambda args: crm.count_active_users(args["tenant_id"]),
    "list_tickets": lambda args: crm.list_tickets(args["tenant_id"], args.get("status", "open")),
}


def make_plan(task: str, llm: Anthropic) -> list[dict]:
    """
    Phase 1 : Planification - Le LLM décompose la tâche en étapes structurées.
    Retourne une liste d'étapes avec tool à appeler et arguments.
    """
    msg = llm.messages.create(
        model=PLANNER_MODEL, max_tokens=800, temperature=0,
        messages=[{"role": "user", "content": PLAN_PROMPT_TEMPLATE.format(task=task)}],
    )
    text = msg.content[0].text.strip()
    # Extraire le JSON même si entouré de texte
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON introuvable dans : {text}")
    return json.loads(match.group(0))["plan"]


def execute_step(step: dict, prev_result, llm: Anthropic) -> dict:
    """
    Phase 2 : Exécution - Exécute une étape du plan en appelant l'outil correspondant.
    Gère les dépendances entre étapes via le placeholder $prev.
    """
    args = step["args"].copy()
    # Substituer $prev par le résultat de l'étape précédente (chaînage)
    for k, v in args.items():
        if v == "$prev" and prev_result is not None:
            if isinstance(prev_result, dict) and "id" in prev_result:
                args[k] = prev_result["id"]
            else:
                args[k] = prev_result
    fn = TOOL_FNS.get(step["tool"])
    return {"step": step["step"], "args": args, "result": fn(args) if fn else "tool inconnu"}


def synthesize(question: str, trace: list[dict], llm: Anthropic) -> str:
    """
    Phase 3 : Synthèse - Le LLM génère une réponse naturelle basée sur les résultats d'exécution.
    """
    trace_str = "\n".join(
        f"Étape {t['step']} : {t['args']} → {t['result']}" for t in trace
    )
    msg = llm.messages.create(
        model=EXECUTOR_MODEL, max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": SYNTHESIZE_PROMPT.format(question=question, trace=trace_str)}],
    )
    return msg.content[0].text.strip()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 03 — Plan-and-Execute[/bold orange3]\n")
    llm = Anthropic()

    task = "Pour Acme Corp, donne le nombre d'utilisateurs actifs et la liste des tickets ouverts."
    console.print(Panel(task, title="Tâche", border_style="cyan"))

    # 1. Planification : génération d'un plan structuré en JSON
    console.print("\n[cyan]→ Planificateur (Sonnet)...[/cyan]")
    plan = make_plan(task, llm)
    table = Table(title="Plan généré", show_lines=True)
    table.add_column("Étape", justify="center")
    table.add_column("Tool")
    table.add_column("Args")
    table.add_column("Description", overflow="fold")
    for step in plan:
        table.add_row(str(step["step"]), step["tool"], str(step["args"]),
                      step.get("description", ""))
    console.print(table)

    # 2. Exécution : chaque étape est exécutée séquentiellement
    console.print("\n[cyan]→ Exécuteur (Haiku) sur chaque étape...[/cyan]")
    trace = []
    prev_result = None
    for step in plan:
        result = execute_step(step, prev_result, llm)
        trace.append(result)
        console.print(f"  [{step['step']}] {step['tool']}({result['args']}) → {result['result']}")
        prev_result = result["result"]  # Passe le résultat à l'étape suivante

    # 3. Synthèse : génération de la réponse finale en langage naturel
    console.print("\n[cyan]→ Synthèse finale...[/cyan]")
    final = synthesize(task, trace, llm)
    console.print(Panel(final, title="🎯 Réponse finale", border_style="green"))

    console.print("\n[bold]Observation :[/bold]")
    console.print("• Le plan est explicite et auditable")
    console.print("• Les étapes pourraient être parallélisées si indépendantes")
    console.print("• Coût : 1 appel planificateur (Sonnet) + N exécuteur (Haiku) + 1 synthèse\n")


if __name__ == "__main__":
    main()
