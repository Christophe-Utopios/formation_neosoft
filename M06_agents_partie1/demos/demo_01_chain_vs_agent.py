from __future__ import annotations

import os
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import _fake_crm as crm

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"


# === CHAÎNE FIXE ===
def fixed_chain(question: str, llm: Anthropic) -> dict:
    """Chaîne hard-codée : séquence prédéfinie d'étapes (pas de décision dynamique)."""
    # Étape 1 : extraction du nom du tenant par LLM
    extract = llm.messages.create(
        model=LLM_MODEL, max_tokens=50, temperature=0,
        messages=[{"role": "user", "content": f"Quel est le nom du tenant mentionné ? Réponds uniquement avec le nom.\n\n{question}"}],
    )
    tenant_name = extract.content[0].text.strip()

    # Étape 2 : appels CRM (toujours les mêmes, pas d'adaptation)
    tenant = crm.find_tenant_by_name(tenant_name)
    if not tenant:
        return {"answer": f"Tenant '{tenant_name}' introuvable.", "n_calls": 1}
    n_users = crm.count_active_users(tenant["id"])

    # Étape 3 : génération de la réponse finale
    answer = llm.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Question : {question}\nDonnées : tenant={tenant}, users_actifs={n_users}\n\nRéponse :"}],
    )
    return {"answer": answer.content[0].text.strip(), "n_calls": 2}


# === AGENT ReAct (boucle libre) ===
TOOLS = [
    {
        "name": "find_tenant_by_name",
        "description": "Recherche un tenant client par nom",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    {
        "name": "count_active_users",
        "description": "Compte les utilisateurs actifs d'un tenant",
        "input_schema": {"type": "object", "properties": {"tenant_id": {"type": "string"}}, "required": ["tenant_id"]},
    },
    {
        "name": "list_tickets",
        "description": "Liste les tickets d'un tenant par statut",
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


def react_agent(question: str, llm: Anthropic, max_iter: int = 6) -> dict:
    """Agent ReAct : boucle autonome où le LLM décide quels outils utiliser."""
    messages = [{"role": "user", "content": question}]
    n_calls = 0
    for _ in range(max_iter):
        # Le LLM raisonne et décide de l'action suivante (tool ou réponse finale)
        response = llm.messages.create(
            model=LLM_MODEL, max_tokens=1024, temperature=0,
            tools=TOOLS, messages=messages,
        )
        n_calls += 1
        # Si le LLM produit une réponse finale (pas d'outil), on termine
        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            return {"answer": text.strip(), "n_calls": n_calls}

        # Exécution des outils demandés par le LLM
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FNS.get(block.name)
                result = fn(**block.input) if fn else {"error": "unknown tool"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        # Injection des résultats d'outils pour le prochain tour
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "Max iterations atteint", "n_calls": n_calls}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 01 — Chaîne vs Agent[/bold orange3]\n")
    llm = Anthropic()

    questions = [
        ("Cas A — simple", "Combien d'utilisateurs actifs sur Acme ?"),
        ("Cas B — multi-step", "Pour Acme, donne-moi : nb d'utilisateurs actifs, nb de tickets ouverts, et leur priorité."),
    ]

    table = Table(title="Comparaison sur 2 cas", show_lines=True)
    table.add_column("Cas", style="cyan")
    table.add_column("Approche")
    table.add_column("LLM calls", justify="right")
    table.add_column("Latence (s)", justify="right")
    table.add_column("Réponse OK ?", justify="center")

    for label, question in questions:
        console.print(f"\n[bold cyan]{label} : {question}[/bold cyan]")

        # Chaîne
        t0 = time.perf_counter()
        chain_result = fixed_chain(question, llm)
        t_chain = time.perf_counter() - t0
        console.print(Panel(chain_result["answer"], title="Chaîne", border_style="orange3"))

        # Agent
        t0 = time.perf_counter()
        agent_result = react_agent(question, llm)
        t_agent = time.perf_counter() - t0
        console.print(Panel(agent_result["answer"], title="Agent ReAct", border_style="orange3"))

        table.add_row(label, "Chaîne", str(chain_result["n_calls"]), f"{t_chain:.1f}", "?")
        table.add_row(label, "Agent", str(agent_result["n_calls"]), f"{t_agent:.1f}", "?")

    console.print("\n")
    console.print(table)
    console.print("\n[bold]Observation :[/bold]")
    console.print("• Cas A : la chaîne suffit, l'agent surconsomme")
    console.print("• Cas B : l'agent gère naturellement les multi-steps que la chaîne fixe ne couvre pas")
    console.print("• → Choisir l'agent uniquement quand le pattern le justifie\n")


if __name__ == "__main__":
    main()
