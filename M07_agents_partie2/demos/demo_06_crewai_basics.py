from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

# CrewAI ≥ 0.80 nécessite Python 3.10+. Skip propre si non disponible.
try:
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM
    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_AVAILABLE = False
    _IMPORT_ERROR = str(e)


def main() -> None:
    if not CREWAI_AVAILABLE:
        console.print(Panel(
            f"CrewAI non disponible dans cet environnement.\n\n"
            f"Détail : {_IMPORT_ERROR}\n\n"
            f"Pour cette démo, créer un env Python 3.11 dédié :\n"
            f"  [bold]pyenv install 3.11.10[/bold]\n"
            f"  [bold]pyenv virtualenv 3.11.10 myenv311[/bold]\n"
            f"  [bold]pip install crewai>=0.80[/bold]\n\n"
            f"Python en cours : {sys.version.split()[0]}",
            title="⚠ Démo CrewAI nécessite Python 3.10+",
            border_style="yellow",
        ))
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 06 — CrewAI 2 agents[/bold orange3]\n")

    # CrewAI utilise litellm sous le capot — pour Anthropic, format "anthropic/claude-..."
    llm = LLM(model="anthropic/claude-haiku-4-5-20251001", temperature=0)

    # Agent 1 : Researcher
    researcher = Agent(
        role="Senior Tech Researcher",
        goal="Identifier les 3 différences clés entre LangGraph et CrewAI pour un projet d'agent en production.",
        backstory=(
            "Vous êtes un AI Engineer expérimenté qui a déployé plus de 20 agents en production. "
            "Vous savez identifier les compromis pratiques entre frameworks."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # Agent 2 : Writer
    writer = Agent(
        role="Tech Documentation Writer",
        goal="Rédiger un résumé clair et structuré, adapté à un public d'AI Engineers.",
        backstory=(
            "Vous êtes un rédacteur technique senior qui écrit pour des développeurs. "
            "Vous privilégiez les bullet points concrets aux longues phrases."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # Tasks (input → output enchaînés via process séquentiel)
    research_task = Task(
        description="Compare LangGraph et CrewAI selon : contrôle, checkpointing, multi-agents. Note 3 forces et 3 limites de chaque.",
        expected_output="Liste structurée en 6 points par framework, factuelle et concise.",
        agent=researcher,
    )

    writing_task = Task(
        description="À partir de l'analyse précédente, rédige un résumé exécutif de 200 mots pour un comité technique.",
        expected_output="Résumé exécutif markdown, ≤ 200 mots, public AI Engineers.",
        agent=writer,
        context=[research_task],
    )

    # Crew avec process séquentiel
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=False,
    )

    console.print("[cyan]Lancement de la crew...[/cyan]")
    result = crew.kickoff()

    console.print(Panel(str(result.raw), title="Résultat final (Writer)", border_style="orange3"))

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• CrewAI = abstraction haut niveau : Agent + Task + Crew")
    console.print("• Le `context=[research_task]` chaîne les outputs")
    console.print("• Productif pour multi-agents collaboratifs, moins de contrôle granulaire que LangGraph\n")


if __name__ == "__main__":
    main()
