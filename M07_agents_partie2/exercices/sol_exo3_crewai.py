from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM
    CREWAI_OK = True
except ImportError as e:
    CREWAI_OK = False
    _IMPORT_ERR = str(e)


def main() -> None:
    if not CREWAI_OK:
        console.print(Panel(
            f"CrewAI non disponible dans cet environnement.\n\n"
            f"Détail : {_IMPORT_ERR}\n\n"
            f"Créer un env Python 3.11 dédié :\n"
            f"  pyenv install 3.11.10 && pyenv virtualenv 3.11.10 myenv311\n"
            f"  pip install crewai>=0.80\n\n"
            f"Python en cours : {sys.version.split()[0]}",
            title="⚠ Solution CrewAI nécessite Python 3.10+", border_style="yellow",
        ))
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Solution TP 3 CrewAI — Veille concurrentielle[/bold orange3]\n")

    llm = LLM(model="anthropic/claude-haiku-4-5-20251001", temperature=0.2)

    researcher = Agent(
        role="Analyste de marché senior",
        goal="Identifier 3 concurrents principaux du produit cible et leurs caractéristiques.",
        backstory="Vous êtes un analyste expérimenté en veille concurrentielle SaaS B2B.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    writer = Agent(
        role="Rédacteur exécutif",
        goal="Produire un rapport markdown synthétique à destination du comité de direction.",
        backstory="Vous excellez à condenser une analyse complexe en moins de 300 mots.",
        llm=llm, verbose=False, allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Pour le produit 'plateforme SaaS de gestion documentaire pour PME', "
            "identifie 3 concurrents principaux. Pour chacun : nom, 2 forces, 2 limites. "
            "Format markdown."
        ),
        expected_output="Une analyse markdown structurée par concurrent.",
        agent=researcher,
    )
    writing_task = Task(
        description=(
            "À partir de l'analyse, produis un rapport markdown ≤ 300 mots avec : "
            "(1) synthèse 3 phrases, (2) tableau comparatif, (3) recommandation."
        ),
        expected_output="Rapport markdown.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task],
                process=Process.sequential, verbose=False)

    t0 = time.perf_counter()
    result = crew.kickoff()
    elapsed = time.perf_counter() - t0

    console.print(Panel(str(result.raw), title="Rapport CrewAI", border_style="green"))
    console.print(f"\n[bold]Durée totale : {elapsed:.1f}s[/bold]")
    console.print("[bold]LOC implementation : ~40 (agents + tasks + crew)[/bold]")


if __name__ == "__main__":
    main()
