from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"
SERVER_PATH = Path(__file__).resolve().parent / "sol_exo1_mcp_server.py"

SYSTEM_PROMPT = """Tu es un assistant RH pour managers d'équipe.

Tu accèdes au système RH via des tools MCP. Règles :
- Toute modification d'employé nécessite l'ID employé exact (e_<N>).
- Si un nom donné est ambigu, fais d'abord search_employees pour récupérer l'ID.
- Pour les changements sensibles (email), confirme l'action en sortie.
- Réponds de façon concise et structurée (tableaux markdown si pertinent)."""


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Solution TP 2 — Agent + MCP[/bold orange3]\n")

    mcp_client = MultiServerMCPClient({
        "hr": {
            "command": sys.executable,
            "args": [str(SERVER_PATH)],
            "transport": "stdio",
        }
    })

    tools = await mcp_client.get_tools()
    console.print(f"[green]✓ {len(tools)} tools chargés depuis le serveur MCP RH[/green]")
    for t in tools:
        console.print(f"  • {t.name}")

    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    agent = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)

    questions = [
        "Combien d'employés y a-t-il dans le département Engineering ?",
        "Qui est Maxime Lemoine et dans quel département ?",
        "Change l'email de Jules Bernard vers j.bernard@nouveau.fr",
        "Donne-moi la liste des employés du département Sales avec leurs emails.",
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Q : {q}[/bold cyan]")
        async for chunk in agent.astream(
            {"messages": [("user", q)]},
            stream_mode="values",
            config={"recursion_limit": 12},
        ):
            msg = chunk["messages"][-1]
            if msg.type == "ai":
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        console.print(f"[yellow]🔧 {call['name']}({call['args']})[/yellow]")
                elif msg.content:
                    text = str(msg.content)
                    console.print(Panel(text[:1500] + ("..." if len(text) > 1500 else ""),
                                        border_style="orange3"))
            elif msg.type == "tool":
                console.print(f"[dim]   ↩ {str(msg.content)[:120]}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
