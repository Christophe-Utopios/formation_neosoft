from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from mcp import StdioServerParameters
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"
SERVER_SCRIPT = Path(__file__).resolve().parent / "demo_01_first_mcp_server.py"


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant[/red]")
        return

    console.print("\n[bold orange3]Demo 04 — Agent LangGraph + serveur MCP[/bold orange3]\n")

    # Configuration multi-serveurs MCP
    client = MultiServerMCPClient({
        "crm": {
            "command": sys.executable,
            "args": [str(SERVER_SCRIPT)],
            "transport": "stdio",
        }
    })

    # Récupérer les tools depuis le(s) serveur(s) MCP
    tools = await client.get_tools()
    console.print(f"[green]✓ {len(tools)} tools chargés depuis le serveur MCP[/green]")
    for t in tools:
        console.print(f"  • {t.name}")

    # Construire l'agent normalement
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    agent = create_react_agent(llm, tools=tools)

    question = ("Pour le tenant Acme, donne le nombre d'utilisateurs actifs "
                "ainsi que la liste des tickets ouverts avec leur priorité.")
    console.print(f"\n[bold cyan]Question :[/bold cyan] {question}\n")

    # Streamer la conversation
    async for chunk in agent.astream(
        {"messages": [("user", question)]},
        stream_mode="values",
    ):
        msg = chunk["messages"][-1]
        if msg.type == "ai":
            if msg.tool_calls:
                for call in msg.tool_calls:
                    console.print(f"[yellow]🔧 {call['name']}({call['args']})[/yellow]")
            elif msg.content:
                console.print(Panel(str(msg.content)[:1500], border_style="orange3"))
        elif msg.type == "tool":
            console.print(f"[dim]   ↩ {str(msg.content)[:120]}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
