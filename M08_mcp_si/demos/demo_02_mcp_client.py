from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
SERVER_SCRIPT = Path(__file__).resolve().parent / "demo_01_first_mcp_server.py"


async def main() -> None:
    console.print("\n[bold orange3]Demo 02 — Client MCP stdio[/bold orange3]\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize
            await session.initialize()
            console.print("[green]✓ Session MCP initialisée[/green]")

            # 2. Lister les tools disponibles
            tools_resp = await session.list_tools()
            table = Table(title="Tools exposés par le serveur", show_lines=True)
            table.add_column("Nom", style="cyan")
            table.add_column("Description", overflow="fold")
            for t in tools_resp.tools:
                desc = (t.description or "")[:120]
                table.add_row(t.name, desc)
            console.print(table)

            # 3. Appeler un tool
            console.print("\n[bold cyan]Appel : find_tenant_by_name(name='Acme')[/bold cyan]")
            result = await session.call_tool("find_tenant_by_name", {"name": "Acme"})
            for item in result.content:
                console.print(Panel(str(item.text), border_style="orange3"))

            # 4. Appel chaîné : tenant → users actifs
            console.print("\n[bold cyan]Appel : count_active_users(tenant_id='t_42')[/bold cyan]")
            result2 = await session.call_tool("count_active_users", {"tenant_id": "t_42"})
            for item in result2.content:
                console.print(Panel(str(item.text), border_style="orange3"))

            # 5. Appel avec arg manquant pour voir l'erreur
            console.print("\n[bold cyan]Test erreur : tenant_id inexistant[/bold cyan]")
            result3 = await session.call_tool("count_active_users", {"tenant_id": "t_999"})
            for item in result3.content:
                console.print(Panel(str(item.text), border_style="red"))

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• Le client `spawn` le serveur en sous-process (transport stdio)")
    console.print("• La session gère initialize/list/call via JSON-RPC")
    console.print("• Le résultat arrive en `content` (list de blocks text/image/etc.)")


if __name__ == "__main__":
    asyncio.run(main())
