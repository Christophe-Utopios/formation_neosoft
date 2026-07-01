from __future__ import annotations

import sqlite3
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id TEXT,
            status TEXT,
            total_eur REAL
        )
    """)
    cur.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [
            (1, "c_1", "active", 120.0),
            (2, "c_1", "shipped", 80.0),
            (3, "c_2", "active", 45.0),
            (4, "c_3", "cancelled", 200.0),
        ],
    )
    conn.commit()
    return conn


# ANTI-PATTERN : SQL libre exposé comme tool
def unsafe_query(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Exécute du SQL fourni par le LLM. Dangereux."""
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in rows]


# BON PATTERN : tool métier avec requête paramétrée
def list_orders_by_customer(
    conn: sqlite3.Connection,
    customer_id: str,
    status: Literal["active", "shipped", "cancelled"] = "active",
) -> list[dict]:
    """Liste les commandes d'un client par statut.

    Args:
        customer_id: ID client (ex: 'c_1')
        status: 'active', 'shipped' ou 'cancelled'

    Returns:
        Liste des commandes correspondantes (peut être vide).
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, customer_id, status, total_eur FROM orders WHERE customer_id = ? AND status = ?",
        (customer_id, status),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# BON PATTERN avec analyse globale
def total_revenue_by_customer(conn: sqlite3.Connection) -> list[dict]:
    """Total des commandes par client (status 'shipped' uniquement)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT customer_id, SUM(total_eur) AS revenue
        FROM orders
        WHERE status = 'shipped'
        GROUP BY customer_id
        ORDER BY revenue DESC
    """)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def main() -> None:
    console.print("\n[bold orange3]Demo 05 — SQL safe vs unsafe[/bold orange3]\n")
    conn = setup_db()

    # 1. Cas du SQL libre — démontrer l'injection
    console.print("[bold red] ANTI-PATTERN : SQL libre[/bold red]")
    malicious_input = "SELECT * FROM orders; DROP TABLE orders; --"
    console.print(f"[dim]LLM envoie : {malicious_input}[/dim]")
    try:
        # sqlite3 refuse les statements multiples par défaut, mais le concept est là
        result = unsafe_query(conn, malicious_input)
        console.print(f"[red]Résultat : {result}[/red]")
    except sqlite3.ProgrammingError as e:
        console.print(f"[red]Exception (sqlite refuse les multi-statements) : {e}[/red]")
    # Plus subtil : `1=1` qui retourne tout
    bypass = "SELECT * FROM orders WHERE customer_id = 'c_1' OR 1=1"
    result_bypass = unsafe_query(conn, bypass)
    console.print(f"[red]Bypass via 1=1 → {len(result_bypass)} lignes (devrait être ≤ 2)[/red]\n")

    # 2. Bons patterns
    console.print("[bold green]✓ BON PATTERN : tool métier paramétré[/bold green]\n")

    table = Table(title="list_orders_by_customer(c_1, active)", show_lines=False)
    table.add_column("id")
    table.add_column("status")
    table.add_column("total_eur", justify="right")
    for row in list_orders_by_customer(conn, "c_1", "active"):
        table.add_row(str(row["id"]), row["status"], f"{row['total_eur']:.2f}")
    console.print(table)

    # Essai d'injection sur le tool métier
    console.print("\n[cyan]Tentative d'injection via customer_id :[/cyan]")
    injection = "c_1' OR '1'='1"
    result = list_orders_by_customer(conn, injection, "active")
    console.print(f"[green]→ {len(result)} ligne(s) (l'injection est traitée comme une chaîne)[/green]")

    # Analyse agrégée
    console.print("\n[bold]Tool d'analyse : total_revenue_by_customer[/bold]")
    table2 = Table()
    table2.add_column("customer_id")
    table2.add_column("revenue", justify="right")
    for row in total_revenue_by_customer(conn):
        table2.add_row(row["customer_id"], f"{row['revenue']:.2f}")
    console.print(table2)

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• Le tool métier accepte des paramètres, pas du SQL")
    console.print("• Les paramètres sont passés à `?` placeholders → pas d'injection")
    console.print("• Le LLM ne voit jamais le SQL réel → moindre surface d'attaque\n")
    conn.close()


if __name__ == "__main__":
    main()
