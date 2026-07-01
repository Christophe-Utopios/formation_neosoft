from __future__ import annotations

import sys
from pathlib import Path

# Réutilise le mock RH du M7
M7_EX = Path(__file__).resolve().parent.parent.parent / "M07_agents_partie2" / "exercises"
sys.path.insert(0, str(M7_EX))
import _mock_hr as hr

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hr-server")


@mcp.tool()
def search_employees(query: str) -> list[dict]:
    """TODO 1 : recherche par nom partiel.

    Description à compléter : ce que ça fait, QUAND l'utiliser, format args.
    """
    return []


@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """TODO 2."""
    return {}


@mcp.tool()
def list_employees_by_dept(dept: str) -> list[dict]:
    """TODO 3 : liste par département (Engineering, Sales, HR)."""
    return []


# TODO 4 : implémenter update_employee_email avec EmailStr validation
# from pydantic import EmailStr
# @mcp.tool()
# def update_employee_email(employee_id: str, new_email: EmailStr) -> dict:
#     ...


if __name__ == "__main__":
    mcp.run()
