from __future__ import annotations

import sys
from pathlib import Path

# Réutilise le mock RH du M7
M7_EX = Path(__file__).resolve().parent.parent.parent / "M07_agents_partie2" / "exercises"
sys.path.insert(0, str(M7_EX))
import _mock_hr as hr

from mcp.server.fastmcp import FastMCP
from pydantic import EmailStr

mcp = FastMCP("hr-server")


@mcp.tool()
def search_employees(query: str) -> list[dict]:
    """Recherche d'employés par nom partiel (case-insensitive).

    Utiliser quand : on connaît une partie du nom et veut trouver l'ID employé.
    Ne PAS utiliser pour : recherche par département (utiliser list_employees_by_dept).

    Args:
        query: morceau de nom à chercher (ex: 'maxime', 'Lemoine').

    Returns:
        Liste des employés correspondants : [{id, name, email, dept, manager, status}].
        Vide si aucun match.
    """
    return [e for e in hr.EMPLOYEES.values() if query.lower() in e["name"].lower()]


@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """Récupère le profil complet d'un employé par son ID interne.

    Utiliser quand : on connaît l'ID (format 'e_<N>').

    Args:
        employee_id: ID interne (ex: 'e_1').

    Returns:
        Dict employé ou {"error": "not_found"}.
    """
    return hr.EMPLOYEES.get(employee_id) or {"error": "not_found", "employee_id": employee_id}


@mcp.tool()
def list_employees_by_dept(dept: str) -> list[dict]:
    """Liste tous les employés actifs d'un département.

    Départements valides : Engineering, Sales, HR.

    Args:
        dept: nom exact du département (case-sensitive).
    """
    return [e for e in hr.EMPLOYEES.values()
            if e["dept"] == dept and e["status"] == "active"]


@mcp.tool()
def update_employee_email(employee_id: str, new_email: EmailStr) -> dict:
    """Modifie l'email d'un employé. Action sensible : audit obligatoire côté appelant.

    Args:
        employee_id: ID employé (format 'e_<N>').
        new_email: nouvelle adresse email (validation Pydantic stricte).

    Returns:
        {success, old_email, new_email} ou {error}.
    """
    emp = hr.EMPLOYEES.get(employee_id)
    if not emp:
        return {"error": "employee_not_found", "employee_id": employee_id}
    old = emp["email"]
    emp["email"] = str(new_email)
    return {"success": True, "old_email": old, "new_email": str(new_email)}


if __name__ == "__main__":
    mcp.run()
