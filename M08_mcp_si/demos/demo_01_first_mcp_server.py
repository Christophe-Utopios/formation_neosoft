"""
Test interactif :
    npx @modelcontextprotocol/inspector python demo_01_first_mcp_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Réutilise le mock CRM du M6
M6_DEMOS = Path(__file__).resolve().parent.parent.parent / "M06_agents_partie1" / "demos"
sys.path.insert(0, str(M6_DEMOS))
import _fake_crm as crm

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-crm-server")


@mcp.tool()
def find_tenant_by_name(name: str) -> dict:
    """Recherche un tenant client par son nom (exact ou contenant).

    Utiliser quand : on connaît le nom commercial du tenant (ex: 'Acme', 'Globex').
    Ne PAS utiliser pour : recherche par ID (utiliser get_tenant à la place).

    Args:
        name: nom partiel ou complet du tenant.

    Returns:
        Dict {id, name, plan, seats} ou {"error": "not_found"}.
    """
    t = crm.find_tenant_by_name(name)
    return t if t else {"error": "not_found", "query": name}


@mcp.tool()
def count_active_users(tenant_id: str) -> dict:
    """Compte les utilisateurs ACTIFS d'un tenant donné.

    Un utilisateur 'actif' = compte non désactivé, peu importe la connexion récente.

    Args:
        tenant_id: identifiant interne (format 't_<N>', ex: 't_42').
    """
    if tenant_id not in crm.TENANTS:
        return {"error": "tenant_not_found", "tenant_id": tenant_id}
    return {"tenant_id": tenant_id, "count": crm.count_active_users(tenant_id)}


@mcp.tool()
def list_tickets(tenant_id: str, status: str = "open") -> list[dict]:
    """Liste les tickets d'un tenant filtrés par statut.

    Args:
        tenant_id: ID du tenant.
        status: 'open' (par défaut) ou 'closed'.

    Returns:
        Liste de dicts {id, subject, priority, status} (vide si rien trouvé).
    """
    return crm.list_tickets(tenant_id, status)


if __name__ == "__main__":
    # Mode stdio — lancement par le client MCP
    mcp.run()