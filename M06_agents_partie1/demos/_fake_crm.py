"""
Mock CRM partagé entre les démos M6.

Simule un mini-SI : tenants, utilisateurs, abonnements, tickets.
Pas de DB, juste des dicts en mémoire pour la pédagogie.
"""
from __future__ import annotations

TENANTS: dict[str, dict] = {
    "t_42": {"id": "t_42", "name": "Acme Corp", "plan": "Pro", "seats": 100},
    "t_43": {"id": "t_43", "name": "Globex", "plan": "Enterprise", "seats": 500},
    "t_44": {"id": "t_44", "name": "Initech", "plan": "Starter", "seats": 10},
}

USERS: dict[str, dict] = {
    "u_1": {"id": "u_1", "email": "alice@acme.com", "tenant_id": "t_42", "active": True, "role": "admin"},
    "u_2": {"id": "u_2", "email": "bob@acme.com", "tenant_id": "t_42", "active": True, "role": "user"},
    "u_3": {"id": "u_3", "email": "carol@acme.com", "tenant_id": "t_42", "active": False, "role": "user"},
    "u_4": {"id": "u_4", "email": "dave@globex.com", "tenant_id": "t_43", "active": True, "role": "admin"},
    "u_5": {"id": "u_5", "email": "eve@initech.com", "tenant_id": "t_44", "active": True, "role": "owner"},
}

TICKETS: dict[str, dict] = {
    "tk_1": {"id": "tk_1", "tenant_id": "t_42", "subject": "Erreur SAML", "status": "open", "priority": "high"},
    "tk_2": {"id": "tk_2", "tenant_id": "t_42", "subject": "Question facturation", "status": "open", "priority": "low"},
    "tk_3": {"id": "tk_3", "tenant_id": "t_43", "subject": "Performance API", "status": "closed", "priority": "high"},
    "tk_4": {"id": "tk_4", "tenant_id": "t_44", "subject": "Demande de fonctionnalité", "status": "open", "priority": "med"},
}


def find_tenant_by_name(name: str) -> dict | None:
    """Recherche un tenant par nom (exact ou contenant)."""
    name_lower = name.lower()
    for t in TENANTS.values():
        if name_lower == t["name"].lower() or name_lower in t["name"].lower():
            return t
    return None


def get_tenant(tenant_id: str) -> dict | None:
    return TENANTS.get(tenant_id)


def count_active_users(tenant_id: str) -> int:
    return sum(1 for u in USERS.values() if u["tenant_id"] == tenant_id and u["active"])


def list_tickets(tenant_id: str, status: str = "open") -> list[dict]:
    return [t for t in TICKETS.values() if t["tenant_id"] == tenant_id and t["status"] == status]


def create_ticket(tenant_id: str, subject: str, priority: str = "med") -> dict:
    new_id = f"tk_{len(TICKETS) + 1}"
    ticket = {
        "id": new_id, "tenant_id": tenant_id, "subject": subject,
        "status": "open", "priority": priority,
    }
    TICKETS[new_id] = ticket
    return ticket
