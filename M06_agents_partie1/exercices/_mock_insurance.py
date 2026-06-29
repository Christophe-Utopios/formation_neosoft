"""Mock CRM courtier en assurance pour les TPs M6."""
from __future__ import annotations

CLIENTS: dict[str, dict] = {
    "c_1": {"id": "c_1", "name": "Jean Dupont", "email": "jean.dupont@example.com",
            "age": 42, "city": "Lyon", "risk_score": "low"},
    "c_2": {"id": "c_2", "name": "Carole Dubois", "email": "carole.dubois@example.com",
            "age": 58, "city": "Marseille", "risk_score": "med"},
    "c_3": {"id": "c_3", "name": "Maxime Lemoine", "email": "maxime.lemoine@example.com",
            "age": 28, "city": "Paris", "risk_score": "low"},
    "c_4": {"id": "c_4", "name": "Sophie Martin", "email": "sophie.martin@example.com",
            "age": 71, "city": "Bordeaux", "risk_score": "high"},
}

CONTRACTS: dict[str, dict] = {
    "k_1": {"id": "k_1", "client_id": "c_1", "product": "Multirisque Habitation",
            "premium_eur": 380, "status": "active"},
    "k_2": {"id": "k_2", "client_id": "c_1", "product": "Auto Tous Risques",
            "premium_eur": 720, "status": "active"},
    "k_3": {"id": "k_3", "client_id": "c_2", "product": "Santé Senior",
            "premium_eur": 1450, "status": "active"},
    "k_4": {"id": "k_4", "client_id": "c_4", "product": "Multirisque Habitation",
            "premium_eur": 480, "status": "expired"},
}

PRODUCTS: dict[str, dict] = {
    "Multirisque Habitation": {"min_age": 18, "max_age": 99, "max_risk": "high", "base_premium": 350},
    "Auto Tous Risques": {"min_age": 21, "max_age": 80, "max_risk": "med", "base_premium": 700},
    "Auto Premium": {"min_age": 25, "max_age": 75, "max_risk": "low", "base_premium": 950},
    "Santé Famille": {"min_age": 18, "max_age": 65, "max_risk": "med", "base_premium": 1100},
    "Santé Senior": {"min_age": 60, "max_age": 99, "max_risk": "high", "base_premium": 1450},
}

QUOTES: dict[str, dict] = {}


def find_client(query: str) -> dict | None:
    """Recherche par email exact ou nom partiel."""
    q = query.lower().strip()
    for c in CLIENTS.values():
        if q == c["email"].lower() or q in c["name"].lower():
            return c
    return None


def get_client_contracts(client_id: str) -> list[dict]:
    return [k for k in CONTRACTS.values() if k["client_id"] == client_id]


def is_product_available(product: str, client_id: str) -> dict:
    """Vérifie si un produit est disponible pour un client (âge + risque)."""
    p = PRODUCTS.get(product)
    if not p:
        return {"available": False, "reason": f"Produit '{product}' inconnu"}
    c = CLIENTS.get(client_id)
    if not c:
        return {"available": False, "reason": f"Client '{client_id}' inconnu"}

    if c["age"] < p["min_age"]:
        return {"available": False, "reason": f"Âge {c['age']} < min {p['min_age']}"}
    if c["age"] > p["max_age"]:
        return {"available": False, "reason": f"Âge {c['age']} > max {p['max_age']}"}

    risk_order = {"low": 1, "med": 2, "high": 3}
    if risk_order[c["risk_score"]] > risk_order[p["max_risk"]]:
        return {"available": False, "reason": f"Risque {c['risk_score']} > max {p['max_risk']}"}

    return {"available": True, "base_premium_eur": p["base_premium"]}


def create_quote(client_id: str, product: str) -> dict:
    avail = is_product_available(product, client_id)
    if not avail["available"]:
        return {"created": False, "reason": avail["reason"]}

    new_id = f"q_{len(QUOTES) + 1}"
    quote = {
        "id": new_id,
        "client_id": client_id,
        "product": product,
        "premium_eur": avail["base_premium_eur"],
        "status": "draft",
    }
    QUOTES[new_id] = quote
    return {"created": True, "quote": quote}


def list_clients() -> list[dict]:
    return list(CLIENTS.values())
