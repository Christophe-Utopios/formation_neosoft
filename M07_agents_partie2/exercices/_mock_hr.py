"""Mock RH pour le TP 2 HIL."""
from __future__ import annotations

EMPLOYEES: dict[str, dict] = {
    "e_1": {"id": "e_1", "name": "Maxime Lemoine", "email": "maxime.lemoine@corp.fr",
            "dept": "Engineering", "manager": "Alice Martin", "status": "active"},
    "e_2": {"id": "e_2", "name": "Camille Dubois", "email": "camille.dubois@corp.fr",
            "dept": "Sales", "manager": "Bob Smith", "status": "active"},
    "e_3": {"id": "e_3", "name": "Jules Bernard", "email": "jules.bernard@corp.fr",
            "dept": "HR", "manager": "Sophie Martin", "status": "active"},
}


def find_employee(name: str) -> dict | None:
    for e in EMPLOYEES.values():
        if name.lower() in e["name"].lower():
            return e
    return None


def delete_account(employee_id: str, reason: str) -> dict:
    if employee_id not in EMPLOYEES:
        return {"success": False, "reason": "employé introuvable"}
    EMPLOYEES[employee_id]["status"] = "deleted"
    EMPLOYEES[employee_id]["deletion_reason"] = reason
    return {"success": True, "employee_id": employee_id, "reason": reason}
