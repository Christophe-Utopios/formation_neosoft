from __future__ import annotations

import functools
import json
import logging
import sys
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, EmailStr, Field, ValidationError

# Réutilise le mock RH du M7
M7_EX = Path(__file__).resolve().parent.parent.parent / "M07_agents_partie2" / "exercises"
sys.path.insert(0, str(M7_EX))
import _mock_hr as hr

from mcp.server.fastmcp import FastMCP

# Logger audit JSON-only dans audit.jsonl (chemin absolu)
AUDIT_PATH = (Path(__file__).resolve().parent / "audit.jsonl")
audit_logger = logging.getLogger("mcp.audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # éviter le double-log via root
# Réinitialise les handlers existants (réimport modulo --test)
for _h in list(audit_logger.handlers):
    audit_logger.removeHandler(_h)
_fh = logging.FileHandler(str(AUDIT_PATH))
_fh.setFormatter(logging.Formatter("%(message)s"))
audit_logger.addHandler(_fh)

# Rate limit : 10 calls / 60s glissants par actor
CALL_HISTORY: dict[str, deque] = defaultdict(deque)
MAX_CALLS = 10
WINDOW_S = 60


def rate_limit_check(actor_id: str) -> tuple[bool, int]:
    now = time.time()
    h = CALL_HISTORY[actor_id]
    while h and h[0] < now - WINDOW_S:
        h.popleft()
    if len(h) >= MAX_CALLS:
        retry = int(WINDOW_S - (now - h[0]))
        return False, max(retry, 1)
    h.append(now)
    return True, 0


def audit_tool(tool_name: str):
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(actor_id: str = "anonymous", **kwargs):
            trace_id = str(uuid.uuid4())
            started_at = time.perf_counter()

            ok, retry = rate_limit_check(actor_id)
            if not ok:
                entry = {"event": "rate_limited", "tool": tool_name, "actor_id": actor_id,
                         "trace_id": trace_id, "ts": time.time()}
                audit_logger.info(json.dumps(entry))
                return {"error": "rate_limited", "retry_after_s": retry}

            log = {"event": "tool_call", "tool": tool_name, "trace_id": trace_id,
                   "actor_id": actor_id, "args": kwargs, "ts": time.time()}
            try:
                result = fn(actor_id=actor_id, **kwargs)
                log.update({
                    "result": "success",
                    "duration_ms": int((time.perf_counter() - started_at) * 1000),
                })
                audit_logger.info(json.dumps(log))
                return result
            except ValidationError as e:
                log.update({"result": "validation_error", "errors": e.errors(),
                            "duration_ms": int((time.perf_counter() - started_at) * 1000)})
                audit_logger.info(json.dumps(log, default=str))
                return {"error": "validation_error", "details": e.errors()}
            except Exception as e:
                log.update({"result": "error", "error_type": type(e).__name__, "error_msg": str(e),
                            "duration_ms": int((time.perf_counter() - started_at) * 1000)})
                audit_logger.info(json.dumps(log))
                return {"error": "internal_error", "details": str(e)}
        return wrapper
    return decorator


# === Schémas d'input typés ===
class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=100)


class GetEmployeeInput(BaseModel):
    employee_id: str = Field(pattern=r"^e_\d+$")


class UpdateEmailInput(BaseModel):
    employee_id: str = Field(pattern=r"^e_\d+$")
    new_email: EmailStr


# === Tools sécurisés ===
@audit_tool("search_employees")
def _search_employees(actor_id: str, **kwargs) -> dict:
    args = SearchInput(**kwargs)
    matches = [e for e in hr.EMPLOYEES.values() if args.query.lower() in e["name"].lower()]
    return {"count": len(matches), "employees": matches}


@audit_tool("get_employee")
def _get_employee(actor_id: str, **kwargs) -> dict:
    args = GetEmployeeInput(**kwargs)
    emp = hr.EMPLOYEES.get(args.employee_id)
    return emp or {"error": "not_found"}


@audit_tool("update_employee_email")
def _update_email(actor_id: str, **kwargs) -> dict:
    args = UpdateEmailInput(**kwargs)
    emp = hr.EMPLOYEES.get(args.employee_id)
    if not emp:
        return {"error": "employee_not_found"}
    old = emp["email"]
    emp["email"] = str(args.new_email)
    return {"success": True, "old_email": old, "new_email": str(args.new_email)}


# === Couche MCP ===
mcp = FastMCP("hr-secure")


@mcp.tool()
def search_employees(query: str, actor_id: str = "agent_default") -> dict:
    """Recherche employés par nom (rate-limité, audité)."""
    return _search_employees(actor_id=actor_id, query=query)


@mcp.tool()
def get_employee(employee_id: str, actor_id: str = "agent_default") -> dict:
    """Récupère un employé par ID (rate-limité, audité)."""
    return _get_employee(actor_id=actor_id, employee_id=employee_id)


@mcp.tool()
def update_employee_email(employee_id: str, new_email: str,
                          actor_id: str = "agent_default") -> dict:
    """Met à jour l'email d'un employé (validation EmailStr stricte)."""
    return _update_email(actor_id=actor_id, employee_id=employee_id, new_email=new_email)


# === Mode test local (sans serveur MCP) ===
def run_local_tests() -> None:
    # Tronquer le fichier sans le supprimer (pour ne pas invalider le handler)
    AUDIT_PATH.write_text("")

    print("\n=== Test 1 : rate limit (15 calls rapides) ===")
    for i in range(15):
        r = _search_employees(actor_id="bot_test", query="Maxime")
        status = "OK" if "count" in r else r.get("error")
        print(f"  Call {i + 1:>2} → {status}")

    print("\n=== Test 2 : email malformé ===")
    r = _update_email(actor_id="bot_test2", employee_id="e_1", new_email="pas-un-email")
    print(f"  → {r}")

    print("\n=== Test 3 : employee_id mal formé ===")
    r = _get_employee(actor_id="bot_test2", employee_id="employee-42")
    print(f"  → {r}")

    print("\n=== Test 4 : update OK ===")
    r = _update_email(actor_id="bot_test2", employee_id="e_3", new_email="jules.b@nouveau.fr")
    print(f"  → {r}")

    # Flush des handlers pour s'assurer que le fichier est écrit
    for h in audit_logger.handlers:
        h.flush()

    print(f"\n→ Audit logs dans {AUDIT_PATH.resolve()} :")
    if AUDIT_PATH.exists():
        for line in AUDIT_PATH.read_text().splitlines()[:20]:
            print(f"  {line}")
    else:
        print("  (fichier non créé — vérifier les handlers du logger)")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_local_tests()
    else:
        mcp.run()
