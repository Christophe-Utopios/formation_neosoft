# TP 1 — Serveur MCP exposant un mock RH

## Contexte

Vous reprenez le mock RH du M7 (`_mock_hr.py`). Vous devez **l'exposer** comme un serveur MCP pour que n'importe quelle app IA puisse l'utiliser.

## Étape 1 — Définir l'API

Le serveur doit exposer 4 tools :

1. `search_employees(query: str) -> list[dict]` : recherche par nom partiel
2. `get_employee(employee_id: str) -> dict` : récupération par ID
3. `list_employees_by_dept(dept: str) -> list[dict]` : liste par département
4. `update_employee_email(employee_id: str, new_email: str) -> dict` : modification email (Pydantic EmailStr)

Pour chaque tool, **rédiger une description claire** (LLM-readable).

## Étape 2 — Implémenter

Dans `exo1_starter.py` :

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hr-server")

@mcp.tool()
def search_employees(query: str) -> list[dict]:
    """..."""
    ...
```

**Règles** :

- Préfixer chaque tool avec une description complète
- Utiliser Pydantic `EmailStr` pour la validation email
- Retourner toujours du JSON-serializable (dict/list/str/int)

## Étape 3 — Tester avec MCP Inspector

```bash
npx @modelcontextprotocol/inspector python exo1_solution.py
```

Lister les tools, appeler chacun avec différents inputs (valides et invalides). Vérifier :

- Les descriptions sont lisibles
- Les erreurs Pydantic sont propagées proprement
