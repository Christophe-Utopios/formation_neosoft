# TP 3 — Sécurisation : validation + audit + rate limiting

## Contexte

Vous reprenez le serveur du TP 1. Le DSI veut maintenant :

1. **Validation Pydantic stricte** sur chaque tool
2. **Audit logs JSON structurés** pour chaque appel
3. **Rate limiting** : max 10 appels/min par actor

## Étape 1 — Validation Pydantic

Pour chaque tool, créer un modèle d'input :

```python
class UpdateEmailInput(BaseModel):
    employee_id: str = Field(pattern=r"^e_\d+$")
    new_email: EmailStr
```

Le tool reçoit l'objet pré-validé. Les erreurs Pydantic doivent remonter au client.

## Étape 2 — Audit logs

Décorateur `@audit_tool("tool_name")` qui logue :

- `trace_id` (uuid)
- `actor_id` (passé en argument ou via header)
- `tool`, `args`, `result` ou `error`
- `duration_ms`
- `timestamp`

Logs JSON-only, dans `audit.jsonl`.

## Étape 3 — Rate limiting

In-memory rate limiter (par actor_id, 10 appels / 60 secondes glissants) :

```python
from collections import defaultdict, deque
from time import time

CALL_HISTORY: dict[str, deque] = defaultdict(deque)

def rate_limit_check(actor_id: str, max_calls: int = 10, window_s: int = 60) -> bool:
    now = time()
    history = CALL_HISTORY[actor_id]
    while history and history[0] < now - window_s:
        history.popleft()
    if len(history) >= max_calls:
        return False
    history.append(now)
    return True
```

Si dépassement → retourner `{"error": "rate_limited", "retry_after_s": ...}`.

## Étape 4 — Tester

Script de test qui :

1. Appelle 15 fois `search_employees` rapidement → vérifier que les 5 derniers sont rate-limités
2. Tente `update_employee_email` avec un email malformé → vérifier rejet Pydantic
3. Inspecte `audit.jsonl` à la fin
