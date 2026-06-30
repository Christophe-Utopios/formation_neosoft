# TP 2 — Human-in-the-Loop avec modification de l'action

## Contexte

Vous travaillez sur un agent **back-office RH**. L'agent peut :

1. Trouver un employé par nom (`find_employee`)
2. Demander une suppression de compte (`request_account_deletion`)

**Toute suppression doit être validée par le manager RH**, qui peut :

- ✅ Approuver
- ✏️ Modifier les arguments (ex : changer la raison)
- ❌ Rejeter

Vous allez implémenter ce flux en **LangGraph + HIL**.

## Étape 1 — Définir le State

```python
class State(TypedDict):
    user_request: str
    employee: Optional[dict]          # résultat de find_employee
    pending_action: Optional[dict]    # action proposée
    approval: Optional[str]           # "approved" | "rejected" | "modified"
    result: Optional[str]
```

## Étape 2 — Implémenter les nodes

- `find_employee(state)` : recherche par nom (mock dans `_mock_hr.py` fourni)
- `draft_deletion(state)` : prépare l'action JSON `{tool, args, reason}`
- `execute_deletion(state)` : exécute après validation
- `cancel(state)` : message d'annulation

## Étape 3 — interrupt_before

Compiler avec :

```python
graph.compile(checkpointer=..., interrupt_before=["execute_deletion"])
```

## Étape 4 — Workflow de validation

```python
# 1. Lancer jusqu'à interrupt
result = agent.invoke({"user_request": "Supprime le compte de Maxime Lemoine, départ entreprise"}, config)

# 2. Afficher la pending_action et demander une décision
pending = result["pending_action"]
print(pending)

decision = input("Approuver (a), Rejeter (r), Modifier (m) ? ")

if decision == "m":
    pending["reason"] = input("Nouvelle raison ? ")
    agent.update_state(config, {"pending_action": pending, "approval": "modified"})
elif decision == "r":
    agent.update_state(config, {"approval": "rejected"})
else:
    agent.update_state(config, {"approval": "approved"})

# 3. Reprendre
final = agent.invoke(None, config)
```

Adapter le node `execute_deletion` pour lire `state["approval"]` et brancher vers `cancel` si "rejected".
