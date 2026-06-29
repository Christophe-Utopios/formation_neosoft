# TP 2 — Routing dynamique avec conditional edges

## Contexte

Vous reprenez l'agent du TP 1. Le directeur du courtier veut **séparer** les types de demandes :

- **Question informationnelle** (« qui est X ? », « quels contrats ? ») → réponse directe via tools
- **Demande de devis** (« crée un devis pour... ») → étape de validation supplémentaire avant de créer
- **Demande hors-périmètre** (« quel temps fait-il ? », « comment résilier ? ») → escalade au humain

Vous allez construire un **StateGraph manuel** avec 3 branches.

## Étape 1 — Définir le State

Dans `exo2_starter.py`, compléter la classe `State` :

```python
class State(TypedDict):
    question: str
    intent: Optional[str]   # "info", "quote", "out_of_scope"
    answer: Optional[str]
    needs_validation: bool
    path_taken: List[str]
```

## Étape 2 — Implémenter les 4 nodes

1. `classify_intent(state)` : LLM classifieur, met à jour `state.intent`
2. `handle_info(state)` : utilise `find_client`, `get_client_contracts` pour répondre
3. `handle_quote(state)` : appelle `is_product_available`, met `needs_validation=True`
4. `handle_out_of_scope(state)` : message d'escalade

Chaque node doit ajouter son nom à `state.path_taken`.

## Étape 3 — Conditional edges

```python
def route(state: State) -> str:
    return {
        "info": "handle_info",
        "quote": "handle_quote",
        "out_of_scope": "handle_out_of_scope",
    }[state["intent"]]

graph.add_conditional_edges("classify", route, {...})
```

## Étape 4 — Tester

```python
QUESTIONS = [
    "Quels sont les contrats de Jean Dupont ?",
    "Crée-moi un devis pour Sophie Martin sur l'Auto Tous Risques",
    "Vous travaillez le dimanche ?",
]
```

Pour chaque question :

- Afficher le `path_taken` final
- Afficher la réponse
- Vérifier que pour le devis, `needs_validation` est bien `True`

## Pièges

- Oublier `Optional[X]` à la place de `X | None` (compat Python 3.9 / LangGraph)
- Renvoyer `state` au lieu de `{"key": value}` → réécrit toute la map
- `path_taken` doit être étendu, pas écrasé — utiliser `state["path_taken"] + [...]`
