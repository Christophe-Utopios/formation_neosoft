# TP 1 — Premier agent LangGraph multi-tools

## Contexte

Un courtier en assurance vous demande un assistant pour ses agents commerciaux. L'assistant doit pouvoir :

1. Chercher un **client** par nom ou email
2. Récupérer la liste de ses **contrats**
3. Vérifier la **disponibilité** d'un produit pour un client (selon ses caractéristiques)
4. **Créer un devis** pour un produit donné

Vous allez l'implémenter avec **LangGraph + create_react_agent**, en partant du mock CRM fourni dans `_mock_insurance.py`.

## Étape 1 — Lire le mock

Ouvrir `_mock_insurance.py`. Comprendre les structures `CLIENTS`, `CONTRACTS`, `PRODUCTS` et les fonctions exposées. Ne pas modifier ce fichier.

## Étape 2 — Définir les tools

Compléter `exo1_starter.py` en transformant chaque fonction du mock en `@tool` LangChain :

```python
@tool
def find_client(query: str) -> dict:
    """Recherche un client par nom ou email. Retourne ses informations.
    Args:
        query: nom partiel ou email exact du client
    """
    ...
```

⚠️ Les **docstrings** sont essentielles : c'est ce qui est envoyé au LLM pour qu'il sache quel tool appeler. Soyez précis.

## Étape 3 — Construire l'agent

Configurer un **system prompt** approprié : le rôle de l'agent (commercial assurance), le ton attendu, l'interdiction de halluciner les tarifs.

## Étape 4 — Tester sur 4 questions

```python
QUESTIONS = [
    "Bonjour, peux-tu me dire qui est Jean Dupont et quels sont ses contrats ?",
    "Pour la cliente carole.dubois@example.com, prépare un devis pour le produit 'Multirisque Habitation'.",
    "Le produit 'Auto Premium' est-il dispo pour tous nos clients ?",
    "Crée un devis pour Maxime Lemoine sur le produit 'Santé Famille'.",
]
```

Pour chaque question, **streamer** les étapes avec `agent.stream(...)` et afficher :

- Les `tool_calls` faits par l'agent
- Les `tool` results obtenus
- La réponse finale `ai`

## Étape 5 — Garde-fou : `max_iterations`

Ajouter une limite explicite pour éviter les boucles infinies.

Tester avec une question piège qui force l'agent à boucler (`"Liste tous les clients un par un"` par exemple).

## Pièges

- Type hints `int | None` invalide en Python 3.9 → utiliser `Optional[int]`
- Tool sans docstring → le LLM ne saura pas quand l'appeler
- Tool qui retourne un objet non-sérialisable → erreur opaque, retourner toujours dict/str
- Oublier de mettre `temperature=0` → variance entre runs, debug impossible
