# TP 3 — Comparaison LangGraph vs CrewAI

## Contexte

Le directeur technique veut **trancher** entre LangGraph et CrewAI pour un nouveau projet : un assistant **veille concurrentielle**.

L'agent doit :

1. Identifier 3 concurrents pour un produit donné
2. Synthétiser leurs forces / faiblesses
3. Produire un rapport markdown structuré

Vous allez implémenter cette tâche dans **les deux frameworks** et comparer.

## Étape 1 — Implémentation LangGraph

Dans `exo3_langgraph.py` :

- 2 nodes : `find_competitors` puis `synthesize_report`
- État avec `product: str`, `competitors: list[dict]`, `report: str`
- Pas de tool réel : LLM direct (Claude Haiku)

## Étape 2 — Implémentation CrewAI

Dans `exo3_crewai.py` :

- 2 agents : `Researcher`, `Writer`
- 2 tasks chaînées (`context=[research_task]`)
- Process séquentiel

> Si vous êtes coincés en Python 3.9 : commentez cette partie et concentrez-vous sur l'analyse.
