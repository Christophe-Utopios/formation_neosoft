# TP 2 — Agent LangGraph + serveur MCP

## Contexte

Vous avez votre serveur MCP RH (TP 1). Construisez maintenant un **agent LangGraph** qui l'utilise pour répondre à des questions des managers.

## Étape 1 — Configurer le client MCP

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "hr": {
        "command": sys.executable,
        "args": [str(SERVER_PATH)],  # chemin vers exo1_solution.py
        "transport": "stdio",
    }
})
tools = await client.get_tools()
```

## Étape 2 — Créer l'agent

```python
agent = create_react_agent(
    ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0),
    tools=tools,
    prompt="Tu es un assistant RH... [rédiger un system prompt clair]",
)
```

## Étape 3 — Tester sur 4 questions

Streamer les étapes pour chacune :

```python
QUESTIONS = [
    "Combien d'employés y a-t-il dans le département Engineering ?",
    "Qui est Maxime Lemoine et dans quel département ?",
    "Change l'email de Jules Bernard vers j.bernard@nouveau.fr",
    "Donne-moi la liste des employés du département Sales avec leurs emails.",
]
```

Vérifier que :

- L'agent choisit le bon tool selon la question
- Une question avec email malformé est rejetée par Pydantic
- Les chaînages multi-step fonctionnent (search → get → update)

## Étape 4 — Streaming UX

Afficher les tool calls en temps réel via `agent.astream(...)`.
