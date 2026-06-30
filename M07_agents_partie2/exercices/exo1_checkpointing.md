# TP 1 — Checkpointing + reprise de conversation

## Contexte

Vous reprenez l'agent ReAct juridique du M5/M6. La direction veut que **les conversations persistent** entre les sessions des avocats (pause déjeuner, fermeture d'onglet, etc.).

Vous allez ajouter le **checkpointing SQLite** et démontrer la **reprise par thread_id**.

## Étape 1 — Adapter le graphe

Dans `exo1_starter.py` :

1. Compiler le graphe avec un `SqliteSaver`
2. Stocker la DB dans `conversations.db`
3. Veiller à ce que `recursion_limit` soit configuré

## Étape 2 — Simuler 3 utilisateurs

Pour chacun, alterner 2 messages dans le temps :

```
user_id="alice"   T0   : "Je m'appelle Alice, je travaille sur le dossier Martin."
user_id="bob"     T0   : "Bonjour, je m'occupe du contentieux Société X."
user_id="alice"   T+5m : "Rappelle-moi le nom du dossier que je traite."   <-- doit retrouver "Martin"
user_id="bob"     T+5m : "Quel client j'évoquais ?"                          <-- doit retrouver "Société X"
user_id="alice"   T+1h : "Récupère le 1er paragraphe de notre discussion."   <-- doit relire le message initial
```

Vérifier que chaque thread est isolé et que la mémoire est restaurée.

## Étape 3 — Lister l'historique d'un thread

Pour le thread d'Alice, afficher tous les snapshots :

```
Step 0 : 0 messages, next=('__start__',)
Step 1 : 1 messages, next=('llm',)
Step 2 : 2 messages, next=()
...
```

Puis afficher chronologiquement le contenu des messages pour vérifier que tout est bien stocké.
