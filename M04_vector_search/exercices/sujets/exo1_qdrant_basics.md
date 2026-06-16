# TP 1 — Premier index Qdrant sur un corpus Wikipédia

**Durée** : 1h
**Modalité** : individuel
**Prérequis** : Qdrant tournant en local, Python avec `requirements.txt` installé

## Contexte

Vous démarrez un POC d'assistant documentaire pour un nouveau client. Avant d'attaquer le RAG complet, vous voulez **valider que la recherche vectorielle marche** sur un corpus francophone réaliste.

Vous allez :

1. Récupérer 1 000 articles Wikipédia français (extrait public Hugging Face)
2. Les indexer dans Qdrant avec un modèle multilingue
3. Lancer 5 requêtes types et analyser la qualité
4. Tuner `ef_search` pour observer le compromis recall / latence

## Étape 1 — Préparer le corpus (10 min)

À partir du dataset public `wikimedia/wikipedia` (config `20231101.fr`), prendre les **1 000 premiers articles**.

```python
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
articles = list(ds.take(1000))
# Chaque article a : id, url, title, text
```

Stocker dans une liste de dicts avec champs : `id`, `title`, `text`, `url`.

## Étape 2 — Découper et indexer (20 min)

À implémenter dans `exo1_starter.py` :

1. Découper chaque article en chunks d'environ **500 tokens** (utilisez `RecursiveCharacterTextSplitter`)
2. Embeddings : `intfloat/multilingual-e5-base` (préfixe `passage:`)
3. Créer une collection Qdrant `wikifr_demo` avec :
   - Distance cosine
   - Index payload sur `article_id` et `title`
4. Indexer tous les points avec leur payload

## Étape 3 — 5 requêtes types (15 min)

Lancer ces requêtes et analyser le top-5 retourné :

| #   | Requête                                 |
| --- | --------------------------------------- |
| 1   | « Qui était Napoléon Bonaparte ? »      |
| 2   | « Symptômes de la grippe saisonnière »  |
| 3   | « Inventeur du téléphone »              |
| 4   | « Capitale de l'Australie »             |
| 5   | « Théorie de la relativité d'Einstein » |

Pour chaque requête, noter :

- Le top-1 a-t-il une réponse plausible ?
- Y a-t-il des résultats clairement hors-sujet dans le top-5 ?
- Le score cosinus du top-1 est-il > 0.85 ?

## Livrable

Un notebook ou script `exo1_solution.py` qui :

- Reproduit l'indexation
- Affiche les 5 requêtes et leurs top-5
- Affiche le tableau ef_search vs latence vs top-5

Plus une **note d'observation** (5-10 lignes) répondant :

- Le modèle multilingue est-il assez bon pour du français Wikipédia ?
- À quel niveau d'`ef_search` voyez-vous un plateau ?
- Y a-t-il des cas où Qdrant retourne moins de 5 résultats ? Pourquoi ?

## Pièges à éviter

- ⚠️ Oublier le préfixe `passage:` à l'indexation → -10 points de recall
- ⚠️ Oublier le préfixe `query:` à la requête → asymétrie qui dégrade les scores
- ⚠️ Indexer les chunks **un par un** → très lent, batch par 100
- ⚠️ Tester sans warmup → les premières requêtes incluent le chargement du modèle
