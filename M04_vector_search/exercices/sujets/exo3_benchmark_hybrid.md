# TP 3 — Benchmark complet : dense vs sparse vs hybride

## Contexte

Vous êtes en revue d'architecture devant le DSI. La question : « Pourquoi vous proposez de l'**hybride** ? Le dense seul ne suffit-il pas ? Ça a l'air complexe à opérer. »

Votre mission : **prouver chiffres en main** que l'hybride apporte un gain mesurable, et présenter un compromis défendable.

## Étape 1 — Préparer le golden set

À partir du corpus support technique (réutilisez `_corpus.py` du dossier `demos/`) :

Construire un golden set de **30 questions** réparties en 3 catégories :

- **10 paraphrases** : la question est une reformulation libre du chunk pertinent (avantage dense)
- **10 références techniques** : la question contient un code, ID, version exact (avantage sparse)
- **10 questions mixtes** : naturelles + un terme technique précis

Format JSON :

```json
[
  {"category": "paraphrase", "q": "...", "relevant": [12, 15]},
  ...
]
```

> Astuce : les "relevant" peuvent contenir 1-3 chunks. Vérifier humainement chaque cas.

## Étape 2 — Implémenter les 3 modes

- `search_dense(query, top_k)` : recherche Qdrant uniquement
- `search_sparse(query, top_k)` : recherche BM25 uniquement
- `search_hybrid(query, top_k, alpha=None)` : fusion RRF des deux

Bonus : implémenter aussi `search_hybrid_weighted(query, top_k, alpha)` qui combine les scores normalisés au lieu d'utiliser RRF.

## Étape 3 — Mesurer

Pour chaque mode et chaque catégorie, calculer :

- recall@5, recall@10
- MRR
- latence p50, p95

Stocker les résultats dans un DataFrame Pandas pour faciliter l'export.

## Étape 4 — Visualiser

Produire **2 graphiques** matplotlib :

1. Bar chart : recall@5 par mode et par catégorie
2. Bar chart : latence p95 par mode

Sauver en PNG.

## Étape 5 — Note de recommandation

Rédiger une note d'une demi-page contenant :

```markdown
## Conclusion benchmark retrieval (corpus support technique)

### Constats chiffrés

- [tableau 3 lignes (modes) × 5 colonnes (recall@5, recall@10, MRR, p50, p95)]

### Analyse par catégorie

- Paraphrases : [quel mode gagne, de combien]
- Références techniques : [...]
- Mixtes : [...]

### Recommandation

[1 phrase claire]

### Conditions de mise en prod

- [point 1, ex : index payload sur category]
- [point 2, ex : ef_search ≥ 100]
- [point 3, ex : monitoring du recall hebdo via re-run du benchmark]
```
