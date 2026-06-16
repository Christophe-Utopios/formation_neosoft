# TP 2 — Indexation production-ready

**Durée** : 1h30
**Modalité** : binôme recommandé
**Prérequis** : avoir terminé TP 1, Qdrant et corpus prêts

## Contexte

Vous reprenez le corpus Wikipédia FR du TP 1. Le PO de votre projet veut maintenant :

1. **Améliorer la pertinence** : mettre en place du **parent-document retrieval**
2. **Réduire la consommation mémoire** : activer la **quantization** scalaire int8
3. **Permettre des filtres temporels et thématiques** : ajouter des index payload riches

## Étape 1 — Restructurer en parent-document (30 min)

À partir de `exo1_solution.py` (votre solution du TP 1) :

1. **Parents** : 1 parent = 1 article Wikipédia complet (ou 800 tokens max)
2. **Children** : chaque parent est découpé en chunks de 200 tokens (overlap 30)
3. À l'indexation, chaque point Qdrant contient :
   - `vector` = embedding du child
   - `payload.child_content` = texte du child
   - `payload.parent_id` = id du parent (article)
   - `payload.parent_content` = texte du parent (la section ou article entier)
   - `payload.title` = titre de l'article
   - `payload.category` = catégorie déduite (voir étape 3)
   - `payload.indexed_at` = timestamp ISO

## Étape 2 — Activer la quantization (15 min)

À la création de la collection, ajouter :

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

quantization_config=ScalarQuantization(
    scalar=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        always_ram=True,
        quantile=0.99,
    )
)
```

Mesurer (via l'API Qdrant ou `du` sur le volume) :

- Taille de l'index avant quantization
- Taille de l'index après quantization
- Ratio de compression

## Étape 3 — Index payload riches (15 min)

Déduire une **catégorie** par article via les premières 200 caractères + classification simple basée sur mots-clés. Au moins 3 catégories : `personne`, `lieu`, `science`, `histoire`, `autre`.

Créer les index payload :

- `category` : KEYWORD
- `title` : KEYWORD (pour filtre exact)
- `indexed_at` : DATETIME
- `parent_id` : INTEGER

Tester un filtre combiné :

> Top-5 chunks pour « relativité » dans la catégorie `science` indexés depuis moins de 1 jour.
