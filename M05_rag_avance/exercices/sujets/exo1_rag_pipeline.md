# TP 1 — Pipeline RAG advanced sur corpus juridique

## Contexte

Vous travaillez pour un cabinet d'avocats qui veut **un assistant** pour interroger leur base de jurisprudence interne. Les avocats posent des questions en langage naturel, parfois imprécises, et veulent des réponses **citant systématiquement leurs sources**.

Vous allez construire un pipeline RAG complet sur un corpus juridique simplifié (Code civil français, premiers articles).

## Étape 1 — Préparer le corpus (10 min)

À partir du dataset `louisbrulenaudet/code-civil` sur Hugging Face (libre d'usage), prendre **300 articles en vigueur** :

```python
import re
from datasets import load_dataset

ds = load_dataset("louisbrulenaudet/code-civil", split="train")
articles = []
for row in ds:
    texte = re.sub(r"\s+", " ", (row.get("texte") or "")).strip()
    if texte and row.get("etat") == "VIGUEUR":            # ne garder que le droit en vigueur
        articles.append({
            "article_id": f"article_{row['num']}",         # ex : "article_414"
            "content": texte,
            "ref": row.get("ref"),                         # ex : "Code civil, art. 414"
        })
    if len(articles) >= 300:
        break
# Champs disponibles : texte, texteHtml, num, ref, etat, dateDebut, ...
```

Indexer dans Qdrant avec :

- Modèle d'embedding `intfloat/multilingual-e5-base`
- Index payload sur `article_id` (string)
- Préfixe `passage:` pour les documents

## Étape 2 — Implémenter le pipeline (30 min)

1. **Query rewriting** : reformuler la question utilisateur en requête juridique formelle
2. **Hybrid retrieval** : dense (Qdrant) + sparse (BM25) → top-20 fusionné par RRF
3. **Re-ranking** : `BAAI/bge-reranker-v2-m3` sur top-20 → top-5
4. **Génération avec citations** : prompt qui force `[article_X]` pour chaque affirmation

Exigences :

- Une fonction `pipeline(query: str) -> dict` qui retourne `{"answer": str, "sources": list[str]}`
- Si une question est hors sujet juridique, la fonction doit le détecter et renvoyer une réponse standard

## Étape 3 — Tester sur 5 questions (10 min)

```python
QUESTIONS = [
    "À partir de quel âge on est majeur en France ?",
    "Comment se passe un divorce par consentement mutuel ?",
    "Combien de temps dure un mariage avant le divorce automatique ?",  # piège
    "Quels sont les droits des enfants adoptés ?",
    "Combien coûte une voiture neuve en 2026 ?",  # hors sujet
]
```

Pour chaque question, afficher :

- La question reformulée
- Les top-5 articles
- La réponse avec citations
- Les sources finalement utilisées

## Étape 4 — Commenter (10 min)

Notes d'observations (5-10 lignes) :

- Le rewriting transforme-t-il bien les questions naïves ?
- Le re-ranker change-t-il l'ordre des candidats ?
- Les citations sont-elles précises (le LLM cite bien les bons articles) ?
- La question piège (« divorce automatique ») et la question hors-sujet sont-elles bien gérées ?
