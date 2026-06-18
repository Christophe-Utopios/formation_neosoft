# TP 2 — Évaluation Ragas et comparaison de 3 prompts

## Contexte

Vous reprenez le pipeline du TP 1. Le directeur juridique du cabinet veut **valider la qualité** avant de mettre en production, et **comparer 3 versions de prompt système** pour choisir le meilleur.

## Étape 1 — Construire un golden set (20 min)

À partir du corpus juridique du TP 1, construire **15 paires** :

```json
{
  "question": "À partir de quel âge un Français est-il majeur ?",
  "ground_truth": "La majorité civile en France est fixée à 18 ans selon l'article 388 du Code civil.",
  "expected_articles": ["article_388"]
}
```

```python
GOLDEN = [
    # Paraphrases naturelles
    {"question": "À partir de quel âge un Français est-il majeur ?",
     "ground_truth": "La majorité civile en France est fixée à 18 ans selon l'article 388 du Code civil."},
    {"question": "À partir de quand peut-on signer un contrat seul ?",
     "ground_truth": "Une personne peut contracter dès sa majorité, soit 18 ans, sauf incapacité spécifique."},
    {"question": "Combien de témoins faut-il pour se marier ?",
     "ground_truth": "Le mariage civil nécessite la présence de 2 à 4 témoins."},
    {"question": "Mon enfant est né, qu'est-ce que je dois faire ?",
     "ground_truth": "Une déclaration de naissance doit être faite à l'état civil dans les 5 jours suivant la naissance."},
    {"question": "Mon conjoint est décédé, comment partager les biens ?",
     "ground_truth": "Le partage des biens dépend du régime matrimonial et de la présence d'enfants ; à défaut de testament, la succession suit les règles légales."},

    # Termes juridiques précis
    {"question": "Que prévoit l'article 388 du Code civil ?",
     "ground_truth": "L'article 388 du Code civil définit le mineur comme l'individu de l'un ou l'autre sexe qui n'a point encore l'âge de dix-huit ans accomplis."},
    {"question": "Définition juridique de l'usufruit ?",
     "ground_truth": "L'usufruit est le droit de jouir des choses dont un autre a la propriété, à charge d'en conserver la substance."},
    {"question": "Régime de la communauté légale, principe ?",
     "ground_truth": "À défaut de contrat de mariage, les époux sont soumis au régime de la communauté réduite aux acquêts."},
    {"question": "Quel est le délai de prescription civile de droit commun ?",
     "ground_truth": "Le délai de prescription extinctive de droit commun est de 5 ans à compter du jour où le titulaire a connu les faits."},
    {"question": "Distinction entre meubles et immeubles ?",
     "ground_truth": "Les biens sont meubles ou immeubles selon leur nature, leur destination ou leur objet."},

    # Multi-articles
    {"question": "Comment fonctionne la tutelle d'un mineur ?",
     "ground_truth": "La tutelle est ouverte lorsqu'un mineur est privé de ses parents ; elle est exercée par un tuteur sous la surveillance du conseil de famille et du juge des tutelles."},
    {"question": "Quels sont les droits successoraux du conjoint survivant ?",
     "ground_truth": "Le conjoint survivant a vocation successorale ; ses droits varient selon la présence d'enfants ou d'autres héritiers, allant de l'usufruit total à la pleine propriété."},
    {"question": "Comment procéder à une adoption simple ?",
     "ground_truth": "L'adoption simple nécessite une requête au tribunal judiciaire, le consentement des parties, et est prononcée si elle correspond à l'intérêt de l'adopté."},
    {"question": "Quels sont les effets d'un PACS ?",
     "ground_truth": "Le PACS crée une vie commune, une aide mutuelle, des droits sociaux ; les biens sont en principe séparés sauf indivision déclarée."},
    {"question": "Comment se passe une séparation de corps ?",
     "ground_truth": "La séparation de corps suit les mêmes règles que le divorce et entraîne le relâchement du lien matrimonial sans le rompre."},
]
```

Couvrir 5 paraphrases naturelles, 5 questions à terminologie juridique précise, 5 questions multi-articles.

Stocker dans `golden.jsonl` (un JSON par ligne).

## Étape 2 — Définir 3 variantes de prompt (15 min)

Dans un fichier `prompts.py`, déclarer 3 variantes :

- **Prompt A — Strict** : forcer citation obligatoire, refuser si pas dans contexte
- **Prompt B — Pédagogique** : citer + expliquer en termes simples
- **Prompt C — Avocat** : ton formel, langage juridique soutenu

Chaque variante est une fonction qui prend `question`, `context` et retourne le prompt complet.

## Étape 3 — Évaluer avec Ragas (30 min)

Pour chaque variante :

1. Générer les réponses sur les 15 questions du golden
2. Lancer Ragas sur les 4 métriques :
   - faithfulness
   - answer_relevancy
   - context_precision
   - context_recall

3. Stocker les résultats dans `results_<variant>.csv`

Le LLM-juge doit être plus fort que le LLM générateur :

- Générateur : `claude-haiku-4-5`
- Juge : `claude-sonnet-4-6`

## Étape 4 — Analyser et choisir (15 min)

Construire un tableau comparatif :

| Métrique          | Prompt A | Prompt B | Prompt C | Gagnant |
| ----------------- | -------- | -------- | -------- | ------- |
| faithfulness      | ?        | ?        | ?        | ?       |
| answer_relevancy  | ?        | ?        | ?        | ?       |
| context_precision | ?        | ?        | ?        | ?       |
| context_recall    | ?        | ?        | ?        | ?       |

Identifier :

- Le prompt avec la **meilleure faithfulness** (priorité réglementaire)
- Le prompt avec le meilleur **équilibre global**
- Les cas où chaque prompt **échoue** (analyser 2-3 réponses qualitativement)
