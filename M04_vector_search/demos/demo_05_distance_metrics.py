from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

from _corpus import get_corpus

console = Console()
MODEL_NAME = "intfloat/multilingual-e5-small"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> None:
    console.print("\n[bold orange3]Demo 05 — Métriques de distance[/bold orange3]\n")

    model = SentenceTransformer(MODEL_NAME)
    corpus = get_corpus()

    # Génération des embeddings sans normalisation pour observer l'effet de chaque métrique
    texts = [f"passage: {c['content']}" for c in corpus]
    embeddings_raw = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    # Version normalisée (norme = 1) pour comparer
    embeddings_norm = embeddings_raw / np.linalg.norm(embeddings_raw, axis=1, keepdims=True)

    query = "Comment activer le SSO avec SAML 2.0 ?"
    query_emb_raw = model.encode(f"query: {query}", normalize_embeddings=False)
    query_emb_norm = query_emb_raw / np.linalg.norm(query_emb_raw)

    # Calcul du top-5 avec chaque métrique de distance sur vecteurs NON normalisés
    console.print(f"[cyan]Requête : {query}[/cyan]\n")

    # Calcul des similarités/distances pour chaque document
    cos_scores = [(i, cosine(query_emb_raw, embeddings_raw[i])) for i in range(len(corpus))]
    dot_scores = [(i, dot_product(query_emb_raw, embeddings_raw[i])) for i in range(len(corpus))]
    euc_scores = [(i, euclidean(query_emb_raw, embeddings_raw[i])) for i in range(len(corpus))]

    # Tri par similarité décroissante (cosine, dot) ou distance croissante (euclidean)
    cos_top5 = sorted(cos_scores, key=lambda x: -x[1])[:5]
    dot_top5 = sorted(dot_scores, key=lambda x: -x[1])[:5]
    euc_top5 = sorted(euc_scores, key=lambda x: x[1])[:5]  # plus petit = plus proche

    # Tableau comparatif
    table = Table(title="Top-5 selon chaque métrique (vecteurs NON normalisés)", show_lines=True)
    table.add_column("Rang", justify="center")
    table.add_column("Cosine", overflow="fold")
    table.add_column("Dot product", overflow="fold")
    table.add_column("Euclidean", overflow="fold")

    for rank in range(5):
        cos_idx, cos_score = cos_top5[rank]
        dot_idx, dot_score = dot_top5[rank]
        euc_idx, euc_score = euc_top5[rank]
        table.add_row(
            str(rank + 1),
            f"[{cos_score:.3f}] {corpus[cos_idx]['title']}",
            f"[{dot_score:.3f}] {corpus[dot_idx]['title']}",
            f"[{euc_score:.3f}] {corpus[euc_idx]['title']}",
        )
    console.print(table)

    # Test avec vecteurs normalisés : cosine et dot doivent donner le même ranking
    console.print("\n[cyan]Avec vecteurs normalisés (||v||=1) :[/cyan]")
    cos_n = [(i, cosine(query_emb_norm, embeddings_norm[i])) for i in range(len(corpus))]
    dot_n = [(i, dot_product(query_emb_norm, embeddings_norm[i])) for i in range(len(corpus))]
    cos_n_top5 = sorted(cos_n, key=lambda x: -x[1])[:5]
    dot_n_top5 = sorted(dot_n, key=lambda x: -x[1])[:5]

    # Vérification : si tous les index sont identiques, le ranking est le même
    same = all(c[0] == d[0] for c, d in zip(cos_n_top5, dot_n_top5))
    if same:
        console.print("[green]✓ Cosine et dot product produisent EXACTEMENT le même ranking[/green]")
        console.print("[green]  → Démontre qu'avec vecteurs normalisés, ce sont la même métrique[/green]")
    else:
        console.print("[yellow]⚠ Les rankings diffèrent (anomalie)[/yellow]")

    console.print("\n[bold]Observations pédagogiques :[/bold]")
    console.print("• Cosine est insensible à la magnitude → robuste face aux longueurs de texte")
    console.print("• Dot product favorise les vecteurs de grande magnitude (peut biaiser)")
    console.print("• Euclidean : peu pertinent pour le texte, à réserver à images/audio")
    console.print("• Si vous normalisez vos vecteurs : utilisez dot product, c'est plus rapide à calculer\n")

if __name__ == "__main__":
  main()