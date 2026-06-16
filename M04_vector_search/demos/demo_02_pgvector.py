from __future__ import annotations

import time
import psycopg
import numpy as np
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

from _corpus import get_corpus

console = Console()
MODEL_NAME = "intfloat/multilingual-e5-small"
CONN_STRING = "postgresql://postgres:postgres@localhost:5432/postgres"


def main() -> None:
    console.print("\n[bold orange3]Demo 03 — pgvector[/bold orange3]\n")

    model = SentenceTransformer(MODEL_NAME)
    corpus = get_corpus()

    with psycopg.connect(CONN_STRING, autocommit=True) as conn:
        # 1. Setup — activation de l'extension pgvector dans PostgreSQL
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Enregistrement du type vector pour psycopg (permet l'utilisation du type vector)
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS demo_chunks")
            # Création de la table avec une colonne de type vector(384) pour les embeddings
            cur.execute("""
                CREATE TABLE demo_chunks (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    product TEXT,
                    version TEXT,
                    category TEXT,
                    title TEXT
                )
            """)
        console.print("[green]✓ Table demo_chunks créée[/green]")

        # 2. Indexation des documents dans PostgreSQL
        console.print(f"[cyan]Indexation de {len(corpus)} chunks...[/cyan]")
        texts = [f"passage: {c['content']}" for c in corpus]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        with conn.cursor() as cur:
            for i, c in enumerate(corpus):
                cur.execute(
                    """INSERT INTO demo_chunks (id, content, embedding, product, version, category, title)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (c["id"], c["content"], embeddings[i], c["product"], c["version"], c["category"], c["title"]),
                )
        console.print(f"[green]✓ {len(corpus)} lignes insérées[/green]\n")

        # 3. Recherche AVANT index HNSW (force brute = scan complet)
        query = "Comment configurer SAML 2.0 ?"
        query_emb = model.encode(f"query: {query}", normalize_embeddings=True)

        console.print(f"[cyan]Requête : {query}[/cyan]")
        console.print("[dim]Sans index HNSW (sequential scan) :[/dim]")
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            # Opérateur <=> : distance cosine (1 - cosine_similarity)
            cur.execute(
                """SELECT id, title, category, 1 - (embedding <=> %s::vector) AS similarity
                   FROM demo_chunks
                   ORDER BY embedding <=> %s::vector
                   LIMIT 5""",
                (query_emb, query_emb),
            )
            rows = cur.fetchall()
            t_brute = (time.perf_counter() - t0) * 1000

        for r in rows:
            console.print(f"  [{r[3]:.3f}] {r[1]} ({r[2]})")
        console.print(f"[dim]Latence sans index : {t_brute:.1f} ms[/dim]\n")

        # 4. Création de l'index HNSW pour accélérer la recherche approximative
        # HNSW = Hierarchical Navigable Small World (graphe de proximité)
        console.print("[cyan]Création de l'index HNSW...[/cyan]")
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            # m=16 : connexions par nœud, ef_construction=64 : précision à la construction
            cur.execute(
                "CREATE INDEX ON demo_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
            t_index = (time.perf_counter() - t0) * 1000
        console.print(f"[green]✓ Index HNSW créé en {t_index:.0f} ms[/green]\n")

        # 5. Recherche APRÈS index (recherche approximative rapide)
        console.print("[dim]Avec index HNSW :[/dim]")
        with conn.cursor() as cur:
            # ef_search : nombre de candidats explorés (plus élevé = plus précis)
            cur.execute("SET hnsw.ef_search = 100")
            t0 = time.perf_counter()
            cur.execute(
                """SELECT id, title, category, 1 - (embedding <=> %s::vector) AS similarity
                   FROM demo_chunks
                   ORDER BY embedding <=> %s::vector
                   LIMIT 5""",
                (query_emb, query_emb),
            )
            rows = cur.fetchall()
            t_index_search = (time.perf_counter() - t0) * 1000

        for r in rows:
            console.print(f"  [{r[3]:.3f}] {r[1]} ({r[2]})")
        console.print(f"[dim]Latence avec index : {t_index_search:.1f} ms[/dim]\n")

        # 6. Recherche avec filtre métier (point fort de pgvector : SQL natif)
        console.print("[bold orange3]Cas typique pgvector : JOIN + filtre + similarity[/bold orange3]\n")
        console.print("[dim]Requête : seulement les docs auth ET version 3.2[/dim]")
        with conn.cursor() as cur:
            # Combinaison naturelle : WHERE SQL + ORDER BY vectoriel
            cur.execute(
                """SELECT id, title, category, 1 - (embedding <=> %s::vector) AS similarity
                   FROM demo_chunks
                   WHERE category = 'auth' AND version = '3.2'
                   ORDER BY embedding <=> %s::vector
                   LIMIT 5""",
                (query_emb, query_emb),
            )
            rows = cur.fetchall()
        for r in rows:
            console.print(f"  [{r[3]:.3f}] {r[1]} ({r[2]})")

        # Synthèse
        console.print("\n[bold]Avantages pgvector observés ici :[/bold]")
        console.print("• Le SQL filter + ORDER BY se compose naturellement")
        console.print("• Transactions ACID, JOINs avec d'autres tables métier")
        console.print("• Une seule base à opérer si on a déjà PostgreSQL")
        console.print(f"• Speedup HNSW : sur 50 chunks c'est négligeable, mais pour 1M+ c'est x100\n")


if __name__ == "__main__":
    main()
