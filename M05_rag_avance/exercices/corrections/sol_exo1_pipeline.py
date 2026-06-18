from __future__ import annotations

import json
import re
from pathlib import Path

from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

COLLECTION = "tp1_juris"
EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANKER = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "claude-haiku-4-5-20251001"

REWRITE_PROMPT = """Reformule cette question utilisateur en UNE requête de recherche
juridique formelle et précise. Utilise le vocabulaire du Code civil français.
Si la question est manifestement hors champ juridique, retourne exactement: HORS_SUJET

Question : {question}

Reformulation :"""

GENERATE_PROMPT = """Tu es un assistant juridique pour un cabinet d'avocats français.
Tu réponds en t'appuyant UNIQUEMENT sur les articles ci-dessous.

Pour chaque affirmation, cite l'article entre crochets : [article_X].
Si la réponse n'est pas dans le contexte, réponds : "Information non disponible dans les articles fournis."

Articles disponibles :
{context}

Question : {question}

Réponse (avec citations [article_X] obligatoires) :"""


HF_DATASET = "louisbrulenaudet/code-civil"
FALLBACK_JSON = Path(__file__).resolve().parents[1] / "exercises" / "juris_extract.json"


def load_corpus(n: int = 300) -> list[dict]:
    """Charge n articles du Code civil français.

    Source HF `louisbrulenaudet/code-civil` (schéma LEGI : `texte`, `num`,
    `ref`, `etat`). On ne garde que les articles en vigueur. En cas de blocage
    réseau, repli sur l'extrait local `juris_extract.json` (mêmes données).
    """
    try:
        ds = load_dataset(HF_DATASET, split="train")
        records: list[dict] = []
        for row in ds:
            content = re.sub(r"\s+", " ", (row.get("texte") or "")).strip()
            if not content or row.get("etat") != "VIGUEUR":
                continue
            num = str(row.get("num") or "")
            records.append(
                {
                    "article_id": f"article_{num}" if num else row["id"],
                    "content": content,
                    "ref": row.get("ref"),  # ex : "Code civil, art. 414"
                }
            )
            if len(records) >= n:
                break
        if records:
            return records
        raise ValueError("dataset vide")
    except Exception as e:  # réseau bloqué, dataset déplacé, etc.
        console.print(f"[yellow]Chargement HF indisponible ({e}), repli local[/yellow]")
        return json.loads(FALLBACK_JSON.read_text(encoding="utf-8"))[:n]


def index_corpus(corpus: list[dict], model: SentenceTransformer) -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = model.get_sentence_embedding_dimension()
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    client.create_payload_index(COLLECTION, "article_id", PayloadSchemaType.KEYWORD)

    texts = [f"passage: {c['content']}" for c in corpus]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=corpus[i])
        for i in range(len(corpus))
    ]
    for start in range(0, len(points), 100):
        client.upsert(COLLECTION, points=points[start:start + 100], wait=False)
    client.update_collection(COLLECTION, optimizer_config={"indexing_threshold": 0})
    return client


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def rewrite_query(question: str, llm: Anthropic) -> str:
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=120, temperature=0,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
    )
    return msg.content[0].text.strip()


def hybrid_retrieve(rewritten: str, model: SentenceTransformer, client: QdrantClient,
                    bm25: BM25Okapi, corpus: list[dict], top_k: int = 20) -> list[dict]:
    emb = model.encode(f"query: {rewritten}", normalize_embeddings=True)
    dense = client.query_points(COLLECTION, query=emb.tolist(), limit=top_k, with_payload=True).points
    dense_ids = [h.id for h in dense]

    sparse_scores = bm25.get_scores(tokenize(rewritten))
    sparse_top = sorted(range(len(corpus)), key=lambda i: -sparse_scores[i])[:top_k]

    # RRF fusion
    scores: dict[int, float] = {}
    for ranking in [dense_ids, sparse_top]:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
    fused_idx = [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
    return [corpus[i] for i in fused_idx]


def rerank(query: str, candidates: list[dict], reranker: CrossEncoder, top_k: int = 5) -> list[dict]:
    pairs = [(query, c["content"][:1500]) for c in candidates]  # truncate pour max_length
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]


def generate(question: str, top5: list[dict], llm: Anthropic) -> str:
    context = "\n\n".join(f"[{c['article_id']}] {c['content']}" for c in top5)
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=500, temperature=0,
        messages=[{"role": "user", "content": GENERATE_PROMPT.format(context=context, question=question)}],
    )
    return msg.content[0].text.strip()


def pipeline(query: str, *, model, client, reranker, bm25, corpus, llm) -> dict:
    rewritten = rewrite_query(query, llm)
    if "HORS_SUJET" in rewritten:
        return {
            "answer": "Cette question sort du périmètre juridique du Code civil français.",
            "sources": [],
            "rewritten": rewritten,
        }

    top20 = hybrid_retrieve(rewritten, model, client, bm25, corpus)
    top5 = rerank(rewritten, top20, reranker)
    answer = generate(query, top5, llm)
    sources = [c["article_id"] for c in top5]

    return {"answer": answer, "sources": sources, "rewritten": rewritten}


def main() -> None:
    console.print("\n[bold orange3]Solution TP 1 — Pipeline RAG juridique[/bold orange3]\n")

    corpus = load_corpus()
    console.print(f"[green]✓ {len(corpus)} articles chargés[/green]")

    model = SentenceTransformer(EMBED_MODEL)
    client = index_corpus(corpus, model)
    console.print("[green]✓ Corpus indexé dans Qdrant[/green]")

    reranker_model = CrossEncoder(RERANKER, max_length=512)
    bm25 = BM25Okapi([tokenize(c["content"]) for c in corpus])
    llm = Anthropic()

    questions = [
        "À partir de quel âge on est majeur en France ?",
        "Comment se passe un divorce par consentement mutuel ?",
        "Combien de temps dure un mariage avant le divorce automatique ?",
        "Quels sont les droits des enfants adoptés ?",
        "Combien coûte une voiture neuve en 2026 ?",
    ]

    for q in questions:
        console.print(f"\n[bold cyan]Q : {q}[/bold cyan]")
        result = pipeline(q, model=model, client=client, reranker=reranker_model,
                          bm25=bm25, corpus=corpus, llm=llm)
        console.print(f"[orange3]→ Reformulé : {result['rewritten']}[/orange3]")
        console.print(Panel(result["answer"], title="Réponse", border_style="orange3"))
        console.print(f"[dim]Sources : {result['sources']}[/dim]")


if __name__ == "__main__":
    main()
