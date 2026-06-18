from __future__ import annotations

import os

from anthropic import Anthropic
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.table import Table

from _setup import get_or_create_index, search_dense

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-4-6"


# Golden set : 5 questions avec réponse de référence (ground truth)
GOLDEN_SET = [
    {
        "question": "Comment configurer SAML 2.0 sur NovaCloud version 3.2 ?",
        "ground_truth": "Pour configurer SAML 2.0 sur NovaCloud version 3.2, il faut aller dans Paramètres > Authentification > SAML, et renseigner l'URL de l'IdP, le certificat X.509 et l'ID de l'entité.",
    },
    {
        "question": "Quelle est la limite de débit par défaut sur l'API ?",
        "ground_truth": "La limite de débit par défaut est de 100 requêtes par minute par token, et 1000 requêtes par minute par organisation.",
    },
    {
        "question": "Combien de temps les sauvegardes sont-elles conservées ?",
        "ground_truth": "Les sauvegardes sont conservées 30 jours en standard, et 90 jours sur l'offre Enterprise.",
    },
    {
        "question": "Quel chiffrement est utilisé pour les données ?",
        "ground_truth": "AES-256 au repos via S3 SSE-KMS, et TLS 1.3 en transit. Customer-Managed Keys disponibles sur Enterprise.",
    },
    {
        "question": "Comment intégrer NovaCloud à Slack ?",
        "ground_truth": "Aller dans Paramètres > Intégrations > Slack, cliquer sur Add to Slack, autoriser l'app, choisir les canaux destinataires des alertes.",
    },
]

NAIVE_PROMPT = """Réponds à la question en t'appuyant sur le contexte.

Contexte : {context}

Question : {question}

Réponse :"""

REWRITE_PROMPT = """Reformule cette question utilisateur en UNE requête de recherche claire,
incluant termes techniques pertinents. Réponds uniquement avec la question reformulée.

Question : {question}"""


def naive_rag(question: str, client, model, llm: Anthropic) -> dict:
    """RAG naïf : recherche directe sans transformation."""
    # Baseline : question brute → retrieval → génération
    hits = search_dense(client, model, question, top_k=5)
    contexts = [h.payload["content"] for h in hits]
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": NAIVE_PROMPT.format(context="\n\n".join(contexts), question=question)}],
    )
    return {"answer": msg.content[0].text.strip(), "contexts": contexts}


def rewriting_rag(question: str, client, model, llm: Anthropic) -> dict:
    """RAG avec query rewriting : reformule avant la recherche."""
    # Variante améliorée : rewriting améliore la qualité du retrieval
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=120, temperature=0,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
    )
    rewritten = msg.content[0].text.strip()
    hits = search_dense(client, model, rewritten, top_k=5)
    contexts = [h.payload["content"] for h in hits]
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": NAIVE_PROMPT.format(context="\n\n".join(contexts), question=question)}],
    )
    return {"answer": msg.content[0].text.strip(), "contexts": contexts}


def build_dataset(rag_fn, client, model, llm: Anthropic) -> Dataset:
    """Construit le dataset Ragas avec question, answer, contexts, ground_truth.

    Format requis par Ragas :
    - question : str - Question utilisateur
    - answer : str - Réponse générée par le RAG
    - contexts : list[str] - Chunks récupérés
    - ground_truth : str - Réponse de référence correcte
    """
    rows = []
    for case in GOLDEN_SET:
        result = rag_fn(case["question"], client, model, llm)
        rows.append({
            "question": case["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": case["ground_truth"],
        })
    return Dataset.from_list(rows)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant.[/red]")
        return

    console.print("\n[bold orange3]Demo 06 — Ragas sur 2 variantes RAG[/bold orange3]\n")
    client, model, _ = get_or_create_index()
    llm = Anthropic()

    # Configuration Ragas : LLM juge + embeddings pour les métriques
    # LLM juge : évalue la qualité des réponses (Sonnet 4 pour plus de précision)
    judge_llm = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL, temperature=0))
    # Embeddings : calcule la similarité sémantique pour answer_relevancy
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    )
    # Métriques Ragas :
    # - faithfulness : réponse fidèle au contexte (pas d'hallucination)
    # - answer_relevancy : réponse pertinente pour la question
    # - context_precision : chunks récupérés sont pertinents (peu de bruit)
    # - context_recall : chunks contiennent toute l'info nécessaire
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    console.print("[cyan]Variante 1 — Naïve RAG : génération du dataset...[/cyan]")
    # Exécute le RAG naïf sur chaque question du golden set
    ds_naive = build_dataset(naive_rag, client, model, llm)
    console.print("[cyan]Évaluation Ragas (peut prendre 1-2 min)...[/cyan]")
    # Ragas appelle le LLM juge pour chaque métrique sur chaque exemple
    res_naive = evaluate(ds_naive, metrics=metrics, llm=judge_llm, embeddings=embeddings)

    console.print("\n[cyan]Variante 2 — RAG avec query rewriting : génération...[/cyan]")
    # Même processus avec la variante améliorée
    ds_rw = build_dataset(rewriting_rag, client, model, llm)
    console.print("[cyan]Évaluation Ragas...[/cyan]")
    res_rw = evaluate(ds_rw, metrics=metrics, llm=judge_llm, embeddings=embeddings)

    # Comparaison — moyenne des scores par métrique via to_pandas()
    # to_pandas() convertit les résultats Ragas en DataFrame pour faciliter l'analyse
    df_naive = res_naive.to_pandas()
    df_rw = res_rw.to_pandas()

    table = Table(title="Comparaison Ragas — naïve vs rewriting", show_lines=True)
    table.add_column("Métrique", style="cyan")
    table.add_column("Naïve", justify="right")
    table.add_column("+Rewriting", justify="right", style="bold orange3")
    table.add_column("Δ", justify="right")

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for m_name in metric_names:
        v1 = float(df_naive[m_name].dropna().mean()) if m_name in df_naive.columns else 0.0
        v2 = float(df_rw[m_name].dropna().mean()) if m_name in df_rw.columns else 0.0
        delta = v2 - v1
        sign = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        table.add_row(m_name, f"{v1:.3f}", f"{v2:.3f}", f"{sign} {delta:+.3f}")
    console.print(table)

    console.print("\n[bold]Lecture pédagogique :[/bold]")
    console.print("• Faithfulness > 0.85 → peu d'hallucinations")
    console.print("• Answer relevancy > 0.85 → réponses qui adressent bien la question")
    console.print("• Context precision élevée → peu de bruit dans le top-5")
    console.print("• Context recall élevé → infos nécessaires bien récupérées\n")


if __name__ == "__main__":
    main()
