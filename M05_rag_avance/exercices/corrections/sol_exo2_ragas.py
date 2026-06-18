from __future__ import annotations

import json
import sys
from pathlib import Path

from anthropic import Anthropic
from datasets import Dataset
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy, context_precision, context_recall, faithfulness,
)
from rich.console import Console
from rich.table import Table

# Réutilise le pipeline de la solution TP1
sys.path.insert(0, str(Path(__file__).parent))
from sol_exo1_pipeline import (
    EMBED_MODEL, LLM_MODEL, RERANKER,
    hybrid_retrieve, index_corpus, load_corpus, rerank, rewrite_query, tokenize,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()
console = Console()
JUDGE_MODEL = "claude-sonnet-4-6"


# Golden set construit à la main : 15 paires
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


def prompt_a_strict(question: str, context: str) -> str:
    return f"""Tu es un assistant juridique strict.

CONTEXTE :
{context}

INSTRUCTIONS :
- Réponds UNIQUEMENT en t'appuyant sur le contexte
- Cite OBLIGATOIREMENT chaque source entre crochets
- Si l'information n'est PAS dans le contexte, réponds : "Information non disponible."

Question : {question}

Réponse :"""


def prompt_b_pedagogical(question: str, context: str) -> str:
    return f"""Tu es un assistant juridique pédagogue qui explique le droit en termes simples.

Contexte (articles du Code civil) :
{context}

Réponds en :
1. Donnant la réponse en langage simple et clair
2. Citant l'article source entre crochets
3. Ajoutant un exemple concret quand pertinent

Question : {question}

Réponse pédagogique :"""


def prompt_c_lawyer(question: str, context: str) -> str:
    return f"""Vous êtes un avocat répondant à un confrère. Style juridique soutenu.

Sources (Code civil français) :
{context}

Rédigez une réponse formelle citant systématiquement les visas (références entre crochets),
en employant le vocabulaire juridique consacré.

Question : {question}

Réponse formelle :"""


PROMPTS = {
    "A_strict": prompt_a_strict,
    "B_pedago": prompt_b_pedagogical,
    "C_lawyer": prompt_c_lawyer,
}


def retrieve_context_for(question: str, *, model, client, reranker, bm25, corpus, llm) -> tuple[list[str], list[str]]:
    rewritten = rewrite_query(question, llm)
    if "HORS_SUJET" in rewritten:
        return [], []
    top20 = hybrid_retrieve(rewritten, model, client, bm25, corpus)
    top5 = rerank(rewritten, top20, reranker)
    contexts = [f"[{c['article_id']}] {c['content']}" for c in top5]
    sources = [c["article_id"] for c in top5]
    return contexts, sources


def generate_with_prompt(prompt_fn, question: str, context: str, llm: Anthropic) -> str:
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=500, temperature=0,
        messages=[{"role": "user", "content": prompt_fn(question, context)}],
    )
    return msg.content[0].text.strip()


def build_dataset(prompt_fn, *, model, client, reranker, bm25, corpus, llm) -> Dataset:
    rows = []
    for case in GOLDEN:
        contexts, _ = retrieve_context_for(
            case["question"], model=model, client=client, reranker=reranker,
            bm25=bm25, corpus=corpus, llm=llm,
        )
        ctx_str = "\n\n".join(contexts) if contexts else ""
        answer = generate_with_prompt(prompt_fn, case["question"], ctx_str, llm)
        rows.append({
            "question": case["question"],
            "answer": answer,
            "contexts": contexts or [""],
            "ground_truth": case["ground_truth"],
        })
    return Dataset.from_list(rows)


def main() -> None:
    console.print("\n[bold orange3]Solution TP 2 — Comparaison de 3 prompts via Ragas[/bold orange3]\n")

    # Sauvegarder le golden set
    Path("golden.jsonl").write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in GOLDEN), encoding="utf-8"
    )
    console.print(f"[green]✓ Golden set sauvé : {len(GOLDEN)} paires[/green]")

    corpus = load_corpus()
    model = SentenceTransformer(EMBED_MODEL)
    client = index_corpus(corpus, model)
    reranker_model = CrossEncoder(RERANKER, max_length=512)
    bm25 = BM25Okapi([tokenize(c["content"]) for c in corpus])
    llm = Anthropic()

    judge_llm = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results: dict[str, dict[str, float]] = {}
    for name, prompt_fn in PROMPTS.items():
        console.print(f"\n[cyan]→ Variante {name} : génération...[/cyan]")
        dataset = build_dataset(
            prompt_fn, model=model, client=client, reranker=reranker_model,
            bm25=bm25, corpus=corpus, llm=llm,
        )
        console.print(f"[cyan]   Évaluation Ragas (peut prendre 2-3 min)...[/cyan]")
        result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=embeddings)
        df = result.to_pandas()
        df.to_csv(f"results_{name}.csv", index=False)

        results[name] = {
            m.name: float(df[m.name].dropna().mean())
            for m in metrics if m.name in df.columns
        }

    # Tableau comparatif
    table = Table(title="Comparaison Ragas — 3 variantes de prompt", show_lines=True)
    table.add_column("Métrique", style="cyan")
    for v in PROMPTS:
        table.add_column(v, justify="right")
    table.add_column("Gagnant", justify="center", style="bold orange3")

    for m in metrics:
        m_name = m.name
        scores = {v: results[v].get(m_name, 0.0) for v in PROMPTS}
        winner = max(scores, key=lambda k: scores[k])
        table.add_row(m_name, *[f"{scores[v]:.3f}" for v in PROMPTS], winner)
    console.print(table)

    # Recommandation type
    console.print("\n[bold]Recommandation type :[/bold]")
    console.print("• Si faithfulness max → A (strict) pour cas réglementés")
    console.print("• Si relevancy max → B (pédagogique) pour utilisateurs finaux")
    console.print("• Si formalisme max → C (avocat) pour communications confraternelles")
    console.print("• → Choix dépend du persona métier ciblé\n")


if __name__ == "__main__":
    main()
