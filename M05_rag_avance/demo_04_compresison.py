from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from _setup import get_or_create_index, search_dense

load_dotenv()
console = Console()
LLM_MODEL = "claude-haiku-4-5-20251001"

EXTRACT_PROMPT = """Question : {question}

Texte :
{chunk}

Extrais UNIQUEMENT les phrases du texte ci-dessus qui répondent directement à la question.
Reproduis-les MOT POUR MOT, sans paraphraser ni résumer.
Si rien dans le texte n'est pertinent, réponds exactement : NONE"""


def count_tokens_approx(text: str) -> int:
    """Approximation simple : 1 token ≈ 4 caractères en français."""
    return max(1, len(text) // 4)


def extractive_compress(question: str, chunk: str, llm: Anthropic) -> str:
    """Compression extractive : ne garde que les phrases pertinentes du chunk."""
    # Utilise le LLM pour identifier et extraire uniquement les phrases utiles
    # Extractive = copie verbatim (pas de paraphrase), préserve la fidélité
    msg = llm.messages.create(
        model=LLM_MODEL, max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(question=question, chunk=chunk)}],
    )
    out = msg.content[0].text.strip()
    # Si le chunk n'est pas pertinent, le LLM renvoie "NONE" et on le supprime
    return "" if out == "NONE" else out


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY manquant.[/red]")
        return

    console.print("\n[bold orange3]Demo 04 — Compression contextuelle[/bold orange3]\n")
    client, model, _ = get_or_create_index()
    llm = Anthropic()

    questions = [
        "Comment activer SAML ?",
        "Quelle est la durée de rétention des sauvegardes ?",
    ]

    for question in questions:
        console.print(f"\n[bold cyan]Question : {question}[/bold cyan]")
        hits = search_dense(client, model, question, top_k=5)

        # Contexte original (sans compression)
        # Baseline : tous les chunks récupérés, envoyés tels quels au LLM
        original_chunks = [h.payload["content"] for h in hits]
        original_text = "\n\n---\n\n".join(original_chunks)
        original_tokens = count_tokens_approx(original_text)

        # Compression : extraction des phrases pertinentes par chunk
        # Chaque chunk est analysé individuellement pour filtrer le bruit
        compressed_chunks = [extractive_compress(question, c, llm) for c in original_chunks]
        compressed_chunks_kept = [c for c in compressed_chunks if c]
        compressed_text = "\n\n---\n\n".join(compressed_chunks_kept)
        compressed_tokens = count_tokens_approx(compressed_text)

        # Stats
        ratio = compressed_tokens / max(original_tokens, 1) * 100
        kept = len(compressed_chunks_kept)
        dropped = len(compressed_chunks) - kept

        table = Table(show_header=False, show_lines=False)
        table.add_column("", style="cyan")
        table.add_column("", justify="right")
        table.add_row("Tokens originaux", str(original_tokens))
        table.add_row("Tokens compressés", str(compressed_tokens))
        table.add_row("Ratio compression", f"{ratio:.1f}%")
        table.add_row("Chunks gardés", f"{kept}/{len(compressed_chunks)}")
        table.add_row("Chunks droppés (NONE)", str(dropped))
        console.print(table)

        if compressed_text:
            console.print(Panel(
                compressed_text[:600] + ("..." if len(compressed_text) > 600 else ""),
                title="Contexte compressé envoyé au LLM",
                border_style="orange3",
            ))

    console.print("\n[bold]Observation :[/bold]")
    console.print("• Compression typique : 30-50% de réduction des tokens d'entrée")
    console.print("• Coût : 1 appel LLM par chunk → mesurer le break-even")
    console.print("• Alternative production : LLMLingua-2 (plus rapide, sans LLM)\n")


if __name__ == "__main__":
    main()