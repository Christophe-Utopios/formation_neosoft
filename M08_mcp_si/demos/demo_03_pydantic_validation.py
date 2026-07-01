from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, ValidationError, field_validator
from rich.console import Console
from rich.panel import Panel

console = Console()


# Modèles d'entrée typés
class CreateUserInput(BaseModel):
    email: EmailStr
    name: str = Field(min_length=2, max_length=100)
    role: Literal["admin", "user", "viewer"]
    tenant_id: str = Field(pattern=r"^t_\d+$")
    notify_welcome: bool = True


class ExportRangeInput(BaseModel):
    start_date: date
    end_date: date
    format: Literal["csv", "json", "xlsx"]
    max_rows: int = Field(default=1000, ge=1, le=10000)

    @field_validator("end_date")
    @classmethod
    def check_range(cls, v: date, info):
        sd = info.data.get("start_date")
        if sd is None:
            return v
        if v < sd:
            raise ValueError("end_date doit être ≥ start_date")
        if (v - sd).days > 365:
            raise ValueError("Période > 365 jours interdite")
        return v


def try_input(label: str, model, payload: dict) -> None:
    console.print(f"\n[bold cyan]{label}[/bold cyan]")
    console.print(f"[dim]Input : {payload}[/dim]")
    try:
        validated = model(**payload)
        console.print(Panel(str(validated.model_dump()),
                            title="✓ Validation OK", border_style="green"))
    except ValidationError as e:
        # Premières erreurs uniquement (lisibilité)
        errors = "\n".join(f"  • [{err['loc'][0]}] {err['msg']}" for err in e.errors())
        console.print(Panel(errors, title="✗ ValidationError", border_style="red"))


def main() -> None:
    console.print("\n[bold orange3]Demo 03 — Validation Pydantic stricte[/bold orange3]\n")

    # CreateUserInput
    try_input(
        "Cas A — utilisateur valide",
        CreateUserInput,
        {"email": "alice@corp.fr", "name": "Alice Martin", "role": "admin", "tenant_id": "t_42"},
    )

    try_input(
        "Cas B — email malformé",
        CreateUserInput,
        {"email": "pas-un-email", "name": "Alice", "role": "admin", "tenant_id": "t_42"},
    )

    try_input(
        "Cas C — role hors enum",
        CreateUserInput,
        {"email": "a@b.fr", "name": "A", "role": "superadmin", "tenant_id": "t_42"},
    )

    try_input(
        "Cas D — tenant_id format invalide",
        CreateUserInput,
        {"email": "a@b.fr", "name": "Alice Martin", "role": "user", "tenant_id": "tenant42"},
    )

    # ExportRangeInput — règle métier
    try_input(
        "Cas E — export 30 jours OK",
        ExportRangeInput,
        {"start_date": "2026-01-01", "end_date": "2026-01-31", "format": "csv"},
    )

    try_input(
        "Cas F — fin avant début → refus",
        ExportRangeInput,
        {"start_date": "2026-03-01", "end_date": "2026-02-01", "format": "csv"},
    )

    try_input(
        "Cas G — période > 365 jours → refus",
        ExportRangeInput,
        {"start_date": "2024-01-01", "end_date": "2026-01-01", "format": "csv"},
    )

    try_input(
        "Cas H — max_rows hors range",
        ExportRangeInput,
        {"start_date": "2026-01-01", "end_date": "2026-02-01", "format": "csv", "max_rows": 50000},
    )

    console.print("\n[bold]À retenir :[/bold]")
    console.print("• Pydantic capture les types, enums, patterns, ranges en une déclaration")
    console.print("• Les validators custom (`@field_validator`) ajoutent la logique métier")
    console.print("• Les erreurs sont structurées (loc, msg, type) → facilement loggables\n")


if __name__ == "__main__":
    main()
