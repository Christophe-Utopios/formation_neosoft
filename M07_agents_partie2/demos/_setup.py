"""Setup partagé entre démos M7. Réutilise le mock CRM du M6."""
from __future__ import annotations

import sys
from pathlib import Path

# Réutilise le mock CRM du M6
M6_DEMOS = Path(__file__).resolve().parent.parent.parent / "M06_agents_partie1" / "demos"
sys.path.insert(0, str(M6_DEMOS))
import _fake_crm as crm  # noqa: E402,F401
