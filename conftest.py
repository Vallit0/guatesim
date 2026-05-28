"""Hace que los scripts de `scripts/` sean importables desde los tests.

Tras la reorganización (commit que mueve los entry points de la raíz a
`scripts/`), tests que importan `irl_sensitivity_analysis` etc. necesitan
que `scripts/` esté en `sys.path`. Esto reemplaza los `sys.path.insert`
ad-hoc que vivían en cada test individual.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _REPO_ROOT / "scripts"

for path in (_REPO_ROOT, _SCRIPTS):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
