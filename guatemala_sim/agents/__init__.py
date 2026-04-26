"""Agentes políticos/sectoriales (Mesa v3)."""

from .base import AgentesModel, AgenteBase
from .partidos import CongresoOposicion, PartidoOficialista
from .gremiales import CACIF
from .sociales import ProtestaSocial

__all__ = [
    "AgentesModel",
    "AgenteBase",
    "CongresoOposicion",
    "PartidoOficialista",
    "CACIF",
    "ProtestaSocial",
]
