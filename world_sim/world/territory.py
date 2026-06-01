"""Grafo territorial de los 22 departamentos.

Atributos por nodo: pobreza, pct_indigena, sequia_spi, homicidios_100k,
presencia_estatal, rural. Aristas = adyacencia.

Propagación de shocks: un shock climático en un nodo afecta a vecinos con
elasticidad decreciente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass
class TerritorySummary:
    """Resumen agregado que se manda al presidente."""

    n_deptos_en_crisis: int
    regiones_criticas: list[str]
    pobreza_media_ponderada: float
    pobreza_p90: float
    sequia_media: float
    homicidios_p90: float
    deptos_top_pobreza: list[str]
    deptos_top_homicidios: list[str]
    deptos_top_sequia: list[str]

    def as_dict(self) -> dict:
        return {
            "n_deptos_en_crisis": self.n_deptos_en_crisis,
            "regiones_criticas": self.regiones_criticas,
            "pobreza_media_ponderada": round(self.pobreza_media_ponderada, 1),
            "pobreza_p90": round(self.pobreza_p90, 1),
            "sequia_media": round(self.sequia_media, 2),
            "homicidios_p90": round(self.homicidios_p90, 1),
            "deptos_top_pobreza": self.deptos_top_pobreza,
            "deptos_top_homicidios": self.deptos_top_homicidios,
            "deptos_top_sequia": self.deptos_top_sequia,
        }


class Territory:
    def __init__(self, graph: nx.Graph):
        self.G = graph

    @classmethod
    def load_default(cls) -> "Territory":
        nodos = pd.read_csv(DATA_DIR / "departamentos.csv")
        aristas = pd.read_csv(DATA_DIR / "adyacencias.csv")
        G = nx.Graph()
        for _, row in nodos.iterrows():
            G.add_node(row["departamento"], **row.to_dict())
        for _, row in aristas.iterrows():
            G.add_edge(row["a"], row["b"], peso=float(row["peso"]))
        return cls(G)

    # --- consultas ------------------------------------------------------------

    def deptos(self) -> list[str]:
        return list(self.G.nodes)

    def attr(self, name: str) -> pd.Series:
        return pd.Series({n: d[name] for n, d in self.G.nodes(data=True)}, name=name)

    def en_crisis(self) -> list[str]:
        """Un depto está en crisis si pobreza > 70 o sequía > 0.6 o hom > 30."""
        out = []
        for n, d in self.G.nodes(data=True):
            if d["pobreza"] > 70 or d["sequia_spi"] > 0.6 or d["homicidios_100k"] > 30:
                out.append(n)
        return out

    # --- dinámica -------------------------------------------------------------

    def propagar_shock_climatico(self, epicentro: str, intensidad: float = 0.3) -> None:
        """Aumenta sequía/pobreza en epicentro y decae por vecindad."""
        if epicentro not in self.G:
            return
        visitados = {epicentro: 0}
        frontera = [epicentro]
        while frontera:
            nuevo: list[str] = []
            for node in frontera:
                for v in self.G.neighbors(node):
                    if v not in visitados:
                        visitados[v] = visitados[node] + 1
                        nuevo.append(v)
            frontera = nuevo
        for node, dist in visitados.items():
            factor = intensidad * (0.6 ** dist)
            self.G.nodes[node]["sequia_spi"] = min(1.0, self.G.nodes[node]["sequia_spi"] + factor)
            self.G.nodes[node]["pobreza"] = min(95.0, self.G.nodes[node]["pobreza"] + 0.5 * factor * 100.0)

    def aplicar_inversion_infraestructura(self, peso_inversion: float) -> None:
        """Inversión homogénea: rebaja pobreza ligeramente en deptos rurales."""
        for n, d in self.G.nodes(data=True):
            rural = d["rural"] / 100.0
            self.G.nodes[n]["pobreza"] = max(
                5.0, d["pobreza"] - 2.0 * peso_inversion * rural
            )

    def step(self, state, rng) -> None:
        """Paso de tiempo territorial con ruido leve."""
        for n, d in self.G.nodes(data=True):
            # la sequía decae naturalmente
            self.G.nodes[n]["sequia_spi"] = max(0.0, d["sequia_spi"] - 0.05 + 0.02 * float(rng.normal()))
            # homicidios con ruido
            self.G.nodes[n]["homicidios_100k"] = max(
                0.0, d["homicidios_100k"] + 0.8 * float(rng.normal())
            )
        # si hay sequía severa en el estado, aplicar a epicentro del corredor seco
        for sh in state.shocks_activos:
            if "sequía" in sh or "sequia" in sh:
                self.propagar_shock_climatico("Chiquimula", intensidad=0.4)
            if "huracán" in sh or "huracan" in sh:
                self.propagar_shock_climatico("Izabal", intensidad=0.35)

    # --- resumen -------------------------------------------------------------

    def summary(self) -> TerritorySummary:
        pobreza = self.attr("pobreza")
        sequia = self.attr("sequia_spi")
        homs = self.attr("homicidios_100k")
        region = self.attr("region")

        # pobreza ponderada por rural (proxy de peso poblacional rural)
        rural = self.attr("rural")
        w = rural / rural.sum()
        pobreza_ponderada = float((pobreza * w).sum())

        crisis = self.en_crisis()
        regiones_crit = sorted({region[d] for d in crisis})

        return TerritorySummary(
            n_deptos_en_crisis=len(crisis),
            regiones_criticas=regiones_crit,
            pobreza_media_ponderada=pobreza_ponderada,
            pobreza_p90=float(pobreza.quantile(0.9)),
            sequia_media=float(sequia.mean()),
            homicidios_p90=float(homs.quantile(0.9)),
            deptos_top_pobreza=pobreza.sort_values(ascending=False).head(3).index.tolist(),
            deptos_top_homicidios=homs.sort_values(ascending=False).head(3).index.tolist(),
            deptos_top_sequia=sequia.sort_values(ascending=False).head(3).index.tolist(),
        )
