"""Ingesta de datos del Banco de Guatemala (Banguat) vía SOAP.

Banguat expone un *web service* SOAP en `https://banguat.gob.gt/variables/ws/`.
Sólo `TipoCambio.asmx` es estable y verificado al 2026-04. Otros endpoints
(`Variables.asmx`, IPC, IMAE) devuelven HTTP 500 intermitentemente y NO se
soportan acá. Para esos indicadores, ver el comentario al final.

Operaciones soportadas (todas sobre `TipoCambio.asmx`):

* `tipo_cambio_dia()` → tipo de cambio referencial USD/GTQ del día.
* `tipo_cambio_rango(fecha_inicial, fecha_final)` → serie diaria USD/GTQ.
* `tipo_cambio_rango_moneda(moneda, fecha_inicial, fecha_final)` → serie
  diaria para una moneda específica (códigos en `MONEDAS_BANGUAT`).

Diseño:

* Sin dependencia SOAP pesada (no requiere `zeep`/`suds`): cliente
  artesanal con `requests` + parseo XML con `xml.etree.ElementTree`.
* Cada llamada usa retry exponencial frente a fallos transitorios
  (Banguat tiene microcortes frecuentes).
* `to_state_field()` agrega la serie y la mapea al campo `tipo_cambio`
  del `state` para usar como override en la calibración inicial.

Uso desde CLI:
    python -m guatemala_sim.refresh_data --banguat
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests


BANGUAT_NS = {
    "soap": "http://schemas.xmlsoap.org/soap/envelope/",
    "ns": "http://www.banguat.gob.gt/variables/ws/",
}

BANGUAT_TIPO_CAMBIO_URL = "https://banguat.gob.gt/variables/ws/TipoCambio.asmx"

# Códigos de moneda según Banguat (los más usados; lista completa con
# `Variables` operation, pero como ese endpoint es flaky listamos acá).
MONEDAS_BANGUAT: dict[str, int] = {
    "USD": 2,    # dólar EE.UU.
    "EUR": 5,    # euro
    "MXN": 6,    # peso mexicano
    "GBP": 7,    # libra esterlina
    "CAD": 8,    # dólar canadiense
    "JPY": 11,   # yen japonés
    "CRC": 12,   # colón costarricense
    "HNL": 13,   # lempira hondureño
}


# --- cliente SOAP artesanal -----------------------------------------------


def _soap_envelope(operation: str, params_xml: str = "") -> str:
    """Construye un sobre SOAP 1.1 para el web service de Banguat."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
        "<soap:Body>"
        f'<{operation} xmlns="http://www.banguat.gob.gt/variables/ws/">'
        f"{params_xml}"
        f"</{operation}>"
        "</soap:Body>"
        "</soap:Envelope>"
    )


def _post_soap(
    operation: str,
    params_xml: str = "",
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> ET.Element:
    """Llama al SOAP y devuelve el `<soap:Body>` parseado.

    Reintenta exponencialmente: el endpoint tiene picos de 500/timeout.
    """
    body = _soap_envelope(operation, params_xml)
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": f"http://www.banguat.gob.gt/variables/ws/{operation}",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                BANGUAT_TIPO_CAMBIO_URL,
                data=body.encode("utf-8"),
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            soap_body = root.find("soap:Body", BANGUAT_NS)
            if soap_body is None:
                raise RuntimeError(f"respuesta sin soap:Body: {resp.text[:200]}")
            return soap_body
        except (requests.RequestException, ET.ParseError, RuntimeError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise RuntimeError(
        f"Banguat SOAP `{operation}` falló tras {retries} intentos: "
        f"{type(last_err).__name__}: {last_err}"
    )


# --- parseo de respuestas --------------------------------------------------


def _parse_tipo_cambio_dia(body: ET.Element) -> pd.DataFrame:
    """`TipoCambioDia` → 1 fila con `fecha`, `referencia` (USD)."""
    rows: list[dict] = []
    for var in body.iter(f"{{{BANGUAT_NS['ns']}}}VarDolar"):
        fecha = var.findtext("ns:fecha", namespaces=BANGUAT_NS)
        referencia = var.findtext("ns:referencia", namespaces=BANGUAT_NS)
        if fecha is None or referencia is None:
            continue
        rows.append({
            "fecha": _parse_banguat_date(fecha),
            "moneda": "USD",
            "referencia": float(referencia),
        })
    return pd.DataFrame(rows)


def _parse_tipo_cambio_rango(body: ET.Element) -> pd.DataFrame:
    """`TipoCambioRango`/`TipoCambioRangoMoneda` → serie diaria.

    Banguat devuelve `<Var><moneda>2</moneda><fecha>26/04/2026</fecha>
    <venta>7.638</venta><compra>7.625</compra></Var>` en `<CambioDia>`.
    """
    rows: list[dict] = []
    for var in body.iter(f"{{{BANGUAT_NS['ns']}}}Var"):
        moneda_code = var.findtext("ns:moneda", namespaces=BANGUAT_NS)
        fecha = var.findtext("ns:fecha", namespaces=BANGUAT_NS)
        venta = var.findtext("ns:venta", namespaces=BANGUAT_NS)
        compra = var.findtext("ns:compra", namespaces=BANGUAT_NS)
        if fecha is None or venta is None or compra is None:
            continue
        moneda_label = _moneda_code_to_label(int(moneda_code) if moneda_code else 2)
        rows.append({
            "fecha": _parse_banguat_date(fecha),
            "moneda": moneda_label,
            "venta": float(venta),
            "compra": float(compra),
            "referencia": (float(venta) + float(compra)) / 2.0,
        })
    return pd.DataFrame(rows)


def _parse_banguat_date(s: str) -> date:
    """Banguat usa `dd/mm/yyyy`."""
    return datetime.strptime(s.strip(), "%d/%m/%Y").date()


def _moneda_code_to_label(code: int) -> str:
    for label, c in MONEDAS_BANGUAT.items():
        if c == code:
            return label
    return f"COD_{code}"


# --- API pública ----------------------------------------------------------


@dataclass
class TipoCambioObservacion:
    fecha: date
    moneda: str
    referencia: float           # GTQ por unidad de moneda extranjera
    venta: float | None = None
    compra: float | None = None


def tipo_cambio_dia(timeout: float = 15.0) -> TipoCambioObservacion:
    """Tipo de cambio referencial USD/GTQ del día actual.

    Banguat publica un único valor de referencia por día (no compra/venta).
    """
    body = _post_soap("TipoCambioDia", timeout=timeout)
    df = _parse_tipo_cambio_dia(body)
    if df.empty:
        raise RuntimeError("Banguat devolvió respuesta vacía para TipoCambioDia")
    r = df.iloc[0]
    return TipoCambioObservacion(
        fecha=r["fecha"], moneda=r["moneda"], referencia=float(r["referencia"]),
    )


def tipo_cambio_rango(
    fecha_inicial: date,
    fecha_final: date | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Serie diaria USD/GTQ entre `fecha_inicial` y `fecha_final` (inclusive).

    Si `fecha_final` es `None`, usa hoy. Devuelve DataFrame con columnas
    `fecha` (date), `moneda` ('USD'), `venta`, `compra`, `referencia`.
    """
    if fecha_final is None:
        fecha_final = date.today()
    if fecha_inicial > fecha_final:
        raise ValueError(
            f"fecha_inicial {fecha_inicial} > fecha_final {fecha_final}"
        )
    params = (
        f"<fechainit>{fecha_inicial.strftime('%d/%m/%Y')}</fechainit>"
        f"<fechafin>{fecha_final.strftime('%d/%m/%Y')}</fechafin>"
    )
    body = _post_soap("TipoCambioRango", params, timeout=timeout)
    df = _parse_tipo_cambio_rango(body)
    return df.sort_values("fecha").reset_index(drop=True)


def tipo_cambio_rango_moneda(
    moneda: str,
    fecha_inicial: date,
    fecha_final: date | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Idem `tipo_cambio_rango` pero para una moneda específica.

    `moneda` debe estar en `MONEDAS_BANGUAT` (ej. 'EUR', 'MXN').
    """
    if moneda not in MONEDAS_BANGUAT:
        raise ValueError(
            f"moneda desconocida: {moneda}. Disponibles: {list(MONEDAS_BANGUAT)}"
        )
    if fecha_final is None:
        fecha_final = date.today()
    code = MONEDAS_BANGUAT[moneda]
    params = (
        f"<moneda>{code}</moneda>"
        f"<fechainit>{fecha_inicial.strftime('%d/%m/%Y')}</fechainit>"
        f"<fechafin>{fecha_final.strftime('%d/%m/%Y')}</fechafin>"
    )
    body = _post_soap("TipoCambioRangoMoneda", params, timeout=timeout)
    return _parse_tipo_cambio_rango(body).sort_values("fecha").reset_index(drop=True)


# --- persistencia / integración con bootstrap -----------------------------


def save_tipo_cambio_snapshot(df: pd.DataFrame, path: Path) -> Path:
    """Persiste la serie a CSV (compat con `data_ingest.load_snapshot` aunque
    el formato es distinto al del Banco Mundial)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_tipo_cambio_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    return df


def latest_tipo_cambio_promedio(
    df: pd.DataFrame, ventana_dias: int = 30,
) -> float:
    """Promedio de los últimos `ventana_dias` (USD/GTQ) — más estable que el
    valor de un día puntual para usar como override del campo `tipo_cambio`
    del state."""
    if df.empty:
        raise ValueError("DataFrame vacío")
    df = df.sort_values("fecha")
    cutoff = df["fecha"].max() - timedelta(days=ventana_dias)
    sub = df[df["fecha"] >= cutoff]
    if sub.empty:
        sub = df.tail(1)
    return float(sub["referencia"].mean())


def fetch_and_save_default(
    out_path: Path | None = None,
    dias_atras: int = 365,
) -> Path:
    """One-shot: descarga el último año de USD/GTQ y lo guarda en
    `data/banguat_tipo_cambio.csv` (o donde diga `out_path`)."""
    if out_path is None:
        out_path = (
            Path(__file__).resolve().parent.parent
            / "data" / "banguat_tipo_cambio.csv"
        )
    fin = date.today()
    inicio = fin - timedelta(days=dias_atras)
    df = tipo_cambio_rango(inicio, fin)
    save_tipo_cambio_snapshot(df, out_path)
    return out_path


# --- nota sobre indicadores no soportados ---------------------------------
#
# IPC, IMAE, tasa líder, agregados monetarios: Banguat los publica como PDFs
# y XLSX en https://banguat.gob.gt/estadisticas/, pero no expone API REST/
# SOAP estable para ellos. Para incorporarlos, lo razonable es:
#
#   1. Descargar manualmente los XLSX a `data/banguat_<indicador>.xlsx`.
#   2. Agregar un parser dedicado (`banguat_ipc.py`) usando `pandas.read_excel`.
#   3. Mapear al campo del state correspondiente desde `data_ingest.py`.
#
# Cualquier scraper HTML del sitio público va a romperse cada pocos meses;
# preferimos el path manual + parser robusto al PDF/XLSX.
