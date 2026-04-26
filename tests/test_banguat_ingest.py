"""Tests offline de `banguat_ingest`. NO tocan la red: mockeamos
`requests.post` para devolver respuestas SOAP sintéticas con la misma
forma que el endpoint real de Banguat.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from guatemala_sim.banguat_ingest import (
    BANGUAT_TIPO_CAMBIO_URL,
    MONEDAS_BANGUAT,
    latest_tipo_cambio_promedio,
    load_tipo_cambio_snapshot,
    save_tipo_cambio_snapshot,
    tipo_cambio_dia,
    tipo_cambio_rango,
    tipo_cambio_rango_moneda,
)


# --- fixtures de respuestas SOAP -------------------------------------------


def _fake_response_dia(referencia: float = 7.63867, fecha: str = "26/04/2026") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
        "<soap:Body>"
        '<TipoCambioDiaResponse xmlns="http://www.banguat.gob.gt/variables/ws/">'
        "<TipoCambioDiaResult>"
        f"<CambioDolar><VarDolar><fecha>{fecha}</fecha>"
        f"<referencia>{referencia}</referencia></VarDolar></CambioDolar>"
        "<TotalItems>1</TotalItems>"
        "</TipoCambioDiaResult>"
        "</TipoCambioDiaResponse>"
        "</soap:Body></soap:Envelope>"
    )


def _fake_response_rango(rows: list[dict]) -> str:
    """Genera un envelope SOAP con `<Var>` repetidos."""
    items = []
    for r in rows:
        items.append(
            "<Var>"
            f"<moneda>{r.get('moneda_code', 2)}</moneda>"
            f"<fecha>{r['fecha']}</fecha>"
            f"<venta>{r['venta']}</venta>"
            f"<compra>{r['compra']}</compra>"
            "</Var>"
        )
    inner = "".join(items)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        "<soap:Body>"
        '<TipoCambioRangoResponse xmlns="http://www.banguat.gob.gt/variables/ws/">'
        "<TipoCambioRangoResult>"
        f"<Vars>{inner}</Vars>"
        f"<TotalItems>{len(rows)}</TotalItems>"
        "</TipoCambioRangoResult>"
        "</TipoCambioRangoResponse>"
        "</soap:Body></soap:Envelope>"
    )


def _make_mock_response(text: str, status: int = 200):
    m = MagicMock()
    m.status_code = status
    m.text = text
    m.raise_for_status = MagicMock()
    return m


# --- tests -----------------------------------------------------------------


def test_tipo_cambio_dia_parsea_referencia():
    fake = _fake_response_dia(referencia=7.63867, fecha="26/04/2026")
    with patch("guatemala_sim.banguat_ingest.requests.post",
               return_value=_make_mock_response(fake)) as mock_post:
        obs = tipo_cambio_dia()
    assert obs.referencia == pytest.approx(7.63867, abs=1e-5)
    assert obs.moneda == "USD"
    assert obs.fecha == date(2026, 4, 26)
    # verifica que se llamó al endpoint correcto
    args, kwargs = mock_post.call_args
    assert args[0] == BANGUAT_TIPO_CAMBIO_URL
    assert kwargs["headers"]["Content-Type"].startswith("text/xml")


def test_tipo_cambio_rango_parsea_serie():
    rows = [
        {"fecha": "01/04/2026", "venta": 7.7000, "compra": 7.6500},
        {"fecha": "02/04/2026", "venta": 7.7100, "compra": 7.6600},
        {"fecha": "03/04/2026", "venta": 7.6900, "compra": 7.6400},
    ]
    fake = _fake_response_rango(rows)
    with patch("guatemala_sim.banguat_ingest.requests.post",
               return_value=_make_mock_response(fake)):
        df = tipo_cambio_rango(date(2026, 4, 1), date(2026, 4, 3))
    assert len(df) == 3
    assert list(df["fecha"]) == [date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3)]
    # referencia = (venta + compra) / 2
    assert df["referencia"].iloc[0] == pytest.approx(7.6750, abs=1e-4)
    assert (df["moneda"] == "USD").all()


def test_tipo_cambio_rango_rechaza_orden_invalido():
    with pytest.raises(ValueError, match="fecha_inicial"):
        tipo_cambio_rango(date(2026, 5, 1), date(2026, 4, 1))


def test_tipo_cambio_rango_moneda_codigo_correcto():
    """Verifica que se envía el código de moneda correcto en el payload SOAP."""
    fake = _fake_response_rango([
        {"fecha": "01/04/2026", "venta": 8.50, "compra": 8.40, "moneda_code": 5},
    ])
    with patch("guatemala_sim.banguat_ingest.requests.post",
               return_value=_make_mock_response(fake)) as mock_post:
        df = tipo_cambio_rango_moneda("EUR", date(2026, 4, 1), date(2026, 4, 1))
    # el payload debe incluir <moneda>5</moneda>
    body_sent = mock_post.call_args.kwargs["data"].decode("utf-8")
    assert f"<moneda>{MONEDAS_BANGUAT['EUR']}</moneda>" in body_sent
    assert df["moneda"].iloc[0] == "EUR"


def test_tipo_cambio_rango_moneda_rechaza_desconocida():
    with pytest.raises(ValueError, match="moneda desconocida"):
        tipo_cambio_rango_moneda("ARS", date(2026, 1, 1))


def test_save_load_snapshot_roundtrip(tmp_path: Path):
    df = pd.DataFrame({
        "fecha": [date(2026, 4, 1), date(2026, 4, 2)],
        "moneda": ["USD", "USD"],
        "venta": [7.70, 7.71],
        "compra": [7.65, 7.66],
        "referencia": [7.675, 7.685],
    })
    p = tmp_path / "tc.csv"
    save_tipo_cambio_snapshot(df, p)
    df2 = load_tipo_cambio_snapshot(p)
    assert list(df["fecha"]) == list(df2["fecha"])
    assert df["referencia"].tolist() == df2["referencia"].tolist()


def test_latest_tipo_cambio_promedio_ventana():
    """Promedio de los últimos N días, no de toda la serie."""
    fechas = [date(2026, 1, 1), date(2026, 2, 1), date(2026, 4, 25), date(2026, 4, 26)]
    refs = [10.0, 10.0, 7.6, 7.7]
    df = pd.DataFrame({"fecha": fechas, "moneda": "USD", "referencia": refs})
    # ventana de 30 días desde el max → solo abr-25 y abr-26
    avg = latest_tipo_cambio_promedio(df, ventana_dias=30)
    assert avg == pytest.approx(7.65, abs=0.01)
    # ventana muy ancha → toma todo
    avg_full = latest_tipo_cambio_promedio(df, ventana_dias=400)
    assert avg_full == pytest.approx((10 + 10 + 7.6 + 7.7) / 4)


def test_latest_tipo_cambio_promedio_vacio_levanta():
    with pytest.raises(ValueError, match="vacío"):
        latest_tipo_cambio_promedio(pd.DataFrame(columns=["fecha", "referencia"]))


def test_retry_en_fallo_transitorio():
    """Ante un 500, la primera llamada falla; la segunda (retry) sucede."""
    import requests
    fake_ok = _make_mock_response(_fake_response_dia())
    fake_bad = MagicMock()
    fake_bad.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return fake_bad
        return fake_ok

    with patch("guatemala_sim.banguat_ingest.requests.post", side_effect=side_effect), \
         patch("guatemala_sim.banguat_ingest.time.sleep"):  # acelera el test
        obs = tipo_cambio_dia()
    assert call_count["n"] == 2  # un fallo + un éxito
    assert obs.referencia > 0


def test_falla_definitiva_levanta_runtime():
    """Si todos los retries fallan, debe levantar RuntimeError descriptivo."""
    import requests
    fake_bad = MagicMock()
    fake_bad.raise_for_status.side_effect = requests.HTTPError("500")

    with patch("guatemala_sim.banguat_ingest.requests.post", return_value=fake_bad), \
         patch("guatemala_sim.banguat_ingest.time.sleep"), \
         pytest.raises(RuntimeError, match="Banguat SOAP"):
        tipo_cambio_dia()
