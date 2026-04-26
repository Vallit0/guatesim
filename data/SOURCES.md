# Fuentes de datos

## Datos automatizados

### Banco Mundial — `world_bank_gtm.csv`

Generado por `python -m guatemala_sim.refresh_data`. Contiene los últimos
~7 años de los siguientes indicadores (códigos del WB Data API):

| Código | Campo del state | Última actualización típica |
|---|---|---|
| `NY.GDP.MKTP.CD` | `pib_usd_mm` | t-1 |
| `NY.GDP.MKTP.KD.ZG` | `crecimiento_pib` | t-1 |
| `FP.CPI.TOTL.ZG` | `inflacion` | t-1 |
| `GC.DOD.TOTL.GD.ZS` | `deuda_pib` | t-2 (frecuentemente con problemas de JSON; manual) |
| `FI.RES.TOTL.CD` | `reservas_usd_mm` | t-1 |
| `BN.CAB.XOKA.GD.ZS` | `cuenta_corriente_pib` | t-1 |
| `BX.TRF.PWKR.DT.GD.ZS` | `remesas_pib` | t-1 |
| `PA.NUS.FCRF` | `tipo_cambio` | t-1 |
| `BX.KLT.DINV.CD.WD` | `ied_usd_mm` | t-1 |
| `SP.POP.TOTL` | `poblacion_mm` | t-1 |
| `SI.POV.NAHC` | `pobreza_general` | t-2 (ENCOVI; frecuencia variable) |
| `SI.POV.GINI` | `gini` | t-2 |
| `SL.UEM.TOTL.ZS` | `desempleo` | t-1 |
| `VC.IHR.PSRC.P5` | `homicidios_100k` | t-1 |
| `SE.PRM.NENR` | `matricula_primaria` | t-3 (UIS; lag alto) |
| `SH.UHC.SRVS.CV.XD` | `cobertura_salud` | t-3 (frecuentemente con problemas de JSON; manual) |

**Fuente primaria:** World Bank Open Data API
([data.worldbank.org](https://data.worldbank.org)).
**Licencia:** CC BY 4.0.

### Geografía — `departamentos.csv` y `adyacencias.csv`

Estructura calibrada a mano contra:

- INE Censo 2018 (población y composición étnica)
- ENCOVI 2014 / actualizaciones 2023 (pobreza por departamento)
- Ministerio Público / PNC (homicidios por departamento)
- IGN (mapa de adyacencias)
- INSIVUMEH (índice de sequía SPI por región)

| Archivo | Contenido | Última revisión |
|---|---|---|
| `departamentos.csv` | 22 deptos × {region, pobreza, pct_indigena, sequia_spi, homicidios_100k, presencia_estatal, rural} | 2026-04 |
| `adyacencias.csv` | Pares `(a, b, peso)` de deptos limítrofes | 2026-04 |

## Datos pendientes (manuales)

Para indicadores políticos/perceptuales **no expuestos por el Banco Mundial**,
se sugiere mantener un CSV manualmente curado. Los campos del state que
caen en esta categoría:

| Campo | Fuente sugerida | Frecuencia |
|---|---|---|
| `aprobacion_presidencial` | CIEP, Latinobarómetro, encuestas locales | mensual–trimestral |
| `confianza_institucional` | Latinobarómetro, LAPOP | anual–bianual |
| `coalicion_congreso` | Congreso de la República (composición) | por legislatura |
| `libertad_prensa` | RSF, Freedom House | anual |
| `informalidad` | INE ENEI, ILO STAT | trimestral |
| `pobreza_extrema` | INE ENCOVI | irregular |
| `migracion_neta_miles` | OIM, DGM, US CBP | anual |
| `alineamiento_eeuu` / `china` / etc. | juicio experto / IMF Article IV | anual |
| `apoyo_multilateral` | IMF, BID, BM (acceso a programas) | anual |

Pendientes de ingesta automatizada:

- [ ] `banguat_2025.csv` — PIB trimestral, política monetaria (Banguat
  publica Excel, no API estable)
- [ ] `encovi_agregado.csv` — pobreza por depto (INE)
- [ ] `minfin_ejecucion.csv` — ejecución presupuestaria (SICOIN)
- [ ] `tse_congreso.csv` — composición del Congreso (TSE)
- [ ] ACLED API — violencia política y protesta (key opcional)

## Procedimiento de actualización

```bash
# refresh datos del Banco Mundial (se hace cada vez que querés calibrar)
python -m guatemala_sim.refresh_data

# verificar el snapshot
python -c "
from guatemala_sim.bootstrap import initial_state_calibrated
s, meta = initial_state_calibrated()
print(f'PIB: USD {s.macro.pib_usd_mm:,.0f} MM (calibrado contra WB)')
print(f'campos reemplazados: {len(meta[\"campos_reemplazados\"])}')
"
```

## Licencia y reuso

- Datos del Banco Mundial: CC BY 4.0 — se redistribuyen sin
  modificación; cualquier transformación está documentada en
  `WB_INDICATORS` (`data_ingest.py`).
- Datos del INE / Banguat / MINFIN: dominio público (datos oficiales
  guatemaltecos), pero las extracciones manuales en este repo deben citar
  la versión y fecha de descarga.
