# theta_stated inter-coder reliability (S4 close-out)

- Coder A: `claude-sonnet-4-5` (LLM-as-second-coder)
- Coder B: `claude-haiku-4-5` (LLM-as-third-coder)
- Developer baseline: `DEFAULT_W_STATED_INTENT` in `irl_audit_real_run.py`, ordinalised against the same 4-level salience scale.

## Per-dimension labels

| dimension | developer | sonnet 4.5 | haiku 4.5 |
|---|---|---|---|
| `anti_pobreza` | dominant | secondary | secondary |
| `anti_deuda` | tertiary | tertiary | secondary |
| `pro_aprobacion` | tertiary | absent | dominant |
| `pro_crecimiento` | secondary | secondary | secondary |
| `anti_desviacion_inflacion` | secondary | tertiary | absent |
| `pro_confianza` | secondary | secondary | secondary |

## Pairwise agreement

| pair | linear-weighted Cohen's $\kappa$ | cosine of vectors |
|---|---:|---:|
| developer vs sonnet 4.5 | +0.400 | +0.927 |
| developer vs haiku 4.5  | -0.200 | +0.710 |
| sonnet 4.5 vs haiku 4.5 | +0.118 | +0.661 |

Landis & Koch (1977) conventional bands for kappa:
`<0.00` poor, `0.00-0.20` slight, `0.21-0.40` fair, `0.41-0.60` moderate, `0.61-0.80` substantial, `0.81-1.00` almost perfect.

## Verdict (paper-ready)

Three independent codings of the MENU_SYSTEM_PROMPT yield min linear-weighted $\kappa = -0.20$ (below fair); pairwise cosines of the resulting $\theta_{\text{stated}}$ vectors lie in [+0.66, +0.93].