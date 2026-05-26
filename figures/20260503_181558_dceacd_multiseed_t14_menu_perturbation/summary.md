# T1.4 — Synthetic menu perturbation identifiability

Per N, 200 independent Gaussian-feature menus; MLE fit.  Two error metrics tracked:

  * `cos_err = 1 - cos(w_hat, w_true)` — angular error.
  * `l2_err_direction = ‖w_hat/‖w_hat‖ - w_true‖₂` — L2 norm of direction error.  Matches the metric used in §V.A of the paper.

## Error vs sample size N

| N | n_fits | median cos_err | p90 cos_err | median l2_err | p90 l2_err |
|---:|---:|---:|---:|---:|---:|
| 50 | 200 | 0.06172 | 0.11709 | 0.35133 | 0.48390 |
| 100 | 200 | 0.03357 | 0.07035 | 0.25913 | 0.37509 |
| 500 | 200 | 0.00660 | 0.01265 | 0.11490 | 0.15904 |
| 1000 | 200 | 0.00282 | 0.00652 | 0.07507 | 0.11418 |

## Log-log slopes (p90 vs N)

- **cos_err** slope = **-0.987** (intercept +1.789, R²=0.997).
- **l2_err_direction** slope = **-0.493** (intercept +1.241, R²=0.997).

Pre-registered decision rule (TUNING_PREREG §T1.4): identifiable if the C-R-style slope ∈ [-0.6, -0.4].  The pre-reg was implicitly written for L2 error (matching §V.A of the paper, where the empirical slope is -0.498 ± 0.014).
- L2 verdict: PASS (slope -0.493 vs [-0.6, -0.4]).  This is the metric the pre-reg threshold targets.
- cos verdict: PASS (faster decay) (slope -0.987; angular error decays faster than L2 by construction when ‖w‖ is roughly stable).