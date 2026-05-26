"""T2.4 chosen-index agreement across temperatures (pre-registered stat (a)).

For each model and each (seed, turn), compares the menu choice
(`menu_choice.chosen_index`) between two batches collected under
different sampling temperatures but identical seeds and shocks. Reports
the per-model fraction of turns on which the two batches picked the
same menu item.

Pre-registered decision rule (TUNING_PREREG.md T2.4): the
"we measured preference, not sampling noise" framing is robust iff the
chosen-index agreement between T=0 and T=0.7 is >= 0.75 per model.

Usage:
    python scripts/t24_chosen_index_agreement.py \
        --batch-a runs/<T0_batch>_multiseed \
        --batch-b runs/<T07_batch>_multiseed \
        --label-a T=0 --label-b T=0.7
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RE_RUN = re.compile(
    r"^seed(?P<seed>\d{3})(?:_R\d+)?_(?P<label>[a-z][\w]*)\.jsonl$"
)


def chosen_seq(path: Path) -> list[int]:
    out: list[int] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            mc = obj.get("menu_choice") or {}
            ci = mc.get("chosen_index")
            out.append(int(ci) if ci is not None else -1)
    return out


def load_batch(batch_dir: Path) -> dict[tuple[str, int], list[int]]:
    """{(label, seed): [chosen_index per turn]}."""
    d: dict[tuple[str, int], list[int]] = {}
    for p in sorted(batch_dir.glob("*.jsonl")):
        m = RE_RUN.match(p.name)
        if not m:
            continue
        d[(m.group("label").lower(), int(m.group("seed")))] = chosen_seq(p)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-a", type=Path, required=True)
    ap.add_argument("--batch-b", type=Path, required=True)
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    args = ap.parse_args()

    a = load_batch(args.batch_a)
    b = load_batch(args.batch_b)

    per_model_match: dict[str, int] = defaultdict(int)
    per_model_total: dict[str, int] = defaultdict(int)
    common = sorted(set(a) & set(b))
    for key in common:
        label, _seed = key
        sa, sb = a[key], b[key]
        for x, y in zip(sa, sb):
            per_model_total[label] += 1
            if x == y and x != -1:
                per_model_match[label] += 1

    print(f"# T2.4 chosen-index agreement: {args.label_a} vs {args.label_b}")
    print(f"# batch A = {args.batch_a.name}")
    print(f"# batch B = {args.batch_b.name}")
    print(f"# common (seed,model) cells: {len(common)}")
    print()
    print("| model | turns compared | agreement | pre-reg >=0.75 |")
    print("|---|---:|---:|:---:|")
    for label in sorted(per_model_total):
        tot = per_model_total[label]
        mt = per_model_match[label]
        frac = mt / tot if tot else float("nan")
        verdict = "PASS" if frac >= 0.75 else "FAIL"
        print(f"| {label} | {tot} | {frac:.3f} ({mt}/{tot}) | {verdict} |")


if __name__ == "__main__":
    main()
