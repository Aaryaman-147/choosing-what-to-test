"""
The 3-seed sweep put greedy, ucb and the diversity-aware variants inside one
standard deviation of each other. Declaring a winner on that evidence would be
fitting the seed noise.

This script does two things:
  1. re-runs the contenders over 10 seeds, so the comparison has the resolution
     to either separate them or confirm the tie;
  2. scores the secondary axes the brief asks about — batch diversity and model
     uncertainty — which is the legitimate basis for choosing among policies
     whose yield is indistinguishable.

Seeds beyond 42/43/44 are used ONLY here, for policy selection on the
validation pool. The reported test simulation uses the prescribed three.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd
from scipy import stats

from sm03.src import data, features
from sm03.src.policies import policy_by_name
from sm03.src.simulate import run_seeds

CONTENDERS = ("random", "uncertainty", "greedy", "ucb_b05", "greedy_diverse_l05", "ucb_diverse_b05_l03")
SELECTION_SEEDS = tuple(range(42, 52))  # 10 seeds, validation only
OUT = ROOT / "outputs"


def main(source: str, starter_dir: str | None = None) -> None:
    df = data.load_source(source)
    pools = data.resolve_pools(df, starter_dir)
    label_lookup = dict(zip(df["cid"], df["Class"]))
    smiles_lookup = dict(zip(df["cid"], df["canonical_smiles"]))
    fs = features.featurise(df["cid"].tolist(), df["canonical_smiles"].tolist())
    val_ids = list(pools.validation)

    print(f"validation pool {len(val_ids)}, {sum(label_lookup[c] for c in val_ids)} active")
    print(f"policy selection over {len(SELECTION_SEEDS)} seeds {SELECTION_SEEDS}\n")

    rows, diag_rows = [], []
    for name in CONTENDERS:
        cfg = policy_by_name(name)
        results = run_seeds(
            "validation_tiebreak",
            val_ids,
            label_lookup,
            fs,
            smiles_lookup,
            cfg,
            seeds=SELECTION_SEEDS,
            n_estimators=300,
        )
        acquired = np.array([r.found_after_batches for r in results], dtype=float)
        sims, scaffs, uncs = [], [], []
        for r in results:
            sims.extend(d.batch_internal_similarity for d in r.diagnostics)
            scaffs.extend(d.batch_scaffolds for d in r.diagnostics)
            uncs.extend(d.mean_uncertainty for d in r.diagnostics)
            for d in r.diagnostics:
                diag_rows.append(
                    {
                        "policy": name,
                        "seed": r.seed,
                        "round": d.round_index,
                        "hit_rate": d.batch_hit_rate,
                        "batch_internal_sim": d.batch_internal_similarity,
                        "scaffolds": d.batch_scaffolds,
                        "mean_unc": d.mean_uncertainty,
                    }
                )
        rows.append(
            {
                "policy": name,
                "acquired_mean": acquired.mean(),
                "acquired_sd": acquired.std(ddof=1),
                "acquired_sem": stats.sem(acquired),
                "acquired_min": acquired.min(),
                "acquired_max": acquired.max(),
                "batch_internal_sim": float(np.mean(sims)),
                "scaffolds_per_20": float(np.mean(scaffs)),
                "mean_uncertainty": float(np.mean(uncs)),
                "_raw": acquired,
            }
        )
        print(
            f"  {name:22s} {acquired.mean():5.1f} +/- {stats.sem(acquired):4.2f} (sem)   "
            f"sim {np.mean(sims):.3f}  scaffolds {np.mean(scaffs):.1f}/20"
        )

    table = pd.DataFrame(rows).sort_values("acquired_mean", ascending=False)
    raw = {r["policy"]: r["_raw"] for r in rows}
    table = table.drop(columns=["_raw"])

    print("\n== yield, 10 seeds ==")
    print(table[["policy", "acquired_mean", "acquired_sd", "acquired_sem", "acquired_min", "acquired_max"]].to_string(index=False))

    print("\n== secondary axes (mean over all rounds, all seeds) ==")
    print(table[["policy", "batch_internal_sim", "scaffolds_per_20", "mean_uncertainty"]].to_string(index=False))

    # Paired tests against the plain greedy exploitation ceiling: same seeds, so
    # pairing removes the seed-to-seed variation that swamped the 3-seed run.
    print("\n== paired t-test vs greedy (same seeds) ==")
    base = raw["greedy"]
    for name, vals in raw.items():
        if name == "greedy":
            continue
        t, p = stats.ttest_rel(vals, base)
        delta = vals.mean() - base.mean()
        verdict = "separates" if p < 0.05 else "indistinguishable"
        print(f"  {name:22s} delta {delta:+5.1f}  t {t:+5.2f}  p {p:.3f}  -> {verdict}")

    table.to_csv(OUT / "tiebreak_summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "tiebreak_diagnostics.csv", index=False)
    print(f"\nwrote {OUT / 'tiebreak_summary.csv'}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "bace_raw" / "bace.csv")
    main(src, sys.argv[2] if len(sys.argv) > 2 else None)
