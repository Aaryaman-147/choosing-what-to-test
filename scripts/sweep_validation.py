"""
Validation sweep — policy selection.

Runs the full policy ladder on the validation pool across seeds 42/43/44 and
writes the comparison table plus every ledger. The winner chosen here is
frozen to JSON and is the only policy permitted to touch the test pool.

Nothing in this script reads a test-pool label.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd

from sm03.src import data, features
from sm03.src.policies import POLICY_LADDER
from sm03.src.simulate import curves_frame, results_frame, run_seeds

SEEDS = (42, 43, 44)
OUT = ROOT / "outputs"


def environment_record() -> dict:
    import rdkit
    import sklearn

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip() or None
    except Exception:
        git_head = None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rdkit": rdkit.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_head": git_head,
        "seeds": list(SEEDS),
    }


def main(source: str, starter_dir: str | None = None, n_estimators: int = 500) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    df = data.load_source(source)
    pools = data.resolve_pools(df, starter_dir)
    pools.assert_official_sizes()
    pools.assert_disjoint()

    print(f"source           : {source}")
    print(f"compounds        : {len(df)}  ({df['scaffold'].nunique()} scaffolds)")
    print(f"pools            : {pools.sizes()}")
    print(f"pool provenance  : {pools.provenance}")

    label_lookup = dict(zip(df["cid"], df["Class"]))
    smiles_lookup = dict(zip(df["cid"], df["canonical_smiles"]))

    val_ids = list(pools.validation)
    val_actives = sum(label_lookup[c] for c in val_ids)
    base_rate = val_actives / len(val_ids)
    print(f"validation pool  : {val_actives}/{len(val_ids)} active (base rate {base_rate:.1%})")
    print(f"random expectation for 100 acquired: {100 * base_rate:.1f} actives\n")

    print("featurising ...")
    fs = features.featurise(df["cid"].tolist(), df["canonical_smiles"].tolist())
    print(f"  X {fs.X.shape}\n")

    ledger_dir = OUT / "validation_ledgers"
    all_results = []
    for cfg in POLICY_LADDER:
        t = time.perf_counter()
        results = run_seeds(
            "validation",
            val_ids,
            label_lookup,
            fs,
            smiles_lookup,
            cfg,
            seeds=SEEDS,
            ledger_dir=ledger_dir,
            n_estimators=n_estimators,
        )
        all_results.extend(results)
        acquired = [r.found_after_batches for r in results]
        print(
            f"  {cfg.name:24s} acquired {np.mean(acquired):5.1f} +/- {np.std(acquired):4.1f}"
            f"   ({[int(a) for a in acquired]})   {time.perf_counter() - t:5.1f}s"
        )

    summary = results_frame(all_results)
    curves = curves_frame(all_results)
    summary.to_csv(OUT / "validation_summary.csv", index=False)
    curves.to_csv(OUT / "validation_curves.csv", index=False)

    # Rank by mean actives acquired through the five batches. The initial 40 are
    # hash-determined and identical across policies, so they are excluded —
    # otherwise every policy gets the same free head start baked into its score.
    ranking = (
        summary.groupby("policy")["acquired_actives"]
        .agg(["mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    print("\n== ranking by actives acquired over 100 measurements ==")
    print(ranking.to_string())

    random_mean = float(ranking.loc["random", "mean"])
    best_policy = str(ranking.index[0])
    best_mean = float(ranking.loc[best_policy, "mean"])
    print(f"\nrandom baseline : {random_mean:.1f}")
    print(f"best policy     : {best_policy} at {best_mean:.1f}  ({best_mean / random_mean:.2f}x)")

    # Per-round diagnostics for the winner, aggregated over seeds.
    diag_rows = []
    for r in all_results:
        for d in r.diagnostics:
            diag_rows.append(
                {
                    "policy": r.policy.name,
                    "seed": r.seed,
                    "round": d.round_index,
                    "hit_rate": d.batch_hit_rate,
                    "mean_pred": d.mean_predicted_prob,
                    "mean_unc": d.mean_uncertainty,
                    "batch_internal_sim": d.batch_internal_similarity,
                    "max_sim_to_labelled": d.batch_max_sim_to_labelled,
                    "scaffolds": d.batch_scaffolds,
                }
            )
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "validation_diagnostics.csv", index=False)

    print(f"\n== per-round profile, {best_policy} (mean over seeds) ==")
    print(
        diag[diag.policy == best_policy]
        .groupby("round")[["hit_rate", "mean_pred", "mean_unc", "batch_internal_sim", "scaffolds"]]
        .mean()
        .round(3)
        .to_string()
    )

    frozen = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_policy": next(c for c in POLICY_LADDER if c.name == best_policy).to_dict(),
        "selected_on": "validation pool only",
        "validation_mean_acquired": best_mean,
        "validation_random_baseline": random_mean,
        "lift_over_random": best_mean / random_mean,
        "pool_provenance": pools.provenance,
        "n_estimators": n_estimators,
        "environment": environment_record(),
        "ranking": ranking.reset_index().to_dict(orient="records"),
    }
    (OUT / "frozen_policy.json").write_text(json.dumps(frozen, indent=2))

    print(f"\nwrote {OUT / 'frozen_policy.json'}")
    print(f"total wall time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "bace_raw" / "bace.csv")
    starter = sys.argv[2] if len(sys.argv) > 2 else None
    main(src, starter)
