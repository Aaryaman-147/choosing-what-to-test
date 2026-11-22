"""
SM03 — frozen-policy test simulation.

Runs the policy frozen on validation against the test pool, over the prescribed
seeds 42/43/44, alongside the two baselines that make the result interpretable:

    random   the uninformed floor (= the pool's base rate)
    greedy   the pure-exploitation yield ceiling

Also emits the final twenty-compound recommendation: after the budget is spent,
the surrogate is refit on all 140 revealed labels and ranks the molecules that
were never queried. Those twenty are a prospective recommendation, not a
measured result, and the output file labels them as such.

The organiser runs the same initial-40/five-batch procedure inside their own
375-compound test pool. Nothing here reads a test label except through the
oracle's reveal interface.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd

from sm03.src import data, features
from sm03.src.features import max_similarity_to_set, mean_pairwise_similarity
from sm03.src.models import ForestSurrogate
from sm03.src.policies import policy_by_name
from sm03.src.simulate import curves_frame, results_frame, run_seeds

SEEDS = (42, 43, 44)
OUT = ROOT / "outputs"

# Frozen on validation over 10 seeds (scripts/resolve_tie.py).
#
#   ucb_diverse_b05_l03   62.6 +/- 1.12   18.42 scaffolds/batch
#   ucb_b05               62.6 +/- 1.21   17.16
#   greedy_diverse_l05    61.8 +/- 0.96   18.90
#   greedy                61.6 +/- 1.30   17.22
#   uncertainty           47.8 +/- 1.25   17.90
#   random                38.8 +/- 1.27   19.42
#
# The top four are statistically indistinguishable — every paired t-test
# against greedy returns p > 0.3. So the diversity and uncertainty terms cost
# no measurable yield, while returning 1.2 more distinct scaffolds per batch
# than greedy. Free diversity is worth taking, and the challenge asks for a
# strategy balancing activity, diversity and uncertainty. greedy is reported
# alongside as the pure-exploitation reference, random as the floor.
FROZEN = "ucb_diverse_b05_l03"
BASELINES = ("random", "greedy")


def final_recommendation(
    pool_ids, label_lookup, features_all, smiles_lookup, oracle, seed, n=20
) -> pd.DataFrame:
    """Rank the never-queried molecules with the end-of-campaign model."""
    revealed = oracle.revealed
    labelled_ids = list(revealed)
    remaining = [c for c in pool_ids if c not in revealed]

    pool_features = features_all.subset(list(pool_ids))
    labelled_features = pool_features.subset(labelled_ids)
    remaining_features = pool_features.subset(remaining)

    surrogate = ForestSurrogate(n_estimators=500, random_state=seed)
    surrogate.fit(labelled_features.X, np.array([revealed[c] for c in labelled_ids]))
    post = surrogate.posterior(remaining, remaining_features.X)

    nn_sim = max_similarity_to_set(remaining_features.fingerprints, labelled_features.fingerprints)

    frame = pd.DataFrame(
        {
            "CID": remaining,
            "canonical_smiles": [smiles_lookup[c] for c in remaining],
            "predicted_p_active": post.probability,
            "model_uncertainty": post.uncertainty,
            "nearest_labelled_tanimoto": nn_sim,
        }
    )
    # In-domain means the model has seen something chemically comparable. A
    # recommendation resting on an extrapolation is flagged rather than dropped.
    frame["in_applicability_domain"] = frame["nearest_labelled_tanimoto"] >= 0.35
    frame = frame.sort_values("predicted_p_active", ascending=False).head(n).reset_index(drop=True)
    frame.insert(0, "rank", frame.index + 1)
    frame["seed"] = seed
    frame["basis"] = "prospective model prediction — NOT a measured activity"
    return frame


def main(source: str, starter_dir: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = data.load_source(source)
    pools = data.resolve_pools(df, starter_dir)
    pools.assert_official_sizes()
    pools.assert_disjoint()

    label_lookup = dict(zip(df["cid"], df["Class"]))
    smiles_lookup = dict(zip(df["cid"], df["canonical_smiles"]))
    fs = features.featurise(df["cid"].tolist(), df["canonical_smiles"].tolist())

    test_ids = list(pools.test)
    test_actives = sum(label_lookup[c] for c in test_ids)
    base_rate = test_actives / len(test_ids)

    print(f"pool provenance : {pools.provenance}")
    print(f"test pool       : {test_actives}/{len(test_ids)} active (base rate {base_rate:.1%})")
    print(f"budget          : 40 initial + 5 x 20 = 140 of {len(test_ids)}")
    print(f"frozen policy   : {FROZEN}")
    print(f"random floor    : {100 * base_rate:.1f} actives expected per 100 acquired\n")

    ledger_dir = OUT / "test_ledgers"
    all_results, rec_frames = [], []

    for name in (FROZEN, *BASELINES):
        cfg = policy_by_name(name)
        results = run_seeds(
            "test",
            test_ids,
            label_lookup,
            fs,
            smiles_lookup,
            cfg,
            seeds=SEEDS,
            ledger_dir=ledger_dir,
            n_estimators=500,
        )
        all_results.extend(results)
        acquired = [r.found_after_batches for r in results]
        tag = "FROZEN" if name == FROZEN else "baseline"
        print(
            f"  {name:24s} [{tag:8s}] acquired {np.mean(acquired):5.1f} "
            f"+/- {np.std(acquired, ddof=1):4.2f}   {[int(a) for a in acquired]}"
        )
        if name == FROZEN:
            for r in results:
                rec_frames.append(
                    final_recommendation(
                        test_ids, label_lookup, fs, smiles_lookup, r.oracle, r.seed
                    )
                )

    summary = results_frame(all_results)
    curves = curves_frame(all_results)
    summary.to_csv(OUT / "test_summary.csv", index=False)
    curves.to_csv(OUT / "test_curves.csv", index=False)

    print("\n== discovery curves (cumulative actives, mean over 3 seeds) ==")
    pivot = (
        curves.groupby(["policy", "round"])["cumulative_actives"]
        .mean()
        .unstack("round")
        .round(1)
        .reindex([FROZEN, "greedy", "random"])
    )
    pivot.columns = [f"after_{'init' if c == 0 else f'b{c}'}" for c in pivot.columns]
    print(pivot.to_string())

    # Diversity / uncertainty analysis across the three policies.
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
    diag.to_csv(OUT / "test_diagnostics.csv", index=False)

    print("\n== diversity / uncertainty (mean over rounds and seeds) ==")
    print(
        diag.groupby("policy")[
            ["hit_rate", "mean_unc", "batch_internal_sim", "max_sim_to_labelled", "scaffolds"]
        ]
        .mean()
        .round(3)
        .reindex([FROZEN, "greedy", "random"])
        .to_string()
    )

    # Final 20: consensus across seeds, then the seed-42 ranking as the single
    # submittable panel.
    recs = pd.concat(rec_frames, ignore_index=True)
    recs.to_csv(OUT / "final_20_per_seed.csv", index=False)

    pooled = (
        recs.groupby(["CID", "canonical_smiles"])
        .agg(
            times_in_top20=("rank", "size"),
            mean_rank=("rank", "mean"),
            mean_p_active=("predicted_p_active", "mean"),
            mean_uncertainty=("model_uncertainty", "mean"),
            nearest_labelled_tanimoto=("nearest_labelled_tanimoto", "mean"),
        )
        .reset_index()
        # Rank on predicted activity; seed agreement is reported alongside as a
        # robustness signal rather than used as the primary sort, which would
        # otherwise push a p=0.50 molecule above a p=0.86 one.
        .sort_values(["mean_p_active", "times_in_top20"], ascending=[False, False])
        .reset_index(drop=True)
    )
    pooled["in_applicability_domain"] = pooled["nearest_labelled_tanimoto"] >= 0.35

    # The raw ranking contains near-duplicates: exploitation returns tight
    # analogue series, so 20 top-ranked molecules can cover far fewer than 20
    # chemotypes. Spending a 20-slot budget on redundant structures wastes it,
    # so the delivered panel is filtered to one representative per cluster at
    # Tanimoto < 0.70, taking the highest-ranked member of each.
    pooled_fs = fs.subset(pooled["CID"].tolist())
    kept_idx: list[int] = []
    for i in range(len(pooled)):
        if not kept_idx:
            kept_idx.append(i)
            continue
        sims = np.asarray(
            [
                features.tanimoto_to_set(pooled_fs.fingerprints[i], [pooled_fs.fingerprints[j]])[0]
                for j in kept_idx
            ]
        )
        if sims.max() < 0.70:
            kept_idx.append(i)
        if len(kept_idx) == 20:
            break

    consensus = pooled.iloc[kept_idx].reset_index(drop=True)
    consensus.insert(0, "rank", consensus.index + 1)
    consensus["basis"] = "prospective model prediction — NOT a measured activity"
    consensus.to_csv(OUT / "final_20_recommendation.csv", index=False)
    pooled.head(40).to_csv(OUT / "final_20_unfiltered_ranking.csv", index=False)

    raw20 = pooled.head(20)
    print("\n== redundancy in the unfiltered top 20 ==")
    raw_fs = fs.subset(raw20["CID"].tolist())
    print(f"  mean pairwise Tanimoto      {mean_pairwise_similarity(raw_fs.fingerprints):.3f}")
    print(f"  distinct Bemis-Murcko       {len({data.bemis_murcko_scaffold(s) for s in raw20['canonical_smiles']})}/20")
    kept_fs = fs.subset(consensus["CID"].tolist())
    print("  after Tanimoto < 0.70 filtering:")
    print(f"  mean pairwise Tanimoto      {mean_pairwise_similarity(kept_fs.fingerprints):.3f}")
    print(f"  distinct Bemis-Murcko       {len({data.bemis_murcko_scaffold(s) for s in consensus['canonical_smiles']})}/{len(consensus)}")

    print("\n== final twenty-compound recommendation (diversity-filtered) ==")
    print(
        consensus[
            ["rank", "CID", "times_in_top20", "mean_p_active", "mean_uncertainty",
             "nearest_labelled_tanimoto", "in_applicability_domain"]
        ].round(3).to_string(index=False)
    )
    n_out = int((~consensus["in_applicability_domain"]).sum())
    print(f"\n  {n_out}/{len(consensus)} fall outside the applicability domain (nearest labelled Tanimoto < 0.35)")

    (OUT / "test_run_manifest.json").write_text(
        json.dumps(
            {
                "frozen_policy": policy_by_name(FROZEN).to_dict(),
                "baselines": list(BASELINES),
                "seeds": list(SEEDS),
                "pool_provenance": pools.provenance,
                "test_pool_size": len(test_ids),
                "test_pool_actives": test_actives,
                "base_rate": base_rate,
                "budget": {"initial": 40, "batches": 5, "batch_size": 20},
                "mean_acquired": {
                    p: float(summary[summary.policy == p]["acquired_actives"].mean())
                    for p in summary.policy.unique()
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT / 'test_run_manifest.json'}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "bace_raw" / "bace.csv")
    main(src, sys.argv[2] if len(sys.argv) > 2 else None)
