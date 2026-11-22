"""
Harness smoke test.

Proves the pieces fit together and, more importantly, that the reveal protocol
actually bites: the assertions below are the ones a judge would want to see
pass. Runs on the dry-run scaffold split, so no number it prints is a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np

from sm03.src import data, features
from sm03.src.oracle import RevealViolation, build_oracle
from sm03.src.policies import POLICY_LADDER, policy_by_name
from sm03.src.simulate import curves_frame, results_frame, run_seeds

SOURCE = ROOT / "data" / "bace.csv"


def check_protocol(pool_ids, label_lookup):
    print("\n== reveal protocol guards ==")
    oracle = build_oracle(pool_ids, label_lookup)

    start = data.initial_selection(pool_ids, 42, 40)
    oracle.reveal_initial(start)
    print(f"  initial 40 revealed, actives among them: {sum(oracle.revealed.values())}")

    def expect_violation(label, fn):
        try:
            fn()
        except RevealViolation as exc:
            print(f"  OK  {label}: {exc}")
        else:
            raise AssertionError(f"FAILED to block: {label}")

    fresh = oracle.unqueried
    expect_violation("second initial reveal", lambda: oracle.reveal_initial(start))
    expect_violation(
        "re-query of revealed molecule",
        lambda: oracle.reveal_batch(start[:20], {c: 0.5 for c in start[:20]}),
    )
    expect_violation(
        "wrong batch size",
        lambda: oracle.reveal_batch(fresh[:19], {c: 0.5 for c in fresh[:19]}),
    )
    expect_violation(
        "missing predictions",
        lambda: oracle.reveal_batch(fresh[:20], {c: 0.5 for c in fresh[:10]}),
    )
    expect_violation(
        "ID outside pool",
        lambda: oracle.reveal_batch(
            fresh[:19] + ["BACE_DOES_NOT_EXIST"],
            {c: 0.5 for c in fresh[:19] + ["BACE_DOES_NOT_EXIST"]},
        ),
    )

    # Exhaust the budget legitimately, then confirm the 6th batch is refused.
    for _ in range(5):
        nxt = oracle.unqueried[:20]
        oracle.reveal_batch(nxt, {c: 0.5 for c in nxt})
    expect_violation(
        "sixth batch past budget",
        lambda: oracle.reveal_batch(
            oracle.unqueried[:20], {c: 0.5 for c in oracle.unqueried[:20]}
        ),
    )
    assert len(oracle.revealed) == 140, len(oracle.revealed)
    print(f"  budget honoured: {len(oracle.revealed)} labels revealed of {len(pool_ids)} in pool")


def main() -> None:
    print(f"source: {SOURCE}")
    df = data.load_source(SOURCE)
    print(f"loaded {len(df)} compounds, {df['scaffold'].nunique()} Bemis-Murcko scaffolds")
    print(f"class balance: {df['Class'].value_counts().to_dict()}")

    pools = data.resolve_pools(df, starter_dir=None)
    print(f"\npools: {pools.sizes()}")
    print(f"provenance: {pools.provenance}")
    pools.assert_official_sizes()
    pools.assert_disjoint()
    print("  sizes match the 822/316/375 spec and partitions are disjoint")

    # How much does the SHA256 concatenation ambiguity actually matter?
    print("\n== initial-40 hash variant overlap vs 'plain' ==")
    for seed in (42, 43, 44):
        agree = data.initial_selection_agreement(pools.development, seed)
        print(f"  seed {seed}: " + ", ".join(f"{k}={v:.0%}" for k, v in agree.items()))

    label_lookup = dict(zip(df["cid"], df["Class"]))
    smiles_lookup = dict(zip(df["cid"], df["canonical_smiles"]))

    check_protocol(list(pools.development), label_lookup)

    print("\n== featurising ==")
    fs = features.featurise(df["cid"].tolist(), df["canonical_smiles"].tolist())
    print(f"  X shape {fs.X.shape} ({fs.fingerprint_matrix.shape[1]} fp bits + {len(fs.descriptor_names)} descriptors)")

    print("\n== two-policy dry run on the validation pool, seeds 42/43/44 ==")
    val_ids = list(pools.validation)
    base_rate = sum(label_lookup[c] for c in val_ids) / len(val_ids)
    print(f"  pool actives {sum(label_lookup[c] for c in val_ids)}/{len(val_ids)} (base rate {base_rate:.1%})")
    print(f"  random expectation for 100 acquired molecules: {100 * base_rate:.1f} actives\n")

    ledger_dir = ROOT / "outputs" / "smoke_ledgers"
    all_results = []
    for name in ("random", "ucb_diverse_b05_l05"):
        cfg = policy_by_name(name)
        results = run_seeds(
            "validation_dryrun",
            val_ids,
            label_lookup,
            fs,
            smiles_lookup,
            cfg,
            ledger_dir=ledger_dir,
            n_estimators=200,
        )
        all_results.extend(results)
        curve = np.mean([r.discovery_curve for r in results], axis=0)
        print(f"  {name:24s} mean curve {[round(float(x), 1) for x in curve]}")

    print()
    print(results_frame(all_results).to_string(index=False))
    print(f"\nledgers written to {ledger_dir}")

    print("\n== per-round diagnostics, ucb_diverse_b05_l05 seed 42 ==")
    target = next(r for r in all_results if r.policy.name == "ucb_diverse_b05_l05" and r.seed == 42)
    for d in target.diagnostics:
        print(
            f"  round {d.round_index}: hit {d.batch_hit_rate:.0%}  "
            f"pred {d.mean_predicted_prob:.2f}  unc {d.mean_uncertainty:.3f}  "
            f"batch-internal-sim {d.batch_internal_similarity:.2f}  "
            f"scaffolds {d.batch_scaffolds}/20  fit {d.fit_seconds}s"
        )

    print("\nharness OK")


if __name__ == "__main__":
    main()
