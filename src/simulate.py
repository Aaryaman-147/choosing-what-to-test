"""
SM03 — the simulation driver.

One run = one pool, one policy, one seed:

    1. take the 40 initial molecules by lowest SHA256(CID+seed)
    2. reveal their labels through the oracle
    3. five times:
         fit the surrogate on everything revealed so far
         score every unqueried molecule in the pool
         order 20 by the policy
         hand the ordered IDs *and* the pre-reveal predictions to the oracle
         receive only those 20 labels
    4. emit the discovery curve, per-round diagnostics and the full ledger

The model is refit from scratch each round rather than updated, so a round's
predictions depend only on labels legitimately revealed before it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import initial_selection
from .features import FeatureSet, mean_pairwise_similarity, max_similarity_to_set
from .models import ForestSurrogate
from .oracle import RevealOracle, build_oracle
from .policies import PolicyConfig, SelectionContext, make_selector


@dataclass
class RoundDiagnostics:
    round_index: int
    n_labelled_before: int
    n_actives_before: int
    batch_actives: int
    batch_hit_rate: float
    mean_predicted_prob: float
    mean_uncertainty: float
    batch_internal_similarity: float
    batch_max_sim_to_labelled: float
    batch_scaffolds: int
    fit_seconds: float


@dataclass
class RunResult:
    pool_name: str
    policy: PolicyConfig
    seed: int
    hash_variant: str
    pool_size: int
    pool_actives: int
    discovery_curve: list[int]
    batch_hit_rates: list[float]
    diagnostics: list[RoundDiagnostics]
    oracle: RevealOracle = field(repr=False)
    wall_seconds: float = 0.0

    @property
    def total_found(self) -> int:
        return self.discovery_curve[-1] if self.discovery_curve else 0

    @property
    def found_after_batches(self) -> int:
        """Actives discovered by acquisition, excluding the free initial 40."""
        if len(self.discovery_curve) < 2:
            return 0
        return self.discovery_curve[-1] - self.discovery_curve[0]

    def summary_row(self) -> dict:
        return {
            "pool": self.pool_name,
            "policy": self.policy.name,
            "seed": self.seed,
            "initial_actives": self.discovery_curve[0] if self.discovery_curve else 0,
            "acquired_actives": self.found_after_batches,
            "total_actives_found": self.total_found,
            "pool_actives": self.pool_actives,
            "recall_of_pool_actives": self.total_found / self.pool_actives if self.pool_actives else float("nan"),
            "mean_batch_hit_rate": float(np.mean(self.batch_hit_rates)) if self.batch_hit_rates else float("nan"),
            "wall_seconds": round(self.wall_seconds, 2),
        }


def run_simulation(
    pool_name: str,
    pool_ids: list[str],
    label_lookup: dict[str, int],
    features: FeatureSet,
    smiles_lookup: dict[str, str],
    policy: PolicyConfig,
    seed: int,
    hash_variant: str = "plain",
    n_initial: int = 40,
    n_batches: int = 5,
    batch_size: int = 20,
    n_estimators: int = 500,
) -> RunResult:
    t_start = time.perf_counter()

    oracle = build_oracle(
        pool_ids,
        label_lookup,
        n_initial=n_initial,
        n_batches=n_batches,
        batch_size=batch_size,
    )

    pool_actives = sum(label_lookup[c] for c in pool_ids)

    # Round 0 — the prescribed starting set. No model involved.
    start = initial_selection(pool_ids, seed, n_initial, hash_variant)
    oracle.reveal_initial(start)

    pool_features = features.subset(list(pool_ids))
    selector = make_selector(policy)
    rng = np.random.default_rng(seed)

    diagnostics: list[RoundDiagnostics] = []

    for round_index in range(1, n_batches + 1):
        revealed = oracle.revealed
        labelled_ids = list(revealed)
        n_actives_before = sum(revealed.values())

        candidates = oracle.unqueried
        if len(candidates) < batch_size:
            raise RuntimeError(
                f"{pool_name}/{policy.name}/seed{seed}: only {len(candidates)} unqueried "
                f"molecules left, cannot fill a batch of {batch_size}"
            )

        labelled_features = pool_features.subset(labelled_ids)
        candidate_features = pool_features.subset(candidates)

        t_fit = time.perf_counter()
        surrogate = ForestSurrogate(n_estimators=n_estimators, random_state=seed)
        surrogate.fit(labelled_features.X, np.array([revealed[c] for c in labelled_ids]))
        posterior = surrogate.posterior(candidates, candidate_features.X)
        fit_seconds = time.perf_counter() - t_fit

        ctx = SelectionContext(
            candidate_ids=candidates,
            posterior=posterior,
            features=candidate_features,
            labelled_fingerprints=labelled_features.fingerprints,
            rng=rng,
        )
        ordered = selector(ctx, batch_size)

        preds, uncs = posterior.as_dicts()
        # Everything above this line saw no label for `ordered`.
        labels = oracle.reveal_batch(ordered, preds, uncs)

        batch_features = pool_features.subset(ordered)
        batch_actives = sum(labels.values())
        diagnostics.append(
            RoundDiagnostics(
                round_index=round_index,
                n_labelled_before=len(labelled_ids),
                n_actives_before=n_actives_before,
                batch_actives=batch_actives,
                batch_hit_rate=batch_actives / batch_size,
                mean_predicted_prob=float(np.mean([preds[c] for c in ordered])),
                mean_uncertainty=float(np.mean([uncs[c] for c in ordered])),
                batch_internal_similarity=mean_pairwise_similarity(batch_features.fingerprints),
                batch_max_sim_to_labelled=float(
                    np.mean(
                        max_similarity_to_set(
                            batch_features.fingerprints, labelled_features.fingerprints
                        )
                    )
                ),
                batch_scaffolds=_scaffold_count([smiles_lookup[c] for c in ordered]),
                fit_seconds=round(fit_seconds, 3),
            )
        )

    return RunResult(
        pool_name=pool_name,
        policy=policy,
        seed=seed,
        hash_variant=hash_variant,
        pool_size=len(pool_ids),
        pool_actives=pool_actives,
        discovery_curve=oracle.discovery_curve(),
        batch_hit_rates=oracle.batch_hit_rates(),
        diagnostics=diagnostics,
        oracle=oracle,
        wall_seconds=time.perf_counter() - t_start,
    )


def _scaffold_count(smiles: list[str]) -> int:
    from .data import bemis_murcko_scaffold

    return len({bemis_murcko_scaffold(s) for s in smiles})


def run_seeds(
    pool_name: str,
    pool_ids: list[str],
    label_lookup: dict[str, int],
    features: FeatureSet,
    smiles_lookup: dict[str, str],
    policy: PolicyConfig,
    seeds: tuple[int, ...] = (42, 43, 44),
    ledger_dir: str | Path | None = None,
    **kwargs,
) -> list[RunResult]:
    results = []
    for seed in seeds:
        result = run_simulation(
            pool_name, pool_ids, label_lookup, features, smiles_lookup, policy, seed, **kwargs
        )
        if ledger_dir is not None:
            path = Path(ledger_dir) / f"{pool_name}_{policy.name}_seed{seed}.json"
            result.oracle.write_ledger(
                path,
                metadata={
                    "pool": pool_name,
                    "policy": policy.to_dict(),
                    "seed": seed,
                    "hash_variant": result.hash_variant,
                },
            )
        results.append(result)
    return results


def results_frame(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame([r.summary_row() for r in results])


def curves_frame(results: list[RunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for round_index, cumulative in enumerate(r.discovery_curve):
            rows.append(
                {
                    "pool": r.pool_name,
                    "policy": r.policy.name,
                    "seed": r.seed,
                    "round": round_index,
                    "molecules_tested": r.oracle.n_initial + max(0, round_index) * r.oracle.batch_size
                    if round_index > 0
                    else r.oracle.n_initial,
                    "cumulative_actives": cumulative,
                }
            )
    return pd.DataFrame(rows)
