"""
SM03 — acquisition policies.

Every policy returns an *ordered* list of 20 compound IDs. The order is part of
the deliverable: it records the policy's ranked preference before any label
came back, which is what makes the simulation auditable after the fact.

The design axis the challenge actually scores is the activity / diversity /
uncertainty trade-off, so the policies are laid out as a ladder:

    random          no information          — the baseline to beat
    greedy          activity only           — exploitation ceiling, redundancy risk
    uncertainty     information only        — pure exploration, wastes budget on
                                              molecules that are uninformative
                                              because they are obviously inactive
    ucb             activity + uncertainty  — the usual compromise
    greedy_diverse  activity + diversity    — guards against a batch of
                                              near-duplicate analogues
    ucb_diverse     all three               — the intended submission

Batch construction matters as much as the score. Picking the top 20 by score
independently is the classic active-learning mistake: within one round the
model cannot update, so the 20 highest-scoring molecules are frequently near
neighbours of each other and the batch carries ~1 molecule of information.
Every diversity-aware policy here selects sequentially and penalises each
candidate by its similarity to what is already in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from rdkit import DataStructs

from .features import FeatureSet
from .models import Posterior


@dataclass(frozen=True)
class PolicyConfig:
    """A frozen, serialisable policy definition.

    The brief requires the policy to be chosen on validation and frozen before
    the test simulation. Serialising this object *is* the freeze.
    """

    name: str
    beta: float = 0.0  # weight on epistemic uncertainty
    diversity_lambda: float = 0.0  # weight on similarity-to-batch penalty
    use_entropy: bool = False  # score on entropy instead of P(active)
    random_state: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "beta": self.beta,
            "diversity_lambda": self.diversity_lambda,
            "use_entropy": self.use_entropy,
            "random_state": self.random_state,
        }


@dataclass
class SelectionContext:
    """Everything a policy may legally consult."""

    candidate_ids: list[str]
    posterior: Posterior
    features: FeatureSet  # candidates only
    labelled_fingerprints: list = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


Selector = Callable[[SelectionContext, int], list[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_score(cfg: PolicyConfig, post: Posterior) -> np.ndarray:
    """Score before any diversity penalty."""
    if cfg.use_entropy:
        core = post.entropy
    else:
        core = post.probability
    return core + cfg.beta * post.uncertainty


def _sequential_select(
    cfg: PolicyConfig,
    ctx: SelectionContext,
    k: int,
    base: np.ndarray,
) -> list[str]:
    """Greedy sequential selection with a similarity-to-batch penalty."""
    candidate_fps = ctx.features.fingerprints
    n = len(ctx.candidate_ids)
    k = min(k, n)

    chosen_idx: list[int] = []
    # Penalise similarity to molecules already labelled: re-measuring a close
    # analogue of something we have already tested buys little.
    if cfg.diversity_lambda > 0 and ctx.labelled_fingerprints:
        from .features import max_similarity_to_set

        sim_to_labelled = max_similarity_to_set(candidate_fps, ctx.labelled_fingerprints)
    else:
        sim_to_labelled = np.zeros(n, dtype=np.float64)

    # Running max similarity to the batch assembled so far.
    sim_to_batch = np.zeros(n, dtype=np.float64)
    available = np.ones(n, dtype=bool)

    for _ in range(k):
        penalty = cfg.diversity_lambda * np.maximum(sim_to_batch, sim_to_labelled)
        score = base - penalty
        score[~available] = -np.inf
        pick = int(np.argmax(score))
        chosen_idx.append(pick)
        available[pick] = False

        if cfg.diversity_lambda > 0:
            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(candidate_fps[pick], candidate_fps),
                dtype=np.float64,
            )
            sim_to_batch = np.maximum(sim_to_batch, sims)

    return [ctx.candidate_ids[i] for i in chosen_idx]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def select_random(ctx: SelectionContext, k: int) -> list[str]:
    """Uninformed baseline. Its discovery curve is the pool's base rate."""
    idx = ctx.rng.permutation(len(ctx.candidate_ids))[:k]
    return [ctx.candidate_ids[i] for i in idx]


def make_selector(cfg: PolicyConfig) -> Selector:
    if cfg.name == "random":
        return select_random

    def selector(ctx: SelectionContext, k: int) -> list[str]:
        base = _base_score(cfg, ctx.posterior)
        return _sequential_select(cfg, ctx, k, base)

    return selector


# The ladder swept on validation.
POLICY_LADDER: tuple[PolicyConfig, ...] = (
    PolicyConfig(name="random"),
    PolicyConfig(name="greedy"),
    PolicyConfig(name="uncertainty", use_entropy=True),
    PolicyConfig(name="ucb_b05", beta=0.5),
    PolicyConfig(name="ucb_b10", beta=1.0),
    PolicyConfig(name="greedy_diverse_l03", diversity_lambda=0.3),
    PolicyConfig(name="greedy_diverse_l05", diversity_lambda=0.5),
    PolicyConfig(name="ucb_diverse_b05_l03", beta=0.5, diversity_lambda=0.3),
    PolicyConfig(name="ucb_diverse_b05_l05", beta=0.5, diversity_lambda=0.5),
    PolicyConfig(name="ucb_diverse_b10_l05", beta=1.0, diversity_lambda=0.5),
)


def policy_by_name(name: str) -> PolicyConfig:
    for cfg in POLICY_LADDER:
        if cfg.name == name:
            return cfg
    raise KeyError(f"unknown policy {name!r}; available: {[c.name for c in POLICY_LADDER]}")
