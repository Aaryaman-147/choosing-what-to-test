# SM03 — BACE1 sequential selection: results

**Scope.** Retrospective simulation of the SciSprint 2026 "Molecule Masters" SM03
challenge: from a 375-compound held-out test pool, choose 40 initial molecules
by SHA256 hash, then five sequential batches of 20 using only previously
revealed activity. Frozen policy: `ucb_diverse_b05_l03` (UCB on predicted
activity with tree-disagreement uncertainty, plus a Tanimoto diversity penalty,
selected on validation only). Baselines: plain greedy exploitation and random.

## 1. Discovery curves against the random floor

Test pool: 375 compounds, 155 active (base rate 41.3%). Budget: 40 initial +
5 x 20 = 140 tested. Mean over seeds 42/43/44:

| policy | actives acquired per 100 | per seed |
|---|---|---|
| UCB + diversity (frozen) | **75.0** | 77, 75, 73 |
| greedy | 70.0 | 75, 67, 68 |
| random | 39.3 | 37, 45, 36 |

The random floor is the base rate (39.3 ≈ 41.3, the small gap is the hash-seeded
initial 40). The frozen policy finds **1.9x** as many actives per 100 tested as
random; greedy is 1.78x. The discovery curves (figure, panel A) show the frozen
policy separating from random from the first batch and holding a steady ~35-40
actives-per-100 advantage through the fifth batch — the gap does not erode as
the pool of unqueried molecules shrinks.

## 2. Which policy differences are real and which are noise

Three seeds cannot resolve small differences (the spread is ~2-5 actives), so
policy selection used **ten** seeds (42-51) on validation with paired t-tests
against plain greedy (same seeds, so seed-to-seed variation is removed):

| policy | acquired (10 seeds) | ±sem | paired t vs greedy | p |
|---|---|---|---|---|
| UCB + diversity (frozen) | 62.6 | 1.12 | +0.97 | 0.36 |
| UCB | 62.6 | 1.21 | +1.02 | 0.33 |
| greedy + diversity | 61.8 | 0.96 | +0.21 | 0.84 |
| greedy | 61.6 | 1.30 | — | — |
| uncertainty only | 47.8 | 1.25 | -7.28 | <0.001 |
| random | 38.8 | 1.27 | -12.45 | <0.001 |

**Real:** exploitation (all four of greedy/UCB ± diversity) beats
uncertainty-only by 13.8 actives (p < 0.001), and both beat random by ~23
(p < 0.001). The 75 vs 39 headline is a genuine, large effect.

**Noise:** the four exploitation-driven policies are statistically
indistinguishable from one another — every paired comparison against greedy has
p > 0.3. The +1.0 actives for the frozen policy over greedy is seed noise, not
a yield gain. **The sophistication bought no yield.**

## 3. What the diversity and uncertainty terms actually buy

Secondary axes, mean over 50 rounds x 10 seeds:

| policy | batch-internal Tanimoto | distinct scaffolds / 20 | mean uncertainty |
|---|---|---|---|
| UCB + diversity (frozen) | 0.222 | 18.42 | 0.448 |
| UCB | 0.243 | 17.16 | 0.451 |
| greedy + diversity | 0.209 | 18.90 | 0.447 |
| greedy | 0.245 | 17.22 | 0.452 |
| uncertainty only | 0.195 | 17.90 | 0.497 |
| random | 0.166 | 19.42 | 0.430 |

- **The diversity term buys chemical breadth at zero yield cost.** Adding it to
  greedy lifts scaffolds/batch from 17.2 to 18.9 (+1.7) and drops internal
  similarity from 0.245 to 0.209, with no yield penalty (Δ +0.2, p 0.84).
  Adding it to UCB does the same (17.2→18.4) with no yield penalty (Δ +1.0,
  p 0.36). That is the entire justification for the frozen policy: free
  diversity, same yield.
- **The uncertainty term does not buy yield.** UCB and greedy have identical
  mean uncertainty in the selected batches (0.448 vs 0.452) — the UCB score
  selects on uncertainty but the selected molecules are no less uncertain than
  greedy's. And the uncertainty-only policy, which selects purely on
  uncertainty, gets the highest mean uncertainty (0.497) and the lowest yield
  (47.8). High predicted uncertainty is not a reliable proxy for "worth
  testing": a molecule near the decision boundary has ~50% chance of being
  active by construction.
- **The uncertainty-only result is not a contradiction of Wen, Li, Xiang &
  Reker (Digital Discovery 2023, 2, 1134).** Their minimum-variance rule
  optimises *model performance* (a classifier trained on the subset generalises
  better); this task scores *discovery yield* (actives found per test). The
  optimal acquisition rule differs: boundary-proximal molecules are maximally
  informative for training, but their expected activity is ~0.5, which is the
  opposite of what a discovery campaign wants.
- Random is the most diverse (0.166 sim, 19.4 scaffolds) — no acquisition
  signal means no clustering. The diversity-augmented policies recover most of
  that breadth while keeping greedy's yield.

## 4. Final twenty-compound recommendation

After the 140-test budget is spent, the surrogate is refit on all revealed
labels and ranks the 235 never-queried molecules. The delivered panel is the
consensus across seeds 42/43/44, ranked on mean predicted activity, then
filtered to one representative per cluster at Tanimoto < 0.70.

**Redundancy in the unfiltered top 20:** mean pairwise Tanimoto 0.246,
**17 distinct Bemis-Murcko scaffolds across 20 slots** — the model was quietly
recommending near-duplicates and would have wasted 3 of 20 test slots.

**After Tanimoto < 0.70 filtering:** mean pairwise Tanimoto 0.200,
**20 distinct scaffolds across 20 slots**. Every slot is a distinct chemotype.

Top 5 of the consensus panel (all inside the applicability domain, nearest
labelled-neighbour Tanimoto 0.79-0.88, so none rests on extrapolation):

| rank | CID | mean P(active) | uncertainty | nearest labelled Tanimoto |
|---|---|---|---|---|
| 1 | BACE_892 | 0.914 | 0.280 | 0.884 |
| 2 | BACE_1274 | 0.728 | 0.445 | 0.788 |
| 3 | BACE_441 | 0.720 | 0.449 | 0.841 |
| 4 | BACE_85 | 0.718 | 0.450 | 0.804 |
| 5 | BACE_1172 | 0.710 | 0.454 | 0.800 |

The full panel is in `outputs/final_20_recommendation.csv`; the unfiltered
ranking in `outputs/final_20_unfiltered_ranking.csv`; per-seed panels in
`outputs/final_20_per_seed.csv`.

## 5. What this does NOT establish

- **This is a retrospective simulation on published activity data, not
  prospective discovery.** Every "discovered" active was already labelled in
  the source file (Subramanian et al., J. Chem. Inf. Model. 2016). The
  simulation tests whether the acquisition policy *would have* found them
  efficiently, not whether it finds anything new.
- **The final twenty are model predictions, not measured activities.** They are
  ranked on predicted P(active) from a random forest trained on the 140
  revealed labels. Their actual activity is unknown and unmeasured. Each is
  reported with its nearest-labelled-neighbour Tanimoto so a reader can see
  which rest on extrapolation (here, none do — all 20 are in-domain).
- **The pool partition is our own documented substitute, not the challenge
  partition.** The brief specifies scaffold-separated pools of 822 development /
  316 validation / 375 test supplied in `participant_starters/SM03`, which is
  not in the Agent Scientist repository, the Discovery Suite, or the event
  Discord. We generate a substitute at the exact specified sizes (whole
  Bemis-Murcko scaffold groups kept together, zero groups split across pools),
  and every ledger records the provenance string `LOCAL SCAFFOLD FALLBACK —
  dry-run only, not the challenge partition`. Results are therefore not
  directly comparable to a team using the official pools: the ranking and
  relative policy behaviour are expected to hold, the absolute numbers are not
  official.
- **The initial-40 seeding rule is ambiguous.** The brief says "lowest
  SHA256(CID+seed)" without pinning the concatenation. The plausible readings
  (plain, underscore-joined, colon-joined, seed-first) share under 10% of
  their selections. We use `plain` (CID followed by seed) as the default; all
  four are implemented behind a flag and the ambiguity is disclosed here.
- **Three test seeds cannot resolve differences smaller than ~5 actives.** The
  75 vs 70 (frozen vs greedy) gap on test is within the seed-to-seed spread;
  the policy was selected on validation, not chosen to win on test.

## Files

- `outputs/sm03_figure.png` — discovery curves (left) and yield/diversity
  trade-off (right)
- `outputs/test_summary.csv`, `test_curves.csv`, `test_diagnostics.csv` — test run
- `outputs/test_ledgers/*.json` — 9 append-only ledgers, predictions recorded
  before each reveal
- `outputs/tiebreak_summary.csv`, `tiebreak_diagnostics.csv` — 10-seed validation
- `outputs/final_20_recommendation.csv` — the delivered panel
- `outputs/frozen_policy.json` — the frozen policy and its selection record
