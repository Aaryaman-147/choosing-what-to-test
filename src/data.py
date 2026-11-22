"""
SM03 — data loading and pool resolution.

Rules this module enforces:
  * `Class`, `pIC50` and every source descriptor column are stripped out of
    anything a policy can see. Labels reach the policy ONLY through
    oracle.RevealOracle.
  * The source file's `Model` column is NOT the challenge partition and is
    discarded on load.
  * Pool membership comes from the organiser's starter files when present.
    The scaffold fallback exists so the harness can be dry-run before those
    files are in hand; it is never valid for a submitted result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Columns that must never reach a model or a policy.
BANNED_COLUMNS = {"Class", "pIC50", "Model"}

# The MoleculeNet BACE release names its structure column `mol`; some mirrors
# rename it to `smiles`. Accept either rather than making the caller rename.
SMILES_COL_CANDIDATES = ("mol", "smiles", "SMILES", "canonical_smiles")
ID_COL = "CID"
LABEL_COL = "Class"


def _resolve_smiles_column(columns) -> str:
    for candidate in SMILES_COL_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"no structure column found; expected one of {SMILES_COL_CANDIDATES}, got {list(columns)[:8]}"
    )


@dataclass(frozen=True)
class Pools:
    """Immutable pool membership: compound IDs per partition."""

    development: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    provenance: str

    def sizes(self) -> dict[str, int]:
        return {
            "development": len(self.development),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def assert_official_sizes(self) -> None:
        expected = {"development": 822, "validation": 316, "test": 375}
        actual = self.sizes()
        if actual != expected:
            raise ValueError(
                f"pool sizes {actual} != challenge spec {expected} "
                f"(provenance: {self.provenance})"
            )

    def assert_disjoint(self) -> None:
        d, v, t = set(self.development), set(self.validation), set(self.test)
        if d & v or d & t or v & t:
            raise ValueError("pools overlap; partitions must be disjoint")


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def bemis_murcko_scaffold(smi: str) -> str:
    """Bemis-Murcko scaffold SMILES. Acyclic compounds collapse to ''.

    Uses GetScaffoldForMol, whose signature has been stable across RDKit
    releases, and falls back to the MurckoScaffoldSmiles helper only if that
    fails. Failures raise: an earlier version swallowed them and returned '',
    which silently collapsed every molecule into a single scaffold group and
    produced a partition of 1513/0/0 instead of 822/316/375. A scaffold split
    that quietly stops being a scaffold split is worse than a crash.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smi!r}")
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception as exc:  # pragma: no cover - version-dependent path
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            raise RuntimeError(
                f"Murcko scaffold perception failed for {smi!r} on RDKit "
                f"{Chem.rdBase.rdkitVersion}: {exc}"
            ) from exc


def load_source(path: str | Path) -> pd.DataFrame:
    """Load the BACE source file and audit it.

    Returns a frame carrying ID, SMILES, canonical SMILES, scaffold and the
    label. The label column is present here because the oracle needs it; no
    other module is permitted to read it.
    """
    df = pd.read_csv(path)

    for required in (ID_COL, LABEL_COL):
        if required not in df.columns:
            raise ValueError(f"source file missing required column {required!r}")
    smiles_col = _resolve_smiles_column(df.columns)

    out = pd.DataFrame(
        {
            "cid": df[ID_COL].astype(str),
            "smiles": df[smiles_col].astype(str),
            LABEL_COL: df[LABEL_COL].astype(int),
        }
    )

    out["canonical_smiles"] = out["smiles"].map(canonical_smiles)
    n_unparsed = int(out["canonical_smiles"].isna().sum())
    if n_unparsed:
        raise ValueError(f"{n_unparsed} SMILES failed RDKit parsing; audit before proceeding")

    out["scaffold"] = out["canonical_smiles"].map(bemis_murcko_scaffold)

    # Degeneracy guard. A real drug-discovery set has many distinct scaffolds;
    # a handful means scaffold perception is broken on this RDKit build, and
    # every downstream "scaffold-separated" claim would be false.
    n_scaffolds = out["scaffold"].nunique()
    if n_scaffolds < max(10, 0.02 * len(out)):
        raise RuntimeError(
            f"only {n_scaffolds} distinct Bemis-Murcko scaffolds across {len(out)} "
            f"compounds — scaffold perception is not working on RDKit "
            f"{Chem.rdBase.rdkitVersion}. Refusing to build a scaffold split."
        )

    if out["cid"].duplicated().any():
        dupes = out.loc[out["cid"].duplicated(), "cid"].tolist()
        raise ValueError(f"duplicate CIDs in source: {dupes[:5]}")

    return out


def public_view(df: pd.DataFrame) -> pd.DataFrame:
    """The only frame a model or policy may consume: no labels, no pIC50."""
    keep = [c for c in df.columns if c not in BANNED_COLUMNS]
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Initial-40 selection: lowest SHA256(CID + seed)
# ---------------------------------------------------------------------------

# The brief says "lowest SHA256(CID+seed) hash" without pinning the byte
# layout of the concatenation. These are the plausible readings; `plain` is the
# default and the others let us match the organiser's runner exactly once the
# starter code or a Discord answer settles it.
HASH_VARIANTS = {
    "plain": lambda cid, seed: f"{cid}{seed}",
    "underscore": lambda cid, seed: f"{cid}_{seed}",
    "colon": lambda cid, seed: f"{cid}:{seed}",
    "seed_first": lambda cid, seed: f"{seed}{cid}",
}


def hash_key(cid: str, seed: int, variant: str = "plain") -> str:
    try:
        joiner = HASH_VARIANTS[variant]
    except KeyError:
        raise ValueError(f"unknown hash variant {variant!r}; choose from {sorted(HASH_VARIANTS)}")
    return hashlib.sha256(joiner(cid, seed).encode("utf-8")).hexdigest()


def initial_selection(
    cids: list[str] | tuple[str, ...],
    seed: int,
    n: int = 40,
    variant: str = "plain",
) -> list[str]:
    """The 40 starting molecules: lowest SHA256(CID+seed).

    Hex digests are fixed width, so lexicographic order on the digest string
    is the same as numeric order on the hash value.
    """
    if n > len(cids):
        raise ValueError(f"cannot take {n} from a pool of {len(cids)}")
    ranked = sorted(cids, key=lambda c: hash_key(c, seed, variant))
    return ranked[:n]


def initial_selection_agreement(cids, seed: int, n: int = 40) -> dict[str, float]:
    """How much the hash-variant ambiguity actually matters.

    Reports the overlap of each variant's initial-40 against `plain`, so we can
    say in the report whether the ambiguity is material or cosmetic.
    """
    base = set(initial_selection(cids, seed, n, "plain"))
    return {
        variant: len(base & set(initial_selection(cids, seed, n, variant))) / n
        for variant in HASH_VARIANTS
    }


# ---------------------------------------------------------------------------
# Pool resolution
# ---------------------------------------------------------------------------


def load_pools_from_starter(starter_dir: str | Path, all_cids: set[str]) -> Pools | None:
    """Look for the organiser's pool definitions in participant_starters/SM03.

    Accepts either per-pool CSVs or a single CSV/JSON carrying a pool column.
    Returns None if nothing recognisable is found, so the caller can fall back.
    """
    starter = Path(starter_dir)
    if not starter.is_dir():
        return None

    # Shape 1: separate files per pool.
    name_map = {
        "development": ("development", "dev", "train", "pool_dev"),
        "validation": ("validation", "valid", "val"),
        "test": ("test",),
    }
    found: dict[str, list[str]] = {}
    for pool, aliases in name_map.items():
        for path in sorted(starter.rglob("*.csv")):
            stem = path.stem.lower()
            if any(a in stem for a in aliases):
                frame = pd.read_csv(path)
                id_col = next(
                    (c for c in frame.columns if c.lower() in {"cid", "id", "compound_id"}),
                    None,
                )
                if id_col is None:
                    continue
                found[pool] = frame[id_col].astype(str).tolist()
                break

    if len(found) == 3:
        pools = Pools(
            development=tuple(found["development"]),
            validation=tuple(found["validation"]),
            test=tuple(found["test"]),
            provenance=f"organiser starter files under {starter}",
        )
        _validate_against_source(pools, all_cids)
        return pools

    # Shape 2: one file with a pool/split column.
    for path in sorted(starter.rglob("*.csv")):
        frame = pd.read_csv(path)
        id_col = next((c for c in frame.columns if c.lower() in {"cid", "id", "compound_id"}), None)
        pool_col = next(
            (c for c in frame.columns if c.lower() in {"pool", "split", "partition", "set"}),
            None,
        )
        if id_col is None or pool_col is None:
            continue
        buckets: dict[str, list[str]] = {}
        for pool, aliases in name_map.items():
            mask = frame[pool_col].astype(str).str.lower().isin(aliases)
            buckets[pool] = frame.loc[mask, id_col].astype(str).tolist()
        if all(buckets.values()):
            pools = Pools(
                development=tuple(buckets["development"]),
                validation=tuple(buckets["validation"]),
                test=tuple(buckets["test"]),
                provenance=f"organiser pool column {pool_col!r} in {path}",
            )
            _validate_against_source(pools, all_cids)
            return pools

    # Shape 3: curation summary JSON listing IDs.
    for path in sorted(starter.rglob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict):
            continue
        lowered = {k.lower(): v for k, v in blob.items()}
        picks = {}
        for pool, aliases in name_map.items():
            for a in aliases:
                if isinstance(lowered.get(a), list):
                    picks[pool] = [str(x) for x in lowered[a]]
                    break
        if len(picks) == 3:
            pools = Pools(
                development=tuple(picks["development"]),
                validation=tuple(picks["validation"]),
                test=tuple(picks["test"]),
                provenance=f"organiser JSON {path}",
            )
            _validate_against_source(pools, all_cids)
            return pools

    return None


def _validate_against_source(pools: Pools, all_cids: set[str]) -> None:
    pools.assert_disjoint()
    covered = set(pools.development) | set(pools.validation) | set(pools.test)
    unknown = covered - all_cids
    if unknown:
        raise ValueError(f"pool files reference {len(unknown)} CIDs absent from source, e.g. {sorted(unknown)[:5]}")
    missing = all_cids - covered
    if missing:
        raise ValueError(f"{len(missing)} source CIDs are in no pool, e.g. {sorted(missing)[:5]}")


def scaffold_fallback_pools(df: pd.DataFrame, seed: int = 0) -> Pools:
    """DRY-RUN ONLY scaffold-separated split at the challenge's 822/316/375 sizes.

    Whole scaffold groups are kept together and assigned largest-group-first to
    whichever pool is furthest below its quota. This reproduces the *shape* of
    the organiser's partition so the harness can be exercised end to end. It is
    NOT the challenge partition and any result produced on it is unsubmittable.
    """
    import random

    groups: dict[str, list[str]] = {}
    for cid, scaffold in zip(df["cid"], df["scaffold"]):
        groups.setdefault(scaffold, []).append(cid)

    rng = random.Random(seed)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    quotas = {"development": 822, "validation": 316, "test": 375}

    if sum(quotas.values()) != len(df):
        raise ValueError(f"quotas sum to {sum(quotas.values())} but source has {len(df)} compounds")

    largest = len(ordered[0][1])
    if largest > max(quotas.values()):
        raise RuntimeError(
            f"largest scaffold group holds {largest} compounds, more than the biggest "
            f"pool quota ({max(quotas.values())}). No scaffold-separated partition at "
            f"these sizes exists; check scaffold perception before continuing."
        )

    # Pass 1 — whole groups, largest first, each to the pool furthest below
    # quota. Whole-group packing cannot land on exact quotas by itself.
    assignment: dict[str, str] = {}  # scaffold -> pool
    sizes = {k: 0 for k in quotas}
    for scaffold, members in ordered:
        rng.shuffle(members)
        target = max(quotas, key=lambda k: quotas[k] - sizes[k])
        assignment[scaffold] = target
        sizes[target] += len(members)

    # Pass 2 — repair to exact quotas by relocating whole groups from an
    # over-quota pool to an under-quota one. Groups stay intact, so scaffold
    # separation is preserved; only which pool a group sits in changes.
    group_size = {s: len(m) for s, m in ordered}
    for _ in range(10000):
        over = [k for k in quotas if sizes[k] > quotas[k]]
        under = [k for k in quotas if sizes[k] < quotas[k]]
        if not over:
            break
        src = max(over, key=lambda k: sizes[k] - quotas[k])
        dst = max(under, key=lambda k: quotas[k] - sizes[k])
        surplus = sizes[src] - quotas[src]
        deficit = quotas[dst] - sizes[dst]
        movable = [s for s, pool in assignment.items() if pool == src]
        # Prefer the group that best closes the gap without overshooting either side.
        best = min(
            movable,
            key=lambda s: (group_size[s] > min(surplus, deficit), abs(group_size[s] - min(surplus, deficit))),
        )
        assignment[best] = dst
        sizes[src] -= group_size[best]
        sizes[dst] += group_size[best]
    else:
        raise RuntimeError("scaffold fallback repair did not converge")

    filled: dict[str, list[str]] = {k: [] for k in quotas}
    for scaffold, members in ordered:
        filled[assignment[scaffold]].extend(members)

    actual = {k: len(v) for k, v in filled.items()}
    if actual != quotas:
        raise RuntimeError(
            f"scaffold fallback produced {actual}, expected {quotas}. "
            f"{len(ordered)} scaffold groups were available."
        )

    return Pools(
        development=tuple(filled["development"]),
        validation=tuple(filled["validation"]),
        test=tuple(filled["test"]),
        provenance="LOCAL SCAFFOLD FALLBACK — dry-run only, not the challenge partition",
    )


def resolve_pools(df: pd.DataFrame, starter_dir: str | Path | None) -> Pools:
    all_cids = set(df["cid"])
    if starter_dir is not None:
        pools = load_pools_from_starter(starter_dir, all_cids)
        if pools is not None:
            return pools
    return scaffold_fallback_pools(df)
