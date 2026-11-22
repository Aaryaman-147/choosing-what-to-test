"""
SM03 — featurisation.

Descriptors are computed from the SMILES strings only. The source file ships
~590 precomputed descriptor columns alongside `pIC50` and `Class`; we do not
touch them. Recomputing from structure keeps the feature path auditable and
removes any question about whether a source column encodes the label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048

# Physchem block. Deliberately small and interpretable: these are the axes we
# report diversity and applicability domain against.
DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "RotB": Lipinski.NumRotatableBonds,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "Rings": rdMolDescriptors.CalcNumRings,
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "MolMR": Crippen.MolMR,
    "HeteroAtoms": Lipinski.NumHeteroatoms,
    "StereoCentres": rdMolDescriptors.CalcNumAtomStereoCenters,
    "AmideBonds": rdMolDescriptors.CalcNumAmideBonds,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
}

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


@dataclass
class FeatureSet:
    """Aligned feature matrices plus the fingerprint objects for similarity."""

    cids: list[str]
    fingerprint_matrix: np.ndarray  # (n, FP_BITS) uint8
    descriptor_matrix: np.ndarray  # (n, len(DESCRIPTORS)) float64
    fingerprints: list  # RDKit ExplicitBitVect, for Tanimoto
    descriptor_names: list[str]

    @property
    def X(self) -> np.ndarray:
        """Model input: fingerprint bits concatenated with physchem block."""
        return np.hstack([self.fingerprint_matrix, self.descriptor_matrix])

    def index_of(self, cid: str) -> int:
        return self._lookup[cid]

    def __post_init__(self) -> None:
        self._lookup = {c: i for i, c in enumerate(self.cids)}

    def subset(self, cids: list[str]) -> "FeatureSet":
        idx = [self.index_of(c) for c in cids]
        return FeatureSet(
            cids=list(cids),
            fingerprint_matrix=self.fingerprint_matrix[idx],
            descriptor_matrix=self.descriptor_matrix[idx],
            fingerprints=[self.fingerprints[i] for i in idx],
            descriptor_names=list(self.descriptor_names),
        )


def _describe(mol) -> list[float]:
    values = []
    for fn in DESCRIPTORS.values():
        try:
            values.append(float(fn(mol)))
        except Exception:
            values.append(np.nan)
    return values


def featurise(cids: list[str], smiles: list[str]) -> FeatureSet:
    if len(cids) != len(smiles):
        raise ValueError("cids and smiles must align")

    fps, fp_rows, desc_rows = [], [], []
    for cid, smi in zip(cids, smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"{cid}: SMILES failed to parse — audit the source file")
        fp = _GEN.GetFingerprint(mol)
        arr = np.zeros((FP_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(fp)
        fp_rows.append(arr)
        desc_rows.append(_describe(mol))

    descriptor_matrix = np.asarray(desc_rows, dtype=np.float64)
    # Median-impute the rare descriptor failure rather than dropping a molecule.
    col_median = np.nanmedian(descriptor_matrix, axis=0)
    nan_mask = np.isnan(descriptor_matrix)
    descriptor_matrix[nan_mask] = np.take(col_median, np.where(nan_mask)[1])

    return FeatureSet(
        cids=list(cids),
        fingerprint_matrix=np.asarray(fp_rows, dtype=np.uint8),
        descriptor_matrix=descriptor_matrix,
        fingerprints=fps,
        descriptor_names=list(DESCRIPTORS),
    )


# ---------------------------------------------------------------------------
# Similarity / diversity
# ---------------------------------------------------------------------------


def tanimoto_to_set(query_fp, reference_fps: list) -> np.ndarray:
    if not reference_fps:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps), dtype=np.float64)


def max_similarity_to_set(query_fps: list, reference_fps: list) -> np.ndarray:
    """For each query, its Tanimoto similarity to the nearest reference.

    Doubles as the applicability-domain measure: a candidate whose nearest
    labelled neighbour is distant is one the model is extrapolating to.
    """
    if not reference_fps:
        return np.zeros(len(query_fps), dtype=np.float64)
    return np.asarray(
        [float(tanimoto_to_set(fp, reference_fps).max()) for fp in query_fps],
        dtype=np.float64,
    )


def mean_pairwise_similarity(fps: list) -> float:
    """Internal similarity of a set. Lower means a more diverse batch."""
    if len(fps) < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        total += float(sum(sims))
        count += len(sims)
    return total / count


def scaffold_count(smiles: list[str]) -> int:
    from .data import bemis_murcko_scaffold

    return len({bemis_murcko_scaffold(s) for s in smiles})
