from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.feature_selection import VarianceThreshold, f_classif, f_regression

def extract_morgan_fingerprints(mols: Sequence[Any], radius: int = 2, n_bits: int = 1024, use_counts: bool = False,) -> tuple[np.ndarray, list[str]]:
    n_mols = len(mols)
    feature_matrix = np.zeros((n_mols, n_bits), dtype=np.float32)
    feature_names = [f"ecfp_{radius}_{n_bits}.bit_{b}" for b in range(n_bits)]
    for idx, item in enumerate(mols):
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            continue
        if use_counts:
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
            for bit, val in fp.GetNonzeroElements().items():
                feature_matrix[idx, bit] = float(val)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

            on_bits = list(fp.GetOnBits())
            if on_bits:
                feature_matrix[idx, on_bits] = 1.0
    return feature_matrix, feature_names

def extract_rdkit_2d_descriptors(mols: Sequence[Any],) -> tuple[np.ndarray, list[str]]:
    desc_names = [name for name, _ in Descriptors.descList]
    desc_fns = [fn for _, fn in Descriptors.descList]
    n_descs = len(desc_names)
    n_mols = len(mols)
    matrix = np.zeros((n_mols, n_descs), dtype=np.float32)
    for i, item in enumerate(mols):
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            continue
        for j, fn in enumerate(desc_fns):
            try:
                val = float(fn(mol))
                if not math.isfinite(val) or abs(val) > 1e10:
                    val = 0.0
                matrix[i, j] = np.float32(val)
            except Exception:
                matrix[i, j] = 0.0

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    return matrix, [f"rdkit_2d.{name}" for name in desc_names]

def select_top_features(feature_matrix: np.ndarray, feature_names: list[str], targets: np.ndarray, task_kind: str = "scalar_regression", top_k: int = 64, max_features: int | None = None,) -> tuple[np.ndarray, list[str], np.ndarray]:
    if max_features is not None:
        top_k = max_features
    n_samples, n_features = feature_matrix.shape
    if n_features <= top_k:
        return feature_matrix, feature_names, np.arange(n_features)

    selector_var = VarianceThreshold(threshold=1e-4)
    try:
        reduced_matrix = selector_var.fit_transform(feature_matrix)
        retained_indices = selector_var.get_support(indices=True)
    except ValueError:

        return feature_matrix[:, :top_k], feature_names[:top_k], np.arange(top_k)
    if reduced_matrix.shape[1] <= top_k:
        selected_names = [feature_names[i] for i in retained_indices]
        return reduced_matrix, selected_names, retained_indices

    finite_mask = np.isfinite(reduced_matrix)
    if not finite_mask.all():
        for col_idx in range(reduced_matrix.shape[1]):
            col = reduced_matrix[:, col_idx]
            valid_col = col[np.isfinite(col)]
            med = np.median(valid_col) if len(valid_col) > 0 else 0.0
            col[~np.isfinite(col)] = med
            reduced_matrix[:, col_idx] = col

    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    finite_y = np.isfinite(y)
    if not finite_y.all():

        reduced_matrix = reduced_matrix[finite_y]
        y = y[finite_y]

    is_classification = "classification" in task_kind
    try:
        if is_classification:

            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                scores = np.ones(reduced_matrix.shape[1])
            else:
                scores, _ = f_classif(reduced_matrix, y)
        else:
            scores, _ = f_regression(reduced_matrix, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:

        scores = np.abs([
            np.corrcoef(reduced_matrix[:, c], y)[0, 1] if np.std(reduced_matrix[:, c]) > 1e-6 else 0.0
            for c in range(reduced_matrix.shape[1])
        ])
        scores = np.nan_to_num(scores, nan=0.0)

    top_indices_local = np.argsort(scores)[::-1][:top_k]
    final_indices = retained_indices[top_indices_local]
    selected_matrix = feature_matrix[:, final_indices]
    selected_names = [feature_names[i] for i in final_indices]
    return selected_matrix, selected_names, final_indices

class AutoPharmacophoreBlock:
    def __init__(self, top_k: int = 64, radius: int = 2, n_bits: int = 1024, use_morgan: bool = True, use_rdkit_2d: bool = True,) -> None:
        self.top_k = top_k
        self.radius = radius
        self.n_bits = n_bits
        self.use_morgan = use_morgan
        self.use_rdkit_2d = use_rdkit_2d
        self.selected_indices: np.ndarray | None = None
        self.selected_names: list[str] = []

    @property
    def selected_feature_names(self) -> list[str]:
        return self.selected_names

    def fit_transform(self, mols: Sequence[Any], arg2: Any, arg3: Any = None, task_kind: str = "scalar_regression",) -> tuple[np.ndarray, list[str]]:
        if arg3 is None or isinstance(arg3, str):
            targets = np.asarray(arg2)
            train_idx = np.arange(len(mols))
            if isinstance(arg3, str):
                task_kind = arg3
        else:
            train_idx = np.asarray(arg2, dtype=int)
            targets = np.asarray(arg3)
        matrices = []
        names = []
        if self.use_morgan:
            fp_mat, fp_names = extract_morgan_fingerprints(mols, radius=self.radius, n_bits=self.n_bits)
            matrices.append(fp_mat)
            names.extend(fp_names)
        if self.use_rdkit_2d:
            rdkit_mat, rdkit_names = extract_rdkit_2d_descriptors(mols)
            matrices.append(rdkit_mat)
            names.extend(rdkit_names)
        if not matrices:
            raise ValueError("At least one of use_morgan or use_rdkit_2d must be True")
        combined_mat = np.hstack(matrices)
        all_names = names
        train_feats = combined_mat[train_idx]
        train_y = targets[train_idx]
        _, selected_names, selected_indices = select_top_features(
            train_feats, all_names, train_y, task_kind=task_kind, top_k=self.top_k
        )
        self.selected_indices = selected_indices
        self.selected_names = selected_names
        final_mat = combined_mat[:, self.selected_indices]
        return final_mat, self.selected_names

    def transform(self, mols: Sequence[Any]) -> np.ndarray:
        if self.selected_indices is None:
            raise RuntimeError("AutoPharmacophoreBlock must be fit before transform")
        matrices = []
        if self.use_morgan:
            fp_mat, _ = extract_morgan_fingerprints(mols, radius=self.radius, n_bits=self.n_bits)
            matrices.append(fp_mat)
        if self.use_rdkit_2d:
            rdkit_mat, _ = extract_rdkit_2d_descriptors(mols)
            matrices.append(rdkit_mat)
        combined_mat = np.hstack(matrices)
        return combined_mat[:, self.selected_indices]
