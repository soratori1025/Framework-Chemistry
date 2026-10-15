from __future__ import annotations
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from ..config import ExperimentConfig, load_config
from ..data import load_configured_dataset

@dataclass
class PropertySummary:
    name: str
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": round(self.min, 2),
            "q25": round(self.q25, 2),
            "median": round(self.median, 2),
            "q75": round(self.q75, 2),
            "max": round(self.max, 2),
        }

@dataclass
class ColumnFeatureSummary:
    name: str
    missing_count: int
    missing_fraction: float
    mean: float
    std: float
    min: float
    max: float
    variance: float
    is_constant: bool
    is_near_zero_variance: bool
    skewness: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_count": self.missing_count,
            "missing_fraction": round(self.missing_fraction, 4),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "variance": round(self.variance, 6),
            "is_constant": self.is_constant,
            "is_near_zero_variance": self.is_near_zero_variance,
            "skewness": round(self.skewness, 4),
        }

@dataclass
class TabularFeatureProfile:
    feature_count: int
    column_stats: dict[str, ColumnFeatureSummary] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)
    near_zero_variance_columns: list[str] = field(default_factory=list)
    high_missing_columns: list[str] = field(default_factory=list)
    collinear_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    recommended_keep_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_count": self.feature_count,
            "column_stats": {k: v.to_dict() for k, v in self.column_stats.items()},
            "constant_columns": self.constant_columns,
            "near_zero_variance_columns": self.near_zero_variance_columns,
            "high_missing_columns": self.high_missing_columns,
            "collinear_pairs": [
                {"feat1": f1, "feat2": f2, "correlation": round(r, 4)}
                for f1, f2, r in self.collinear_pairs
            ],
            "recommended_keep_features": self.recommended_keep_features,
        }

@dataclass
class ChemicalProfile:
    total_records: int
    valid_smiles_count: int
    invalid_smiles_count: int
    invalid_fraction: float
    unique_smiles_count: int
    duplicate_count: int
    unique_scaffolds_count: int
    scaffold_diversity_ratio: float
    mean_tanimoto_similarity: float = 0.0
    tanimoto_diversity_score: float = 1.0
    property_stats: dict[str, PropertySummary] = field(default_factory=dict)
    invalid_examples: list[tuple[int, str]] = field(default_factory=list)

@dataclass
class TargetProfile:
    name: str
    task_kind: str
    total_count: int
    valid_count: int
    missing_count: int
    missing_fraction: float

    class_counts: dict[str, int] = field(default_factory=dict)
    class_imbalance_ratio: float = 1.0

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    outliers_count: int = 0
    outliers_fraction: float = 0.0

@dataclass
class RiskItem:
    severity: str
    category: str
    title: str
    message: str
    recommendation: str

@dataclass
class AdvisorRecommendation:
    health_score: int
    health_status: str
    risks: list[RiskItem] = field(default_factory=list)
    recommended_split: str = "scaffold"
    recommended_encoder: str = "dmpnn"
    recommended_model_type: str = "hybrid"
    recommended_head: str = "standard"
    recommended_loss: str = "huber"
    recommended_metrics: list[str] = field(default_factory=list)
    recommended_batch_size: int = 64
    recommended_epochs: int = 40
    recommended_lr: float = 0.001
    recommended_scheduler: str = "cosine"
    recommended_warmup_epochs: int = 3
    recommended_ensemble_size: int = 1
    recommended_pos_weight: float | None = None
    target_scaling: bool = False
    feature_imputation: str | None = None
    feature_scaling: str | None = None
    config_dict: dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetReport:
    dataset_name: str
    chemical_profile: ChemicalProfile
    target_profiles: dict[str, TargetProfile]
    recommendation: AdvisorRecommendation
    tabular_profile: TabularFeatureProfile | None = None
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "health_score": self.recommendation.health_score,
            "health_status": self.recommendation.health_status,
            "chemical_profile": {
                "total_records": self.chemical_profile.total_records,
                "valid_count": self.chemical_profile.valid_smiles_count,
                "invalid_count": self.chemical_profile.invalid_smiles_count,
                "invalid_fraction": round(self.chemical_profile.invalid_fraction, 4),
                "unique_smiles": self.chemical_profile.unique_smiles_count,
                "unique_scaffolds": self.chemical_profile.unique_scaffolds_count,
                "scaffold_diversity": round(self.chemical_profile.scaffold_diversity_ratio, 4),
                "tanimoto_similarity": round(self.chemical_profile.mean_tanimoto_similarity, 4),
                "tanimoto_diversity": round(self.chemical_profile.tanimoto_diversity_score, 4),
                "properties": {k: v.to_dict() for k, v in self.chemical_profile.property_stats.items()},
            },
            "targets": {
                name: {
                    "task_kind": t.task_kind,
                    "valid_count": t.valid_count,
                    "missing_count": t.missing_count,
                    "class_counts": t.class_counts,
                    "imbalance_ratio": round(t.class_imbalance_ratio, 2) if t.class_imbalance_ratio else None,
                    "mean": round(t.mean, 4) if t.mean is not None else None,
                    "std": round(t.std, 4) if t.std is not None else None,
                    "skewness": round(t.skewness, 2) if t.skewness is not None else None,
                    "outliers_fraction": round(t.outliers_fraction, 4),
                }
                for name, t in self.target_profiles.items()
            },
            "risks": [asdict(r) for r in self.recommendation.risks],
            "prescriptions": {
                "split": self.recommendation.recommended_split,
                "encoder": self.recommendation.recommended_encoder,
                "model_type": self.recommendation.recommended_model_type,
                "head": self.recommendation.recommended_head,
                "loss": self.recommendation.recommended_loss,
                "metrics": self.recommendation.recommended_metrics,
                "scheduler": self.recommendation.recommended_scheduler,
                "ensemble_size": self.recommendation.recommended_ensemble_size,
                "feature_imputation": self.recommendation.feature_imputation,
                "feature_scaling": self.recommendation.feature_scaling,
            },
        }
        if self.tabular_profile is not None:
            data["tabular_features"] = self.tabular_profile.to_dict()
        return data

    def save_json(self, output_path: str | Path) -> Path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    def save_markdown(self, output_path: str | Path) -> Path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        advisor = DatasetAdvisor()
        content = advisor.format_markdown_report(self)
        with out.open("w", encoding="utf-8") as f:
            f.write(content)
        return out

    def save_config(self, output_path: str | Path) -> Path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(self.recommendation.config_dict, f, sort_keys=False)
        return out

def _calc_stats(arr: np.ndarray, name: str) -> PropertySummary:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return PropertySummary(name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q25, median, q75 = np.percentile(valid, [25, 50, 75])
    return PropertySummary(
        name=name,
        mean=float(valid.mean()),
        std=float(valid.std()),
        min=float(valid.min()),
        q25=float(q25),
        median=float(median),
        q75=float(q75),
        max=float(valid.max()),
    )

class DatasetAdvisor:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def profile_dataset(
        self,
        smiles_list: Sequence[str],
        targets: Mapping[str, Sequence[Any]],
        tabular_features: Mapping[str, Sequence[Any]] | None = None,
        dataset_name: str = "dataset",
        source_path: str | None = None,
        smiles_column: str = "smiles",) -> DatasetReport:
        total = len(smiles_list)
        if total == 0:
            raise ValueError("smiles_list cannot be empty")
        valid_mols: list[Any] = []
        valid_indices: list[int] = []
        invalid_examples: list[tuple[int, str]] = []
        mw_list, logp_list, ha_list, hbd_list, hba_list, tpsa_list, rotb_list = (
            [], [], [], [], [], [], []
        )
        scaffolds_set = set()
        for idx, s in enumerate(smiles_list):
            if not isinstance(s, str) or not s.strip():
                invalid_examples.append((idx + 1, str(s)))
                continue
            mol = Chem.MolFromSmiles(s.strip())
            if mol is None:
                invalid_examples.append((idx + 1, s.strip()))
                continue
            valid_mols.append(mol)
            valid_indices.append(idx)
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
            ha_list.append(mol.GetNumHeavyAtoms())
            hbd_list.append(Lipinski.NumHDonors(mol))
            hba_list.append(Lipinski.NumHAcceptors(mol))
            tpsa_list.append(Descriptors.TPSA(mol))
            rotb_list.append(Descriptors.NumRotatableBonds(mol))
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaff:
                scaffolds_set.add(scaff)
        valid_count = len(valid_mols)
        invalid_count = total - valid_count
        invalid_frac = invalid_count / total
        unique_smiles_count = len(set(smiles_list))
        duplicate_count = total - unique_smiles_count
        scaffold_ratio = len(scaffolds_set) / max(1, valid_count)

        tanimoto_sim = 0.0
        tanimoto_diversity = 1.0
        if len(valid_mols) > 1:
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
            sample_mols = valid_mols[:min(250, len(valid_mols))]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in sample_mols]
            sims = []
            for i in range(len(fps)):
                if i + 1 < len(fps):
                    sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
            if sims:
                tanimoto_sim = float(np.mean(sims))
                tanimoto_diversity = float(max(0.0, 1.0 - tanimoto_sim))

        prop_stats = {
            "Molecular Weight (MW)": _calc_stats(np.array(mw_list), "MW"),
            "LogP": _calc_stats(np.array(logp_list), "LogP"),
            "Heavy Atoms": _calc_stats(np.array(ha_list), "HeavyAtoms"),
            "H-Bond Donors": _calc_stats(np.array(hbd_list), "HBD"),
            "H-Bond Acceptors": _calc_stats(np.array(hba_list), "HBA"),
            "TPSA (Å²)": _calc_stats(np.array(tpsa_list), "TPSA"),
            "Rotatable Bonds": _calc_stats(np.array(rotb_list), "RotB"),
        }
        chem_profile = ChemicalProfile(
            total_records=total,
            valid_smiles_count=valid_count,
            invalid_smiles_count=invalid_count,
            invalid_fraction=invalid_frac,
            unique_smiles_count=unique_smiles_count,
            duplicate_count=duplicate_count,
            unique_scaffolds_count=len(scaffolds_set),
            scaffold_diversity_ratio=scaffold_ratio,
            mean_tanimoto_similarity=tanimoto_sim,
            tanimoto_diversity_score=tanimoto_diversity,
            property_stats=prop_stats,
            invalid_examples=invalid_examples[:10],
        )

        target_profiles: dict[str, TargetProfile] = {}
        for t_name, t_values in targets.items():
            target_profiles[t_name] = self._profile_target(t_name, t_values)

        tab_profile: TabularFeatureProfile | None = None
        if tabular_features:
            tab_profile = self._profile_tabular_features(tabular_features)

        recommendation = self._generate_recommendations(
            chem_profile,
            target_profiles,
            tab_profile,
            dataset_name,
            source_path,
            smiles_column=smiles_column,
        )
        return DatasetReport(
            dataset_name=dataset_name,
            chemical_profile=chem_profile,
            target_profiles=target_profiles,
            tabular_profile=tab_profile,
            recommendation=recommendation,
            source_path=source_path,
        )

    def profile_csv(self, csv_path: str | Path, smiles_column: str | None = None, target_columns: Sequence[str] | str | None = None, feature_columns: Sequence[str] | None = None, dataset_name: str | None = None,) -> DatasetReport:
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset file not found: {path}")
        sep = "\t" if path.suffix in {".tab", ".tsv"} else ","
        df = pd.read_csv(path, sep=sep)
        if df.empty:
            raise ValueError(f"dataset file {path} is empty")
        if dataset_name is None:
            dataset_name = path.stem

        if smiles_column is None:
            candidates = ["smiles", "canonical_smiles", "drug", "molecule", "structure", "compound", "mol"]
            found = None
            for c in df.columns:
                if c.lower() in candidates:
                    found = c
                    break
            if found is None:
                for c in df.columns:
                    if df[c].dtype == object:
                        sample = df[c].dropna().head(5)
                        if len(sample) > 0 and all(Chem.MolFromSmiles(str(x)) is not None for x in sample):
                            found = c
                            break
            if found is None:
                raise ValueError(
                    f"could not automatically detect SMILES column in {list(df.columns)}. "
                    "Please specify smiles_column explicitly."
                )
            smiles_column = found
        smiles_list = df[smiles_column].astype(str).tolist()

        if target_columns is None:
            common_targets = ["target", "label", "activity", "pic50", "ic50", "value", "y", "class", "target_value"]
            found_targets = []
            for c in df.columns:
                if c != smiles_column and c.lower() in common_targets:
                    found_targets.append(c)
            if not found_targets:
                for c in df.columns:
                    if c != smiles_column and pd.api.types.is_numeric_dtype(df[c]) and not c.lower().endswith("id"):
                        found_targets.append(c)
            if not found_targets:
                raise ValueError(f"no numerical target columns detected in {path}")
            target_cols = [found_targets[-1]]
        elif isinstance(target_columns, str):
            target_cols = [target_columns]
        else:
            target_cols = list(target_columns)
        targets = {col: df[col].tolist() for col in target_cols if col in df.columns}
        if not targets:
            raise ValueError(f"specified target columns {target_cols} not found in {list(df.columns)}")

        tabular_features: dict[str, list[Any]] = {}
        if feature_columns is not None:
            for col in feature_columns:
                if col in df.columns:
                    tabular_features[col] = df[col].tolist()
        else:
            excluded = set(target_cols) | {smiles_column}
            for col in df.columns:
                if col not in excluded and pd.api.types.is_numeric_dtype(df[col]) and not col.lower().endswith("id"):
                    tabular_features[col] = df[col].tolist()
        return self.profile_dataset(
            smiles_list=smiles_list,
            targets=targets,
            tabular_features=tabular_features if tabular_features else None,
            dataset_name=dataset_name,
            source_path=str(path),
            smiles_column=smiles_column,
        )

    def profile_config(self, config_path_or_obj: str | Path | Any) -> DatasetReport:
        from ..config import ExperimentConfig, load_config
        from ..data import load_configured_dataset

        if isinstance(config_path_or_obj, ExperimentConfig):
            config = config_path_or_obj
        else:
            config = load_config(config_path_or_obj)
        if config.dataset.provider == "csv" and config.dataset.path is not None:
            csv_path = config.resolve_path(config.dataset.path)
            smiles_col = config.dataset.smiles_column
            target_cols = [t.column for t in config.dataset.targets.values()]
            return self.profile_csv(
                csv_path=csv_path,
                smiles_column=smiles_col,
                target_columns=target_cols,
                dataset_name=config.name,
            )
        ds = load_configured_dataset(config)
        return self.profile_dataset(
            smiles_list=ds.smiles,
            targets=ds.targets,
            dataset_name=config.name,
            source_path=f"provider:{config.dataset.provider}/{config.dataset.name or ''}",
        )

    def _profile_target(self, name: str, values: Sequence[Any]) -> TargetProfile:
        total = len(values)
        arr = np.asarray(values, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        valid_vals = arr[finite_mask]
        valid_count = int(valid_vals.size)
        missing_count = total - valid_count
        missing_frac = missing_count / total if total > 0 else 0.0
        if valid_count == 0:
            return TargetProfile(
                name=name,
                task_kind="scalar_regression",
                total_count=total,
                valid_count=0,
                missing_count=total,
                missing_fraction=1.0,
            )
        unique_vals = np.unique(valid_vals)

        if set(unique_vals).issubset({0.0, 1.0}):
            pos_count = int((valid_vals == 1.0).sum())
            neg_count = int((valid_vals == 0.0).sum())
            min_c = max(1, min(pos_count, neg_count))
            max_c = max(pos_count, neg_count)
            imbalance_ratio = max_c / min_c
            return TargetProfile(
                name=name,
                task_kind="binary_classification",
                total_count=total,
                valid_count=valid_count,
                missing_count=missing_count,
                missing_fraction=missing_frac,
                class_counts={"0": neg_count, "1": pos_count},
                class_imbalance_ratio=float(imbalance_ratio),
            )

        mean = float(valid_vals.mean())
        std = float(valid_vals.std())
        v_min = float(valid_vals.min())
        v_max = float(valid_vals.max())
        median = float(np.median(valid_vals))
        skewness = float(stats.skew(valid_vals)) if valid_count > 2 else 0.0
        kurt = float(stats.kurtosis(valid_vals)) if valid_count > 3 else 0.0

        q25, q75 = np.percentile(valid_vals, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = (valid_vals < lower_bound) | (valid_vals > upper_bound)
        outliers_count = int(outliers.sum())
        outliers_frac = outliers_count / valid_count
        return TargetProfile(
            name=name,
            task_kind="scalar_regression",
            total_count=total,
            valid_count=valid_count,
            missing_count=missing_count,
            missing_fraction=missing_frac,
            mean=mean,
            std=std,
            min=v_min,
            max=v_max,
            median=median,
            skewness=skewness,
            kurtosis=kurt,
            outliers_count=outliers_count,
            outliers_fraction=outliers_frac,
        )

    def _profile_tabular_features(self, features: Mapping[str, Sequence[Any]],) -> TabularFeatureProfile:
        stats_dict: dict[str, ColumnFeatureSummary] = {}
        constant_cols: list[str] = []
        near_zero_cols: list[str] = []
        high_missing_cols: list[str] = []
        valid_feature_names: list[str] = []
        matrix_cols: list[np.ndarray] = []
        for name, vals in features.items():
            arr = np.asarray(vals, dtype=np.float64)
            n_total = len(arr)
            finite_mask = np.isfinite(arr)
            n_valid = int(finite_mask.sum())
            n_missing = n_total - n_valid
            missing_frac = n_missing / n_total if n_total > 0 else 0.0
            if n_valid == 0:
                summary = ColumnFeatureSummary(
                    name=name,
                    missing_count=n_missing,
                    missing_fraction=missing_frac,
                    mean=0.0,
                    std=0.0,
                    min=0.0,
                    max=0.0,
                    variance=0.0,
                    is_constant=True,
                    is_near_zero_variance=True,
                    skewness=0.0,
                )
                stats_dict[name] = summary
                constant_cols.append(name)
                high_missing_cols.append(name)
                continue
            valid_vals = arr[finite_mask]
            mean_v = float(valid_vals.mean())
            std_v = float(valid_vals.std())
            var_v = float(valid_vals.var())
            min_v = float(valid_vals.min())
            max_v = float(valid_vals.max())
            is_const = bool(var_v == 0.0 or len(np.unique(valid_vals)) <= 1)
            is_nzv = bool(std_v < 1e-4)
            skew_v = 0.0 if (is_const or is_nzv or n_valid <= 2) else float(stats.skew(valid_vals))
            if is_const:
                constant_cols.append(name)
            elif is_nzv:
                near_zero_cols.append(name)

            if missing_frac > 0.20:
                high_missing_cols.append(name)
            summary = ColumnFeatureSummary(
                name=name,
                missing_count=n_missing,
                missing_fraction=missing_frac,
                mean=mean_v,
                std=std_v,
                min=min_v,
                max=max_v,
                variance=var_v,
                is_constant=is_const,
                is_near_zero_variance=is_nzv,
                skewness=skew_v,
            )
            stats_dict[name] = summary

            if not is_const and missing_frac < 0.50:
                valid_feature_names.append(name)
                imputed = np.copy(arr)
                imputed[~finite_mask] = mean_v
                matrix_cols.append(imputed)

        collinear_pairs: list[tuple[str, str, float]] = []
        if len(matrix_cols) > 1:
            data_mat = np.column_stack(matrix_cols)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_mat = np.corrcoef(data_mat, rowvar=False)
            num_cols = len(valid_feature_names)
            for i in range(num_cols):
                for j in range(i + 1, num_cols):
                    r_val = corr_mat[i, j]
                    if np.isfinite(r_val) and abs(r_val) > 0.90:
                        collinear_pairs.append((valid_feature_names[i], valid_feature_names[j], float(r_val)))

        dropped_collinear = set()
        for f1, f2, _ in collinear_pairs:
            f1_missing = stats_dict[f1].missing_fraction
            f2_missing = stats_dict[f2].missing_fraction
            if f2_missing >= f1_missing:
                dropped_collinear.add(f2)
            else:
                dropped_collinear.add(f1)

        recommended_keep = [
            f for f in valid_feature_names
            if f not in dropped_collinear
            and stats_dict[f].missing_fraction < 0.50
            and not stats_dict[f].is_near_zero_variance
        ]

        return TabularFeatureProfile(
            feature_count=len(features),
            column_stats=stats_dict,
            constant_columns=constant_cols,
            near_zero_variance_columns=near_zero_cols,
            high_missing_columns=high_missing_cols,
            collinear_pairs=collinear_pairs,
            recommended_keep_features=recommended_keep,
        )

    def _generate_recommendations(
        self,
        chem: ChemicalProfile,
        targets: dict[str, TargetProfile],
        tab: TabularFeatureProfile | None,
        dataset_name: str,
        source_path: str | None,
        smiles_column: str = "smiles",
    ) -> AdvisorRecommendation:
        score = 100
        risks: list[RiskItem] = []

        if chem.invalid_fraction > 0.05:
            score -= 25
            risks.append(
                RiskItem(
                    severity="HIGH",
                    category="data_quality",
                    title="High Invalid SMILES Ratio",
                    message=f"{chem.invalid_smiles_count} of {chem.total_records} molecules ({chem.invalid_fraction:.1%}) failed RDKit parsing.",
                    recommendation="Filter or canonicalize invalid SMILES before training to avoid data leakage or dropped batches.",
                )
            )
        elif chem.invalid_fraction > 0.01:
            score -= 10
            risks.append(
                RiskItem(
                    severity="MEDIUM",
                    category="data_quality",
                    title="Minor Invalid SMILES Found",
                    message=f"{chem.invalid_smiles_count} molecules failed parsing.",
                    recommendation="Ensure input data is sanitized with standard RDKit Chem.MolFromSmiles.",
                )
            )

        if chem.duplicate_count > (0.05 * chem.total_records):
            score -= 10
            risks.append(
                RiskItem(
                    severity="MEDIUM",
                    category="data_quality",
                    title="Duplicate Molecules Detected",
                    message=f"{chem.duplicate_count} duplicate SMILES found ({chem.duplicate_count / chem.total_records:.1%}).",
                    recommendation="Deduplicate compounds by averaging replicate target values to prevent train/test contamination.",
                )
            )

        if chem.total_records < 300:
            score -= 15
            risks.append(
                RiskItem(
                    severity="HIGH",
                    category="data_quality",
                    title="Small Sample Size (N < 300)",
                    message=f"Dataset has only {chem.total_records} records, making deep neural nets prone to overfitting.",
                    recommendation="Use Hybrid features (D-MPNN + Handcrafted descriptors) with higher dropout (0.2) or Deep Ensemble (K=3) to regularize.",
                )
            )
        elif chem.total_records < 1000:
            score -= 5
            risks.append(
                RiskItem(
                    severity="LOW",
                    category="data_quality",
                    title="Moderate Sample Size (300 <= N < 1000)",
                    message=f"Sample size of {chem.total_records} compounds is well-suited for D-MPNN or GIN with warm-up learning rate schedule.",
                    recommendation="Train with Cosine annealing and early stopping patience=15.",
                )
            )

        feature_imputation = None
        feature_scaling = None
        has_tabular = tab is not None and tab.feature_count > 0

        if tab is not None and tab.feature_count > 0:
            feature_scaling = "standard"
            any_missing = any(cs.missing_count > 0 for cs in tab.column_stats.values())
            if any_missing:
                feature_imputation = "median"

            if tab.constant_columns:
                score -= 10
                risks.append(
                    RiskItem(
                        severity="MEDIUM",
                        category="features",
                        title=f"Constant Features Detected ({len(tab.constant_columns)})",
                        message=f"Columns {tab.constant_columns} have zero variance across all compounds.",
                        recommendation="Remove constant features before model ingestion to reduce noise.",
                    )
                )

            if tab.near_zero_variance_columns:
                score -= 5
                risks.append(
                    RiskItem(
                        severity="LOW",
                        category="features",
                        title=f"Near-Zero Variance Features ({len(tab.near_zero_variance_columns)})",
                        message=f"Columns {tab.near_zero_variance_columns} show near-zero variance (std < 1e-4).",
                        recommendation="Apply variance threshold filtering (e.g., VarianceThreshold(0.01)).",
                    )
                )

            if tab.high_missing_columns:
                score -= 10
                risks.append(
                    RiskItem(
                        severity="HIGH",
                        category="features",
                        title=f"High Missingness in Features ({len(tab.high_missing_columns)})",
                        message=f"Columns {tab.high_missing_columns} contain over 20% missing values.",
                        recommendation="Impute missing features with median/KNN or drop columns with >50% missing values.",
                    )
                )

            if tab.collinear_pairs:
                score -= 5
                example_pairs = [f"{p[0]} vs {p[1]} (|r|={p[2]:.2f})" for p in tab.collinear_pairs[:3]]
                risks.append(
                    RiskItem(
                        severity="LOW",
                        category="features",
                        title=f"Multicollinear Feature Pairs ({len(tab.collinear_pairs)})",
                        message=f"High pairwise correlation (|r| > 0.90) detected: {', '.join(example_pairs)}.",
                        recommendation="Prune redundant features or apply PCA dimensionality reduction to prevent coefficient inflation.",
                    )
                )

        primary_target_name = next(iter(targets))
        t_profile = targets[primary_target_name]

        rec_split = "scaffold" if chem.scaffold_diversity_ratio > 0.1 else "random"
        rec_encoder = "dmpnn"
        rec_model_type = "hybrid" if (has_tabular or chem.total_records < 2000) else "graph"
        rec_head = "standard"
        rec_loss = "huber"
        rec_metrics = ["mae", "rmse", "r2"]
        rec_pos_weight = None
        target_scaling = False
        ensemble_size = 3 if chem.total_records < 800 else 1

        if t_profile.task_kind == "binary_classification":
            rec_head = "classification"
            rec_metrics = ["auroc", "auprc", "accuracy"]

            if t_profile.class_imbalance_ratio > 5.0:
                score -= 15
                pos_c = t_profile.class_counts.get("1", 1)
                neg_c = t_profile.class_counts.get("0", 1)
                rec_pos_weight = float(neg_c / max(1, pos_c))
                rec_loss = "focal"
                rec_metrics = ["auroc", "auprc", "f1"]
                risks.append(
                    RiskItem(
                        severity="HIGH",
                        category="imbalance",
                        title=f"Severe Class Imbalance ({t_profile.class_imbalance_ratio:.1f}:1)",
                        message=f"Class ratio is {neg_c}:{pos_c}. Easy negative molecules will drown gradient updates.",
                        recommendation=f"Use Focal Loss (gamma=2.0) with pos_weight={rec_pos_weight:.2f}. Prioritize AUPRC and AUROC over Accuracy.",
                    )
                )
            elif t_profile.class_imbalance_ratio > 2.0:
                rec_loss = "bce"
                score -= 5
                pos_c = t_profile.class_counts.get("1", 1)
                neg_c = t_profile.class_counts.get("0", 1)
                rec_pos_weight = float(neg_c / max(1, pos_c))
                risks.append(
                    RiskItem(
                        severity="LOW",
                        category="imbalance",
                        title="Moderate Class Imbalance",
                        message=f"Imbalance ratio is {t_profile.class_imbalance_ratio:.1f}:1.",
                        recommendation="Evaluate performance using both AUROC and AUPRC.",
                    )
                )
            else:
                rec_loss = "bce"
        else:
            rec_loss = "mse"
            rec_metrics = ["mae", "rmse", "r2", "spearman"]

            if t_profile.skewness is not None and abs(t_profile.skewness) > 1.5:
                score -= 10
                target_scaling = True
                risks.append(
                    RiskItem(
                        severity="MEDIUM",
                        category="distribution",
                        title=f"High Target Skewness (skew = {t_profile.skewness:.2f})",
                        message="Target distribution is heavily skewed, which can destabilize MSE gradient steps.",
                        recommendation="Use Huber loss with target_scaling=true or logarithmic transformation.",
                    )
                )

            if t_profile.outliers_fraction > 0.03:
                score -= 10
                rec_loss = "huber"
                risks.append(
                    RiskItem(
                        severity="MEDIUM",
                        category="distribution",
                        title=f"Significant Outliers ({t_profile.outliers_fraction:.1%})",
                        message=f"{t_profile.outliers_count} outlier values detected in target distribution.",
                        recommendation="Huber loss is strongly recommended over MSE to downweight outlier gradients.",
                    )
                )
            elif chem.total_records >= 1000:
                rec_head = "evidential"
                rec_loss = "evidential"
            elif chem.total_records >= 250 and not (t_profile.outliers_fraction > 0.03):
                rec_loss = "pairwise_ranking"

        score = max(10, min(100, score))
        if score >= 85:
            status = "EXCELLENT"
        elif score >= 70:
            status = "GOOD"
        elif score >= 50:
            status = "FAIR"
        else:
            status = "NEEDS_ATTENTION"

        use_virtual_node = bool(chem.tanimoto_diversity_score > 0.4 and chem.total_records >= 100)
        rec_aggregation = "attention" if chem.total_records >= 150 else "mean"

        if use_virtual_node:
            risks.append(
                RiskItem(
                    severity="LOW",
                    category="architecture",
                    title="High Chemical Diversity & Scaffold Hopping",
                    message=f"Mean pairwise Tanimoto similarity is {chem.mean_tanimoto_similarity:.2f} (Diversity: {chem.tanimoto_diversity_score:.2f}).",
                    recommendation="Global Virtual Supernode and Graph Attention readout enabled to prevent over-squashing across distant pharmacophores.",
                )
            )

        config_dict = {
            "name": f"{dataset_name}-recommended",
            "dataset": {
                "path": source_path or f"data/{dataset_name}.csv",
                "smiles_column": smiles_column,
                "targets": {
                    primary_target_name: {
                        "column": primary_target_name,
                    }
                },
            },
            "features": {
                "composition": {
                    "type": "elements",
                    "elements": ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"],
                },
                "bonds": {
                    "type": "bond_pairs",
                    "elements": ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"],
                    "bond_orders": ["1", "2", "3", "a"],
                },
                "aromatic": {
                    "type": "smarts",
                    "pattern": "a",
                    "mode": "count",
                },
            },
            "routes": {
                primary_target_name: ["composition", "bonds", "aromatic"],
            },
            "tasks": {
                primary_target_name: {
                    "target": primary_target_name,
                    "kind": "scalar" if t_profile.task_kind == "scalar_regression" else t_profile.task_kind,
                    "route": primary_target_name,
                }
            },
            "split": {
                "method": rec_split,
                "fractions": [0.8, 0.1, 0.1],
                "seed": self.seed,
            },
            "model": {
                "structure": {
                    "encoder": rec_encoder,
                    "hidden_dim": 128 if chem.total_records > 1000 else 64,
                    "depth": 3,
                    "aggregation": rec_aggregation,
                    "dropout": 0.1 if chem.total_records > 1000 else 0.2,
                    "virtual_node": use_virtual_node,
                },
                "descriptors": {
                    "enabled": rec_model_type == "hybrid",
                    "hidden_dim": 64,
                    "depth": 1,
                    "dropout": 0.1,
                },
                "fusion": "concat",
                "ffn": {
                    "hidden_dim": 128,
                    "depth": 2,
                    "dropout": 0.1,
                },
                "head": "auto" if rec_head == "standard" else rec_head,
            },
            "training": {
                "epochs": 35 if chem.total_records > 1000 else 50,
                "batch_size": 64 if chem.total_records > 2000 else 32,
                "learning_rate": 0.001,
                "loss": rec_loss,
                "scheduler": "cosine",
                "warmup_epochs": 3,
                "grad_clip": 1.0,
                "patience": 15,
                "ensemble_size": ensemble_size,
                "device": "auto",
                "checkpoint_dir": "outputs/checkpoints",
            },
            "evaluation": {
                "metrics": rec_metrics,
            },
            "analysis": {
                "enabled": True,
                "output_dir": "outputs/analysis",
            },
            "cache": {
                "enabled": True,
                "directory": ".chemistry_cache/features",
            },
        }

        return AdvisorRecommendation(
            health_score=score,
            health_status=status,
            risks=risks,
            recommended_split=rec_split,
            recommended_encoder=rec_encoder,
            recommended_model_type=rec_model_type,
            recommended_head=rec_head,
            recommended_loss=rec_loss,
            recommended_metrics=rec_metrics,
            recommended_batch_size=config_dict["training"]["batch_size"],
            recommended_epochs=config_dict["training"]["epochs"],
            recommended_lr=config_dict["training"]["learning_rate"],
            recommended_scheduler="cosine",
            recommended_warmup_epochs=3,
            recommended_ensemble_size=ensemble_size,
            recommended_pos_weight=rec_pos_weight,
            target_scaling=target_scaling,
            feature_imputation=feature_imputation,
            feature_scaling=feature_scaling,
            config_dict=config_dict,
        )

    def format_console_report(self, report: DatasetReport) -> str:
        chem = report.chemical_profile
        rec = report.recommendation
        tab = report.tabular_profile

        status_icons = {
            "EXCELLENT": "🟢 EXCELLENT",
            "GOOD": "🟡 GOOD",
            "FAIR": "🟠 FAIR",
            "NEEDS_ATTENTION": "🔴 NEEDS ATTENTION",
        }
        status_badge = status_icons.get(rec.health_status, rec.health_status)

        lines = [
            "",
            "=" * 78,
            f" 🧪 CHEMISTRY FRAMEWORK: DATASET HEALTH & ADVISORY REPORT",
            "=" * 78,
            f" Dataset: {report.dataset_name:<30} Health Score: {rec.health_score}/100 [{status_badge}]",
            "-" * 78,
            f" • Total Compounds:      {chem.total_records:<10} • Valid RDKit SMILES: {chem.valid_smiles_count} ({1 - chem.invalid_fraction:.1%})",
            f" • Unique Scaffolds:     {chem.unique_scaffolds_count:<10} • Scaffold Diversity: {chem.scaffold_diversity_ratio:.2f}",
            f" • Mean Tanimoto Sim:    {chem.mean_tanimoto_similarity:<10.2f} • Tanimoto Diversity: {chem.tanimoto_diversity_score:.2f}",
            f" • Duplicate Molecules:   {chem.duplicate_count:<10} • Invalid SMILES:     {chem.invalid_smiles_count}",
            "-" * 78,
            " 📊 KEY CHEMICAL PROPERTY DISTRIBUTIONS (Mean ± Std [Min, Max]):",
        ]
        for p_name, s in chem.property_stats.items():
            lines.append(f"   - {p_name:<24}: {s.mean:>6.1f} ± {s.std:<5.1f} [{s.min:>5.1f}, {s.max:>5.1f}]")

        if tab is not None and tab.feature_count > 0:
            lines.extend([
                "-" * 78,
                f" 🔬 INPUT TABULAR FEATURES ({tab.feature_count} features detected):",
                f"   - Constant features:     {len(tab.constant_columns)} ({tab.constant_columns or 'None'})",
                f"   - High missing (>20%):   {len(tab.high_missing_columns)} ({tab.high_missing_columns or 'None'})",
                f"   - Collinear pairs (>0.9): {len(tab.collinear_pairs)} pairs",
                f"   - Retained for modeling: {len(tab.recommended_keep_features)} / {tab.feature_count}",
                f"   - Preprocessing advice:  Imputation='{rec.feature_imputation or 'none'}', Scaling='{rec.feature_scaling or 'none'}'",
            ])

        lines.extend([
            "-" * 78,
            " 🎯 TARGET ANALYSIS:",
        ])
        for t_name, t in report.target_profiles.items():
            if t.task_kind == "binary_classification":
                lines.append(f"   - Target '{t_name}' [Binary Classification]:")
                lines.append(f"     Class distribution: {t.class_counts} | Imbalance Ratio: {t.class_imbalance_ratio:.2f}:1")
            else:
                lines.append(f"   - Target '{t_name}' [Scalar Regression]:")
                lines.append(f"     Mean: {t.mean:.2f} ± {t.std:.2f} | Range: [{t.min:.2f}, {t.max:.2f}] | Skewness: {t.skewness:.2f}")

        lines.extend([
            "-" * 78,
            f" ⚠️  RISK FACTORS & WARNINGS ({len(rec.risks)} identified):",
        ])
        if not rec.risks:
            lines.append("   ✓ No significant data risks identified.")
        else:
            for r in rec.risks:
                lines.append(f"   [{r.severity}] {r.title}: {r.message}")
                lines.append(f"          ↳ Advice: {r.recommendation}")

        vn_status = "ON" if rec.config_dict.get("model", {}).get("structure", {}).get("virtual_node") else "OFF"
        lines.extend([
            "-" * 78,
            " 💡 PRESCRIPTIVE TRAINING RECOMMENDATIONS:",
            f"   1. Splitting:      {rec.recommended_split.upper()} (Prevents test memorization)",
            f"   2. Architecture:   {rec.recommended_encoder.upper()} with {rec.recommended_model_type.upper()} (Virtual Supernode: {vn_status})",
            f"   3. Head & Loss:    Head='{rec.recommended_head}', Loss='{rec.recommended_loss}'",
            f"   4. Scheduler:      Cosine Annealing with {rec.recommended_warmup_epochs} warmup epochs",
            f"   5. Ensembling:     Ensemble size = {rec.recommended_ensemble_size}",
            f"   6. Metrics:        {', '.join(rec.recommended_metrics).upper()}",
            "=" * 78,
            "",
        ])
        return "\n".join(lines)

    def format_markdown_report(self, report: DatasetReport) -> str:
        chem = report.chemical_profile
        rec = report.recommendation
        tab = report.tabular_profile

        lines = [
            f"# Chemical Dataset Analysis & Advisory Report: `{report.dataset_name}`",
            "",
            f"**Health Score**: `{rec.health_score}/100` ({rec.health_status})  ",
            f"**Total Records**: `{chem.total_records}` | **Valid Molecules**: `{chem.valid_smiles_count}` | **Unique Scaffolds**: `{chem.unique_scaffolds_count}`  ",
            f"**Tanimoto Diversity**: `{chem.tanimoto_diversity_score:.2f}` (Mean similarity: `{chem.mean_tanimoto_similarity:.2f}`)",
            "",
            "## 1. Executive Summary & Health Check",
            f"The dataset contains **{chem.total_records}** compounds with **{chem.scaffold_diversity_ratio:.1%}** Bemis-Murcko scaffold diversity and **{chem.tanimoto_diversity_score:.2f}** chemical space diversity. ",
            f"Overall health rating is **{rec.health_status}** ({rec.health_score}/100).",
            "",
            "### Risk Assessment Table",
            "| Severity | Category | Risk Factor | Recommendation |",
            "| :--- | :--- | :--- | :--- |",
        ]
        if not rec.risks:
            lines.append("| `INFO` | Quality | None | Data is well-conditioned for model training. |")
        else:
            for r in rec.risks:
                lines.append(f"| `{r.severity}` | `{r.category}` | **{r.title}**: {r.message} | {r.recommendation} |")

        lines.extend([
            "",
            "## 2. Chemical Property Profiling",
            "",
            "| Molecular Property | Mean ± Std | 25% | Median | 75% | Range [Min, Max] |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for p_name, s in chem.property_stats.items():
            lines.append(f"| **{p_name}** | {s.mean:.1f} ± {s.std:.1f} | {s.q25:.1f} | {s.median:.1f} | {s.q75:.1f} | [{s.min:.1f}, {s.max:.1f}] |")

        if tab is not None and tab.feature_count > 0:
            lines.extend([
                "",
                "## 3. Tabular Features Profiling & Quality Check",
                "",
                f"- **Total Tabular Features Detected**: `{tab.feature_count}`",
                f"- **Features Recommended for Training**: `{len(tab.recommended_keep_features)}`",
                f"- **Imputation Strategy**: `{rec.feature_imputation or 'None'}`",
                f"- **Scaling Strategy**: `{rec.feature_scaling or 'None'}`",
                "",
                "| Feature | Missing % | Mean ± Std | Variance | Skewness | Status |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            for f_name, cs in tab.column_stats.items():
                status = "✅ Pass"
                if cs.is_constant:
                    status = "⚠️ Constant"
                elif cs.is_near_zero_variance:
                    status = "⚠️ Near-zero var"
                elif cs.missing_fraction > 0.20:
                    status = "⚠️ High missing"
                lines.append(
                    f"| `{f_name}` | {cs.missing_fraction:.1%} | {cs.mean:.2f} ± {cs.std:.2f} | {cs.variance:.4f} | {cs.skewness:.2f} | {status} |"
                )

            if tab.collinear_pairs:
                lines.extend([
                    "",
                    "### Collinear Feature Pairs (|r| > 0.90)",
                    "| Feature 1 | Feature 2 | Pearson Correlation r | Action |",
                    "| :--- | :--- | :---: | :--- |",
                ])
                for f1, f2, r in tab.collinear_pairs:
                    lines.append(f"| `{f1}` | `{f2}` | `{r:.3f}` | Prune redundant feature or apply PCA |")

        lines.extend([
            "",
            "## 4. Target Distribution Analysis",
            "",
        ])
        for t_name, t in report.target_profiles.items():
            lines.append(f"### Target: `{t_name}`")
            lines.append(f"- **Task Type**: `{t.task_kind}`")
            lines.append(f"- **Valid Entries**: `{t.valid_count}` / {t.total_count} ({t.missing_fraction:.1%} missing)")
            if t.task_kind == "binary_classification":
                lines.append(f"- **Class Distribution**: {t.class_counts}")
                lines.append(f"- **Imbalance Ratio**: `{t.class_imbalance_ratio:.2f}:1`")
            else:
                lines.append(f"- **Distribution**: Mean `{t.mean:.2f}` ± `{t.std:.2f}` | Median `{t.median:.2f}`")
                lines.append(f"- **Skewness**: `{t.skewness:.2f}` | **Outliers**: `{t.outliers_count}` ({t.outliers_fraction:.1%})")
            lines.append("")

        lines.extend([
            "## 5. Prescriptive Action Plan & Recommended Training Recipe",
            "",
            "Based on the profiling above, the automated advisor recommends the following configuration:",
            "",
            f"1. **Validation Split**: `{rec.recommended_split}` split to evaluate out-of-scaffold generalization.",
            f"2. **Representation Architecture**: `{rec.recommended_encoder.upper()}` structure encoder combined with `{rec.recommended_model_type}` descriptor fusion.",
            f"3. **Optimization & Head**: Output head `{rec.recommended_head}`, Loss `{rec.recommended_loss}`.",
            f"4. **Learning Rate & Scheduler**: `Cosine` schedule with {rec.recommended_warmup_epochs} warmup epochs.",
            f"5. **Uncertainty & Ensembling**: `{rec.recommended_ensemble_size}` models in ensemble.",
            "",
            "### Recommended YAML Configuration",
            "```yaml",
            yaml.dump(rec.config_dict, sort_keys=False).strip(),
            "```",
            "",
        ])

        return "\n".join(lines)

__all__ = [
    "AdvisorRecommendation",
    "ChemicalProfile",
    "ColumnFeatureSummary",
    "DatasetAdvisor",
    "DatasetReport",
    "PropertySummary",
    "RiskItem",
    "TabularFeatureProfile",
    "TargetProfile",
]
