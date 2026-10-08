"""CSV-backed molecule and target loading."""
from __future__ import annotations
import csv
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from ..config.schema import DatasetConfig, ExperimentConfig

@dataclass(frozen=True)
class MolecularDataset:
    smiles: tuple[str, ...]
    targets: dict[str, tuple[float, ...] | tuple[tuple[float, ...], ...]]
    source_rows: tuple[int, ...]
    source_path: Path | None

    def __len__(self) -> int:
        return len(self.smiles)

def load_csv_dataset(dataset_config: DatasetConfig, *, base_dir: Path | None = None,) -> MolecularDataset:
    """Load SMILES and configured scalar targets from a delimited text file."""
    if dataset_config.provider != "csv" or dataset_config.path is None:
        raise ValueError("load_csv_dataset requires dataset.provider='csv' and dataset.path")
    path = Path(dataset_config.path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    smiles: list[str] = []
    target_values: dict[str, list[tuple[float, ...]]] = {
        name: [] for name in dataset_config.targets
    }
    source_rows: list[int] = []
    with path.open(encoding=dataset_config.encoding, newline="") as stream:
        reader = csv.DictReader(stream, delimiter=dataset_config.delimiter)
        headers = reader.fieldnames or []
        required = [dataset_config.smiles_column] + [
            column
            for target in dataset_config.targets.values()
            for column in target.columns
        ]
        missing = sorted(set(required) - set(headers))
        if missing:
            raise ValueError(f"dataset {path} is missing required column(s): {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_smiles = (row.get(dataset_config.smiles_column) or "").strip()
            if not raw_smiles:
                raise ValueError(f"empty SMILES in {path} at CSV row {row_number}")
            smiles.append(raw_smiles)
            source_rows.append(row_number)
            for name, target in dataset_config.targets.items():
                row_values: list[float] = []
                for column in target.columns:
                    raw_value = (row.get(column) or "").strip()
                    try:
                        value = float(raw_value) if raw_value else math.nan
                    except ValueError as exc:
                        raise ValueError(
                            f"target {name!r} is not numeric at CSV row {row_number}, "
                            f"column {column!r}: {raw_value!r}"
                        ) from exc
                    row_values.append(value)
                target_values[name].append(tuple(row_values))
    if not smiles:
        raise ValueError(f"dataset contains no records: {path}")
    return MolecularDataset(
        tuple(smiles),
        {
            name: (
                tuple(row[0] for row in values)
                if dataset_config.targets[name].kind == "scalar"
                else tuple(values)
            )
            for name, values in target_values.items()
        },
        tuple(source_rows),
        path,
    )

def load_tdc_dataset(dataset_config: DatasetConfig) -> MolecularDataset:
    """Load one TDC ADME or toxicity dataset using the optional PyTDC package."""
    if dataset_config.provider not in {"tdc_adme", "tdc_tox"} or not dataset_config.name:
        raise ValueError("load_tdc_dataset requires a named TDC dataset provider")
    try:
        tdc_single_pred = importlib.import_module("tdc.single_pred")
    except ImportError as exc:
        raise ImportError(
            "PyTDC is required for TDC datasets; install chemistry-framework[benchmark]"
        ) from exc
    dataset_class_name = "ADME" if dataset_config.provider == "tdc_adme" else "Tox"
    dataset_type = getattr(tdc_single_pred, dataset_class_name, None)
    if not callable(dataset_type):
        raise ImportError(f"PyTDC does not provide tdc.single_pred.{dataset_class_name}")
    frame = dataset_type(name=dataset_config.name).get_data()
    required = [dataset_config.smiles_column] + [
        column
        for target in dataset_config.targets.values()
        for column in target.columns
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"TDC dataset {dataset_config.name!r} is missing required column(s): "
            f"{', '.join(missing)}"
        )
    raw_smiles = frame[dataset_config.smiles_column].tolist()
    smiles: list[str] = []
    target_values: dict[str, list[tuple[float, ...]]] = {
        name: [] for name in dataset_config.targets
    }
    source_rows: list[int] = []
    target_columns = {
        name: [frame[column].tolist() for column in target.columns]
        for name, target in dataset_config.targets.items()
    }
    for index, raw_value in enumerate(raw_smiles):
        if raw_value is None:
            raise ValueError(
                f"empty SMILES in TDC dataset {dataset_config.name!r} at row {index}"
            )
        raw_smiles_text = str(raw_value).strip()
        if not raw_smiles_text:
            raise ValueError(
                f"empty SMILES in TDC dataset {dataset_config.name!r} at row {index}"
            )
        smiles.append(raw_smiles_text)
        source_rows.append(index)
        for name, columns in target_columns.items():
            row_values: list[float] = []
            for column_values in columns:
                raw_target = column_values[index]
                if raw_target is None or (isinstance(raw_target, str) and not raw_target.strip()):
                    row_values.append(math.nan)
                    continue
                try:
                    row_values.append(float(raw_target))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"target {name!r} is not numeric in TDC dataset "
                        f"{dataset_config.name!r} at row {index}: {raw_target!r}"
                    ) from exc
            target_values[name].append(tuple(row_values))
    if not smiles:
        raise ValueError(f"TDC dataset contains no records: {dataset_config.name}")
    return MolecularDataset(
        tuple(smiles),
        {
            name: (
                tuple(row[0] for row in values)
                if dataset_config.targets[name].kind == "scalar"
                else tuple(values)
            )
            for name, values in target_values.items()
        },
        tuple(source_rows),
        None,
    )

def load_configured_dataset(config: ExperimentConfig) -> MolecularDataset:
    if config.dataset.provider != "csv":
        return load_tdc_dataset(config.dataset)
    base_dir = config.source_path.parent if config.source_path is not None else None
    return load_csv_dataset(config.dataset, base_dir=base_dir)
