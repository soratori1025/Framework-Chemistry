"""Typed experiment configuration and validation."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value

def _only_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(sorted(unknown))}")

def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)

def _parse_graph_config(value: Any) -> dict[str, Any]:
    data = _mapping(value, "graph")
    _only_keys(
        data,
        {"encoder", "aggregation", "depth", "hidden_dim", "atom_smarts_flags",
         "bond_smarts_flags", "options"},
        "graph",
    )
    if data.get("aggregation", "sum") != "sum":
        raise ValueError("graph.aggregation is fixed to 'sum' to preserve size-additive behavior")
    for field_name in ("depth", "hidden_dim"):
        if field_name in data and (not isinstance(data[field_name], int) or data[field_name] < 1):
            raise ValueError(f"graph.{field_name} must be a positive integer")
    for field_name in ("atom_smarts_flags", "bond_smarts_flags"):
        flags = data.get(field_name, [])
        if not isinstance(flags, list):
            raise TypeError(f"graph.{field_name} must be a list")
        names: set[str] = set()
        normalized: list[dict[str, str]] = []
        for index, raw_flag in enumerate(flags):
            flag = _mapping(raw_flag, f"graph.{field_name}[{index}]")
            _only_keys(flag, {"name", "pattern"}, f"graph.{field_name}[{index}]")
            name, pattern = flag.get("name"), flag.get("pattern")
            if not isinstance(name, str) or not name or not isinstance(pattern, str) or not pattern:
                raise ValueError(f"graph.{field_name}[{index}] needs non-empty name and pattern")
            if name in names:
                raise ValueError(f"duplicate graph feature flag name {name!r}")
            names.add(name)
            normalized.append({"name": name, "pattern": pattern})
        data[field_name] = normalized
    if "encoder" in data and not isinstance(data["encoder"], str):
        raise TypeError("graph.encoder must be a string")
    if "options" in data:
        data["options"] = _mapping(data["options"], "graph.options")
    data.setdefault("aggregation", "sum")
    return data

@dataclass(frozen=True)
class TargetConfig:
    columns: tuple[str, ...]
    unit: str = ""
    kind: str = "scalar"
    @classmethod
    def parse(cls, value: Any, name: str) -> TargetConfig:
        if isinstance(value, str):
            return cls(columns=(value,))
        data = _mapping(value, f"dataset.targets.{name}")
        _only_keys(data, {"column", "columns", "unit", "kind"}, f"dataset.targets.{name}")
        column = data.get("column")
        columns = data.get("columns")
        if column is not None and columns is not None:
            raise ValueError(f"dataset.targets.{name} specify either column or columns, not both")
        if column is not None:
            if not isinstance(column, str) or not column:
                raise ValueError(f"dataset.targets.{name}.column must be a non-empty string")
            target_columns = (column,)
        else:
            target_columns = _string_list(columns, f"dataset.targets.{name}.columns")
            if not target_columns:
                raise ValueError(f"dataset.targets.{name}.columns must not be empty")
        unit = data.get("unit", "")
        kind = data.get("kind", "scalar")
        if not isinstance(unit, str) or not isinstance(kind, str):
            raise TypeError(f"dataset.targets.{name}.unit and kind must be strings")
        if kind not in {"scalar", "vector", "temperature_dependent", "custom"}:
            raise ValueError(f"unsupported target kind {kind!r} for {name}")
        if kind == "scalar" and len(target_columns) != 1:
            raise ValueError(f"scalar target {name!r} must have exactly one column")
        return cls(target_columns, unit, kind)

@dataclass(frozen=True)
class DatasetConfig:
    path: str
    smiles_column: str = "smiles"
    targets: dict[str, TargetConfig] = field(default_factory=dict)
    delimiter: str = ","
    encoding: str = "utf-8"
    @classmethod
    def parse(cls, value: Any) -> DatasetConfig:
        data = _mapping(value, "dataset")
        _only_keys(data, {"path", "smiles_column", "targets", "delimiter", "encoding"}, "dataset")
        path = data.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("dataset.path must be a non-empty path")
        raw_targets = _mapping(data.get("targets", {}), "dataset.targets")
        targets = {name: TargetConfig.parse(config, name) for name, config in raw_targets.items()}
        smiles_column = data.get("smiles_column", "smiles")
        delimiter = data.get("delimiter", ",")
        encoding = data.get("encoding", "utf-8")
        if not all(isinstance(item, str) and item for item in (smiles_column, encoding)):
            raise ValueError("dataset.smiles_column and encoding must be non-empty strings")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("dataset.delimiter must be exactly one character")
        return cls(path, smiles_column, targets, delimiter, encoding)

@dataclass(frozen=True)
class FeatureConfig:
    type: str
    options: dict[str, Any] = field(default_factory=dict)
    @classmethod
    def parse(cls, value: Any, name: str) -> FeatureConfig:
        data = _mapping(value, f"features.{name}")
        if "type" not in data or not isinstance(data["type"], str):
            raise ValueError(f"features.{name}.type must be a string")
        feature_type = data["type"]
        if feature_type not in {"elements", "bond_pairs", "smarts", "python", "benson_groups"}:
            raise ValueError(f"unsupported feature type {feature_type!r} for {name}")
        options = {key: item for key, item in data.items() if key != "type"}
        if feature_type in {"elements", "bond_pairs"}:
            elements = _string_list(options.get("elements"), f"features.{name}.elements")
            if not elements:
                raise ValueError(f"features.{name}.elements must not be empty")
            allowed = {"elements"} if feature_type == "elements" else {"elements", "bond_orders"}
            _only_keys(options, allowed, f"features.{name}")
            if feature_type == "bond_pairs":
                orders = _string_list(
                    options.get("bond_orders", ["1", "2", "3", "a"]),
                    f"features.{name}.bond_orders",
                )
                if not orders or set(orders) - {"1", "2", "3", "a"}:
                    raise ValueError(f"features.{name}.bond_orders must use a non-empty subset of 1, 2, 3, a")
        elif feature_type == "smarts":
            _only_keys(options, {"pattern", "mode", "applies_to"}, f"features.{name}")
            if not isinstance(options.get("pattern"), str) or not options["pattern"]:
                raise ValueError(f"features.{name}.pattern must be a SMARTS string")
            mode = options.get("mode", "count")
            if not isinstance(mode, str) or mode not in {"count", "presence"}:
                raise ValueError(f"features.{name}.mode must be count or presence")
            if "applies_to" in options:
                _string_list(options["applies_to"], f"features.{name}.applies_to")
        elif feature_type == "python":
            _only_keys(options, {"fn", "domain", "version"}, f"features.{name}")
            if not isinstance(options.get("fn"), str) or ":" not in options["fn"]:
                raise ValueError(f"features.{name}.fn must use 'module:function' syntax")
            if "version" in options and not isinstance(options["version"], str):
                raise TypeError(f"features.{name}.version must be a string")
        elif feature_type == "benson_groups":
            _only_keys(options, {"fn", "min_count", "version"}, f"features.{name}")
            min_count = options.get("min_count", 1)
            if not isinstance(min_count, int) or min_count < 1:
                raise ValueError(f"features.{name}.min_count must be a positive integer")
            if not isinstance(options.get("fn"), str) or ":" not in options["fn"]:
                raise ValueError(
                    f"features.{name}: Benson groups require a user-supplied rule provider "
                    "via fn until a versioned Benson rule library is configured"
                )
            if "version" in options and not isinstance(options["version"], str):
                raise TypeError(f"features.{name}.version must be a string")
        return cls(feature_type, options)

@dataclass(frozen=True)
class PriorConfig:
    use: tuple[str, ...] = ()
    fixed: dict[str, float] = field(default_factory=dict)
    fit: str = "ridge"
    @classmethod
    def parse(cls, value: Any, name: str) -> PriorConfig:
        data = _mapping(value, f"priors.{name}")
        _only_keys(data, {"use", "fixed", "fit"}, f"priors.{name}")
        use = _string_list(data.get("use", []), f"priors.{name}.use")
        fixed_raw = _mapping(data.get("fixed", {}), f"priors.{name}.fixed")
        fixed: dict[str, float] = {}
        for feature, coefficient in fixed_raw.items():
            if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool):
                raise TypeError(f"priors.{name}.fixed.{feature} must be numeric")
            fixed[feature] = float(coefficient)
        fit = data.get("fit", "ridge")
        if not isinstance(fit, str) or fit not in {"ridge", "least_squares", "none"}:
            raise ValueError(f"unsupported prior fit method {fit!r}")
        return cls(use, fixed, fit)

@dataclass(frozen=True)
class SplitConfig:
    method: str = "random"
    fractions: tuple[float, float, float] = (0.81, 0.09, 0.10)
    seed: int = 0
    test_smarts: str | None = None

    @classmethod
    def parse(cls, value: Any) -> SplitConfig:
        data = _mapping(value, "split")
        _only_keys(data, {"method", "fractions", "seed", "test_smarts"}, "split")
        method = data.get("method", "random")
        if not isinstance(method, str) or method not in {"random", "size", "scaffold", "smarts"}:
            raise ValueError("split.method must be random, size, scaffold, or smarts")
        fractions_raw = data.get("fractions", [0.81, 0.09, 0.10])
        if not isinstance(fractions_raw, (list, tuple)) or len(fractions_raw) != 3:
            raise ValueError("split.fractions must contain train, validation, test fractions")
        if any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in fractions_raw
        ):
            raise TypeError("split.fractions must contain numbers")
        fractions: tuple[float, float, float] = (
            float(fractions_raw[0]),
            float(fractions_raw[1]),
            float(fractions_raw[2]),
        )
        if (
            any(not math.isfinite(item) or item <= 0 for item in fractions)
            or abs(sum(fractions) - 1.0) > 1e-8
        ):
            raise ValueError("split.fractions must be positive and sum to 1")
        seed = data.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("split.seed must be an integer")
        test_smarts = data.get("test_smarts")
        if method == "smarts" and (not isinstance(test_smarts, str) or not test_smarts):
            raise ValueError("split.test_smarts is required when split.method is 'smarts'")
        if test_smarts is not None and not isinstance(test_smarts, str):
            raise TypeError("split.test_smarts must be a string")
        return cls(method, fractions, seed, test_smarts)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    loss: str = "mse"
    patience: int | None = None
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def parse(cls, value: Any) -> TrainingConfig:
        data = _mapping(value, "training")
        _only_keys(
            data,
            {"epochs", "batch_size", "learning_rate", "weight_decay", "seed", "device",
             "num_workers", "loss", "patience", "checkpoint_dir"},
            "training",
        )
        config = cls(**data)
        integer_fields = {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "num_workers": config.num_workers,
        }
        if any(not isinstance(item, int) or isinstance(item, bool) for item in integer_fields.values()):
            raise TypeError("training epochs, batch_size, seed, and num_workers must be integers")
        if config.patience is not None and (
            not isinstance(config.patience, int) or isinstance(config.patience, bool)
        ):
            raise TypeError("training.patience must be an integer or null")
        if config.epochs < 1 or config.batch_size < 1:
            raise ValueError("training.epochs and batch_size must be positive")
        if (
            not isinstance(config.learning_rate, (int, float))
            or isinstance(config.learning_rate, bool)
            or not isinstance(config.weight_decay, (int, float))
            or isinstance(config.weight_decay, bool)
            or not math.isfinite(config.learning_rate)
            or not math.isfinite(config.weight_decay)
        ):
            raise TypeError("training.learning_rate and weight_decay must be finite numbers")
        if config.learning_rate <= 0 or config.weight_decay < 0:
            raise ValueError("training.learning_rate must be positive and weight_decay non-negative")
        if config.num_workers < 0 or (config.patience is not None and config.patience < 1):
            raise ValueError("training.num_workers must be non-negative and patience positive")
        if not isinstance(config.loss, str) or config.loss not in {"mse", "mae", "huber", "custom"}:
            raise ValueError(f"unsupported training.loss {config.loss!r}")
        if not isinstance(config.device, str) or config.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported training.device {config.device!r}")
        if not isinstance(config.checkpoint_dir, str) or not config.checkpoint_dir:
            raise ValueError("training.checkpoint_dir must not be empty")
        return config


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...] = ("mae", "rmse", "r2")
    checkpoint: str | None = None

    @classmethod
    def parse(cls, value: Any) -> EvaluationConfig:
        data = _mapping(value, "evaluation")
        _only_keys(data, {"metrics", "checkpoint"}, "evaluation")
        metrics = _string_list(data.get("metrics", ["mae", "rmse", "r2"]), "evaluation.metrics")
        allowed = {"mae", "rmse", "r2", "mape"}
        unknown = set(metrics) - allowed
        if unknown:
            raise ValueError(f"unsupported evaluation metric(s): {', '.join(sorted(unknown))}")
        checkpoint = data.get("checkpoint")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise TypeError("evaluation.checkpoint must be a path string or null")
        return cls(metrics, checkpoint)


@dataclass(frozen=True)
class AnalysisConfig:
    enabled: bool = True
    correlation_threshold: float = 0.95
    support_threshold: int = 1
    bins: int = 20
    output_dir: str = "analysis"

    @classmethod
    def parse(cls, value: Any) -> AnalysisConfig:
        data = _mapping(value, "analysis")
        _only_keys(
            data,
            {"enabled", "correlation_threshold", "support_threshold", "bins", "output_dir"},
            "analysis",
        )
        config = cls(**data)
        if (
            not isinstance(config.correlation_threshold, (int, float))
            or isinstance(config.correlation_threshold, bool)
            or not math.isfinite(config.correlation_threshold)
            or not 0 < config.correlation_threshold <= 1
        ):
            raise ValueError("analysis.correlation_threshold must be in (0, 1]")
        if (
            not isinstance(config.support_threshold, int)
            or isinstance(config.support_threshold, bool)
            or not isinstance(config.bins, int)
            or isinstance(config.bins, bool)
            or config.support_threshold < 1
            or config.bins < 1
        ):
            raise ValueError("analysis.support_threshold and bins must be positive")
        if not isinstance(config.enabled, bool):
            raise TypeError("analysis.enabled must be a boolean")
        if not isinstance(config.output_dir, str) or not config.output_dir:
            raise ValueError("analysis.output_dir must not be empty")
        return config


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    directory: str = ".teb_cache/features"

    @classmethod
    def parse(cls, value: Any) -> CacheConfig:
        data = _mapping(value, "cache")
        _only_keys(data, {"enabled", "directory"}, "cache")
        config = cls(**data)
        if not isinstance(config.enabled, bool):
            raise TypeError("cache.enabled must be a boolean")
        if not isinstance(config.directory, str) or not config.directory:
            raise ValueError("cache.directory must not be empty")
        return config


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    features: dict[str, FeatureConfig]
    priors: dict[str, PriorConfig]
    routes: dict[str, tuple[str, ...]]
    split: SplitConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    analysis: AnalysisConfig
    cache: CacheConfig
    model: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def parse(cls, value: Any, source_path: Path | None = None) -> ExperimentConfig:
        data = _mapping(value, "configuration")
        _only_keys(
            data,
            {"name", "dataset", "features", "priors", "routes", "split", "training",
             "evaluation", "analysis", "cache", "model", "graph", "tasks"},
            "configuration",
        )
        name = data.get("name", "experiment")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        raw_features = _mapping(data.get("features", {}), "features")
        features = {key: FeatureConfig.parse(item, key) for key, item in raw_features.items()}
        if not features:
            raise ValueError("at least one feature definition is required")
        raw_priors = _mapping(data.get("priors", {}), "priors")
        priors = {key: PriorConfig.parse(item, key) for key, item in raw_priors.items()}
        raw_routes = _mapping(data.get("routes", {}), "routes")
        routes = {key: _string_list(item, f"routes.{key}") for key, item in raw_routes.items()}
        for property_name, config in priors.items():
            if unknown := set(config.use) - set(features):
                raise ValueError(
                    f"priors.{property_name} references unknown feature(s): {', '.join(sorted(unknown))}"
                )
        for property_name, selected in routes.items():
            if unknown := set(selected) - set(features):
                raise ValueError(
                    f"routes.{property_name} references unknown feature(s): {', '.join(sorted(unknown))}"
                )
        for property_name, prior in priors.items():
            invalid_fixed = {
                name
                for name in prior.fixed
                if not any(name == block or name.startswith(f"{block}.") for block in prior.use)
            }
            if invalid_fixed:
                raise ValueError(
                    f"priors.{property_name}.fixed references unselected feature(s): "
                    f"{', '.join(sorted(invalid_fixed))}"
                )
        raw_tasks = _mapping(data.get("tasks", {}), "tasks")
        dataset_config = DatasetConfig.parse(data.get("dataset"))
        tasks: dict[str, dict[str, Any]] = {}
        for task_name, raw_task in raw_tasks.items():
            task = _mapping(raw_task, f"tasks.{task_name}")
            _only_keys(task, {"target", "kind", "prior", "route", "loss_weight"}, f"tasks.{task_name}")
            target = task.get("target")
            if not isinstance(target, str) or target not in dataset_config.targets:
                raise ValueError(f"tasks.{task_name}.target must name a configured dataset target")
            kind = task.get("kind", "scalar")
            if not isinstance(kind, str) or kind not in {
                "scalar", "vector", "temperature_dependent", "custom"
            }:
                raise ValueError(f"tasks.{task_name}.kind is unsupported: {kind!r}")
            target_kind = dataset_config.targets[target].kind
            if kind != "custom" and target_kind != "custom" and kind != target_kind:
                raise ValueError(
                    f"tasks.{task_name}.kind {kind!r} does not match "
                    f"dataset target kind {target_kind!r}"
                )
            prior_name = task.get("prior")
            if prior_name is not None and prior_name not in priors:
                raise ValueError(f"tasks.{task_name}.prior references unknown prior {prior_name!r}")
            route_name = task.get("route")
            if route_name is not None and route_name not in routes:
                raise ValueError(f"tasks.{task_name}.route references unknown route {route_name!r}")
            loss_weight = task.get("loss_weight", 1.0)
            if (
                not isinstance(loss_weight, (int, float))
                or isinstance(loss_weight, bool)
                or not math.isfinite(loss_weight)
                or loss_weight < 0
            ):
                raise ValueError(f"tasks.{task_name}.loss_weight must be a non-negative number")
            tasks[task_name] = {
                **task,
                "kind": kind,
                "loss_weight": float(loss_weight),
            }
        model = _mapping(data.get("model", {}), "model")
        graph = _parse_graph_config(data.get("graph", {}))
        return cls(
            name=name,
            dataset=dataset_config,
            features=features,
            priors=priors,
            routes=routes,
            split=SplitConfig.parse(data.get("split", {})),
            training=TrainingConfig.parse(data.get("training", {})),
            evaluation=EvaluationConfig.parse(data.get("evaluation", {})),
            analysis=AnalysisConfig.parse(data.get("analysis", {})),
            cache=CacheConfig.parse(data.get("cache", {})),
            model=model,
            graph=graph,
            tasks=tasks,
            source_path=source_path,
        )

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute() or self.source_path is None:
            return path
        return (self.source_path.parent / path).resolve()
