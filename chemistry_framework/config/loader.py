from __future__ import annotations
from pathlib import Path
from typing import Any
from .schema import ExperimentConfig

def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML config loading requires PyYAML; install chemistry-framework[config]") from exc
    with config_path.open(encoding="utf-8") as stream:
        document: Any = yaml.safe_load(stream)
    if document is None:
        raise ValueError(f"configuration file is empty: {config_path}")
    return ExperimentConfig.parse(document, source_path=config_path)
