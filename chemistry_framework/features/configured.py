"""Compile configured feature definitions into a deterministic registry."""
from __future__ import annotations
import importlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations_with_replacement
from typing import Any
from ..config.schema import FeatureConfig
from .registry import FeatureRegistry
from .specs import FeatureCalculator, FeatureSpec
import hashlib
import inspect
import json

_FEATURE_ENGINE_VERSION = 2

def load_python_callable(reference: str) -> Callable[..., Any]:
    """Load a trusted user plugin referenced as ``package.module:function``."""
    module_name, function_name = reference.split(":", maxsplit=1)
    if not module_name or not function_name:
        raise ValueError(f"invalid plugin reference {reference!r}; expected 'module:function'")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"configured plugin {reference!r} does not resolve to a callable")
    return function

def _normalise_bond_type(bond_type: str) -> str:
    mapping = {"1": "SINGLE", "2": "DOUBLE", "3": "TRIPLE", "a": "AROMATIC"}
    if bond_type not in mapping:
        raise ValueError(
            f"unsupported bond order {bond_type!r}; use '1', '2', '3', or 'a'"
        )
    return mapping[bond_type]

def _bond_pair_calculator(left: str, right: str, bond_type: str) -> FeatureCalculator:
    def calculate(mol: Any) -> float:
        count = 0
        for bond in mol.GetBonds():
            atom_a = bond.GetBeginAtom()
            atom_b = bond.GetEndAtom()
            pair = (atom_a.GetSymbol(), atom_b.GetSymbol())
            if tuple(sorted(pair)) == tuple(sorted((left, right))) and str(
                bond.GetBondType()
            ) == bond_type:
                count += 1
        if "H" in {left, right} and left != right and bond_type == "SINGLE":
            heavy_symbol = right if left == "H" else left
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == heavy_symbol:
                    count += atom.GetTotalNumHs(includeNeighbors=False)
        return float(count)
    return calculate

def build_feature_registry(definitions: Mapping[str, FeatureConfig], molecules: Sequence[Any], train_indices: Sequence[int],) -> tuple[FeatureRegistry, dict[str, tuple[str, ...]]]:
    """Build feature calculators; Benson group vocabulary is fitted on train only.
    A Benson provider plugin must accept an RDKit molecule and return a mapping
    from group label to occurrence count. Its train-set vocabulary is filtered
    by the configured minimum molecule support.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("configured molecular features require RDKit") from exc
    registry = FeatureRegistry()
    resolved_names: dict[str, tuple[str, ...]] = {}
    for block_name, config in definitions.items():
        options = config.options
        names: list[str] = []
        if config.type == "elements":
            elements = options["elements"]

            def element_counter(symbol: str) -> FeatureCalculator:
                return lambda mol: float(
                    sum(atom.GetSymbol() == symbol for atom in mol.GetAtoms())
                    if symbol != "H"
                    else sum(atom.GetTotalNumHs(includeNeighbors=False) for atom in mol.GetAtoms())
                    + sum(atom.GetSymbol() == "H" for atom in mol.GetAtoms())
                )

            for element in elements:
                name = f"{block_name}.{element}"
                registry.register(
                    FeatureSpec(
                        name,
                        element_counter(element),
                        "composition",
                        metadata={"element": element},
                        group=config.group or block_name,
                        description=config.description,
                        interpretation=config.interpretation,
                        source=config.source or "elements",
                        definition=config.definition or f"Count of element {element}",
                    )
                )
                names.append(name)
        elif config.type == "bond_pairs":
            elements = tuple(options["elements"])
            bond_orders = tuple(options.get("bond_orders", ("1", "2", "3", "a")))
            pairs = combinations_with_replacement(elements, 2)
            for left, right in pairs:
                for order in bond_orders:
                    bond_type = _normalise_bond_type(order)
                    order_label = "a" if order == "a" else order
                    name = f"{block_name}.{left}{right}{order_label}"
                    registry.register(
                        FeatureSpec(
                            name,
                            _bond_pair_calculator(left, right, bond_type),
                            "bond_pair",
                            metadata={"elements": [left, right], "bond_order": order},
                            group=config.group or block_name,
                            description=config.description,
                            interpretation=config.interpretation,
                            source=config.source or "bond_pairs",
                            definition=config.definition
                            or f"Count of {order_label} bonds between {left} and {right}",
                        )
                    )
                    names.append(name)
        elif config.type == "smarts":
            name = block_name
            registry.add_smarts(
                name,
                options["pattern"],
                mode=options.get("mode", "count"),
                applies_to=options.get("applies_to", ()),
                group=config.group or block_name,
                description=config.description,
                interpretation=config.interpretation,
                source=config.source or "smarts",
                definition=config.definition,
            )
            names.append(name)
        elif config.type == "python":
            calculator = load_python_callable(options["fn"])
            name = block_name
            registry.register(
                FeatureSpec(
                    name,
                    lambda mol, fn=calculator: float(fn(mol)),
                    options.get("domain", "custom"),
                    metadata={"plugin": options["fn"], "version": options.get("version", "unversioned")},
                    group=config.group or block_name,
                    description=config.description,
                    interpretation=config.interpretation,
                    source=config.source or "python",
                    definition=config.definition or options["fn"],
                )
            )
            names.append(name)
        elif config.type == "benson_groups":
            provider_ref = options["fn"]
            provider = load_python_callable(provider_ref)
            group_counts: Counter[str] = Counter()
            for index in train_indices:
                mol = Chem.MolFromSmiles(molecules[int(index)]) if isinstance(
                    molecules[int(index)], str
                ) else molecules[int(index)]
                if mol is None:
                    raise ValueError(f"invalid molecule at training index {index}")
                result = provider(mol)
                if not isinstance(result, Mapping):
                    raise TypeError(f"Benson provider {provider_ref!r} must return a mapping")
                group_counts.update(
                    group_name for group_name, count in result.items() if float(count) > 0
                )
            minimum = options.get("min_count", 1)
            group_names = sorted(name for name, count in group_counts.items() if count >= minimum)
            for group_name in group_names:
                feature_name = f"{block_name}.{group_name}"

                def count_group(mol: Any, group: str = group_name, fn=provider) -> float:
                    result = fn(mol)
                    if not isinstance(result, Mapping):
                        raise TypeError(f"Benson provider {provider_ref!r} must return a mapping")
                    return float(result.get(group, 0))

                registry.register(
                    FeatureSpec(
                        feature_name,
                        count_group,
                        "benson_group",
                        metadata={"provider": provider_ref, "min_count": minimum},
                        group=config.group or block_name,
                        description=config.description,
                        interpretation=config.interpretation,
                        source=config.source or "benson_groups",
                        definition=config.definition or provider_ref,
                    )
                )
                names.append(feature_name)
        resolved_names[block_name] = tuple(names)
    return registry, resolved_names

def configured_feature_fingerprint(definitions: Mapping[str, FeatureConfig]) -> str:
    """Stable canonical representation used as one part of feature-cache keys."""
    plugin_hashes: dict[str, str] = {}
    for name, config in definitions.items():
        if config.type not in {"python", "benson_groups"}:
            continue
        function = load_python_callable(config.options["fn"])
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            code = getattr(function, "__code__", None)
            if code is None:
                raise ValueError(
                    f"cannot fingerprint plugin {config.options['fn']!r}; "
                    "provide a Python function with inspectable source"
                )
            source = code.co_code.hex() + repr(code.co_consts)
        plugin_hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    payload = {
        "feature_engine_version": _FEATURE_ENGINE_VERSION,
        "definitions": {
            name: {
                "type": config.type,
                "options": config.options,
                "group": config.group,
                "description": config.description,
                "interpretation": config.interpretation,
                "source": config.source,
                "definition": config.definition,
            }
            for name, config in sorted(definitions.items())
        },
        "plugin_hashes": plugin_hashes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
