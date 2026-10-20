"""Split-seed selection shared by the benchmark runners.

The primary analysis in the manuscript uses the ten scaffold splits stored as
``benchmarks/splits/<dataset>_seed_{1..10}.json`` and every runner defaults to
those seeds so that the released numbers can be regenerated with a single
command.  The same runners accept ``--seeds`` so the whole study can be
repeated on any subset of the twenty Bemis-Murcko scaffold splits that ship
with the repository (seeds 1-20), e.g. a shorter exploratory run::

    python benchmarks/run_canonical_baselines.py --dataset all --seeds 1 2 3

Keeping the seed list in one place means the runner code never silently
substitutes a different split from the one the manuscript describes.
"""
from __future__ import annotations

import argparse
from typing import Iterable, Sequence

#: Seeds used by the primary analysis reported in the manuscript.
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

#: Number of scaffold splits prepared by ``prepare_moleculenet_data.py``.
MAX_STORED_SEED: int = 20


def add_seed_argument(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--seeds`` option to ``parser``."""
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Split-seed indices to evaluate, e.g. '--seeds 1 2 3 4 5'. "
            "Defaults to the ten splits used in the manuscript (%s). "
            "Saved scaffold splits are available for seeds 1-%d."
            % (" ".join(str(s) for s in DEFAULT_SEEDS), MAX_STORED_SEED)
        ),
    )


def resolve_seeds(
    cli_seeds: Sequence[int] | None,
    fallback: Iterable[int] = DEFAULT_SEEDS,
) -> list[int]:
    """Return the validated, ordered list of split seeds for a run.

    ``cli_seeds`` is used when the caller supplied ``--seeds``; otherwise the
    ``fallback`` sequence is returned.  Any seed without a saved scaffold
    split raises instead of being silently skipped.
    """
    seeds = [int(s) for s in (fallback if cli_seeds is None else cli_seeds)]
    if not seeds:
        raise SystemExit("--seeds requires at least one seed index")

    duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate seed indices in --seeds: {duplicates}")

    missing = [s for s in seeds if not 1 <= s <= MAX_STORED_SEED]
    if missing:
        raise SystemExit(
            f"No saved scaffold split for seed(s) {missing}; "
            f"prepared splits cover seeds 1-{MAX_STORED_SEED} "
            "(run benchmarks/prepare_moleculenet_data.py to extend them)."
        )

    return seeds
