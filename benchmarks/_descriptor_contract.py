"""Descriptor-column contract shared by every released benchmark artefact.

The published study ran on RDKit 2023.09.6, whose ``Descriptors.descList`` held
210 entries.  Later RDKit releases add descriptors (2026.03.x ships 217), so a
plain re-extraction changes the tabular block that the graph+descriptor models
consume.  Every artefact in ``benchmarks/results`` therefore stores a 210-column
block, and a rerun only stays comparable if it reproduces *that block*.

This module pins the contract by column name (the same names are stored in
``data/moleculenet/.descriptor_cache/*.names.json``) and provides
:func:`select_contract_columns`, which reorders whatever the installed RDKit
returns down to the published columns::

    from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
    from benchmarks._descriptor_contract import select_contract_columns

    matrix, names = select_contract_columns(*extract_rdkit_2d_descriptors(smiles))

The helper raises if a contract column is missing, so a silently shrunken
feature block cannot enter the comparison.

Column *order* and descriptor *values* still depend on the RDKit release; the
OOD/duplicate audit in ``run_structure_audit.py`` documents that sensitivity.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: RDKit release the published 210-column block was extracted with.
PAPER_RDKIT_VERSION = "2023.09.6"

#: Number of descriptor columns in every released artefact.
N_DESCRIPTOR_COLUMNS = 210

PAPER_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "rdkit_2d.MaxAbsEStateIndex",
    "rdkit_2d.MaxEStateIndex",
    "rdkit_2d.MinAbsEStateIndex",
    "rdkit_2d.MinEStateIndex",
    "rdkit_2d.qed",
    "rdkit_2d.SPS",
    "rdkit_2d.MolWt",
    "rdkit_2d.HeavyAtomMolWt",
    "rdkit_2d.ExactMolWt",
    "rdkit_2d.NumValenceElectrons",
    "rdkit_2d.NumRadicalElectrons",
    "rdkit_2d.MaxPartialCharge",
    "rdkit_2d.MinPartialCharge",
    "rdkit_2d.MaxAbsPartialCharge",
    "rdkit_2d.MinAbsPartialCharge",
    "rdkit_2d.FpDensityMorgan1",
    "rdkit_2d.FpDensityMorgan2",
    "rdkit_2d.FpDensityMorgan3",
    "rdkit_2d.BCUT2D_MWHI",
    "rdkit_2d.BCUT2D_MWLOW",
    "rdkit_2d.BCUT2D_CHGHI",
    "rdkit_2d.BCUT2D_CHGLO",
    "rdkit_2d.BCUT2D_LOGPHI",
    "rdkit_2d.BCUT2D_LOGPLOW",
    "rdkit_2d.BCUT2D_MRHI",
    "rdkit_2d.BCUT2D_MRLOW",
    "rdkit_2d.AvgIpc",
    "rdkit_2d.BalabanJ",
    "rdkit_2d.BertzCT",
    "rdkit_2d.Chi0",
    "rdkit_2d.Chi0n",
    "rdkit_2d.Chi0v",
    "rdkit_2d.Chi1",
    "rdkit_2d.Chi1n",
    "rdkit_2d.Chi1v",
    "rdkit_2d.Chi2n",
    "rdkit_2d.Chi2v",
    "rdkit_2d.Chi3n",
    "rdkit_2d.Chi3v",
    "rdkit_2d.Chi4n",
    "rdkit_2d.Chi4v",
    "rdkit_2d.HallKierAlpha",
    "rdkit_2d.Ipc",
    "rdkit_2d.Kappa1",
    "rdkit_2d.Kappa2",
    "rdkit_2d.Kappa3",
    "rdkit_2d.LabuteASA",
    "rdkit_2d.PEOE_VSA1",
    "rdkit_2d.PEOE_VSA10",
    "rdkit_2d.PEOE_VSA11",
    "rdkit_2d.PEOE_VSA12",
    "rdkit_2d.PEOE_VSA13",
    "rdkit_2d.PEOE_VSA14",
    "rdkit_2d.PEOE_VSA2",
    "rdkit_2d.PEOE_VSA3",
    "rdkit_2d.PEOE_VSA4",
    "rdkit_2d.PEOE_VSA5",
    "rdkit_2d.PEOE_VSA6",
    "rdkit_2d.PEOE_VSA7",
    "rdkit_2d.PEOE_VSA8",
    "rdkit_2d.PEOE_VSA9",
    "rdkit_2d.SMR_VSA1",
    "rdkit_2d.SMR_VSA10",
    "rdkit_2d.SMR_VSA2",
    "rdkit_2d.SMR_VSA3",
    "rdkit_2d.SMR_VSA4",
    "rdkit_2d.SMR_VSA5",
    "rdkit_2d.SMR_VSA6",
    "rdkit_2d.SMR_VSA7",
    "rdkit_2d.SMR_VSA8",
    "rdkit_2d.SMR_VSA9",
    "rdkit_2d.SlogP_VSA1",
    "rdkit_2d.SlogP_VSA10",
    "rdkit_2d.SlogP_VSA11",
    "rdkit_2d.SlogP_VSA12",
    "rdkit_2d.SlogP_VSA2",
    "rdkit_2d.SlogP_VSA3",
    "rdkit_2d.SlogP_VSA4",
    "rdkit_2d.SlogP_VSA5",
    "rdkit_2d.SlogP_VSA6",
    "rdkit_2d.SlogP_VSA7",
    "rdkit_2d.SlogP_VSA8",
    "rdkit_2d.SlogP_VSA9",
    "rdkit_2d.TPSA",
    "rdkit_2d.EState_VSA1",
    "rdkit_2d.EState_VSA10",
    "rdkit_2d.EState_VSA11",
    "rdkit_2d.EState_VSA2",
    "rdkit_2d.EState_VSA3",
    "rdkit_2d.EState_VSA4",
    "rdkit_2d.EState_VSA5",
    "rdkit_2d.EState_VSA6",
    "rdkit_2d.EState_VSA7",
    "rdkit_2d.EState_VSA8",
    "rdkit_2d.EState_VSA9",
    "rdkit_2d.VSA_EState1",
    "rdkit_2d.VSA_EState10",
    "rdkit_2d.VSA_EState2",
    "rdkit_2d.VSA_EState3",
    "rdkit_2d.VSA_EState4",
    "rdkit_2d.VSA_EState5",
    "rdkit_2d.VSA_EState6",
    "rdkit_2d.VSA_EState7",
    "rdkit_2d.VSA_EState8",
    "rdkit_2d.VSA_EState9",
    "rdkit_2d.FractionCSP3",
    "rdkit_2d.HeavyAtomCount",
    "rdkit_2d.NHOHCount",
    "rdkit_2d.NOCount",
    "rdkit_2d.NumAliphaticCarbocycles",
    "rdkit_2d.NumAliphaticHeterocycles",
    "rdkit_2d.NumAliphaticRings",
    "rdkit_2d.NumAromaticCarbocycles",
    "rdkit_2d.NumAromaticHeterocycles",
    "rdkit_2d.NumAromaticRings",
    "rdkit_2d.NumHAcceptors",
    "rdkit_2d.NumHDonors",
    "rdkit_2d.NumHeteroatoms",
    "rdkit_2d.NumRotatableBonds",
    "rdkit_2d.NumSaturatedCarbocycles",
    "rdkit_2d.NumSaturatedHeterocycles",
    "rdkit_2d.NumSaturatedRings",
    "rdkit_2d.RingCount",
    "rdkit_2d.MolLogP",
    "rdkit_2d.MolMR",
    "rdkit_2d.fr_Al_COO",
    "rdkit_2d.fr_Al_OH",
    "rdkit_2d.fr_Al_OH_noTert",
    "rdkit_2d.fr_ArN",
    "rdkit_2d.fr_Ar_COO",
    "rdkit_2d.fr_Ar_N",
    "rdkit_2d.fr_Ar_NH",
    "rdkit_2d.fr_Ar_OH",
    "rdkit_2d.fr_COO",
    "rdkit_2d.fr_COO2",
    "rdkit_2d.fr_C_O",
    "rdkit_2d.fr_C_O_noCOO",
    "rdkit_2d.fr_C_S",
    "rdkit_2d.fr_HOCCN",
    "rdkit_2d.fr_Imine",
    "rdkit_2d.fr_NH0",
    "rdkit_2d.fr_NH1",
    "rdkit_2d.fr_NH2",
    "rdkit_2d.fr_N_O",
    "rdkit_2d.fr_Ndealkylation1",
    "rdkit_2d.fr_Ndealkylation2",
    "rdkit_2d.fr_Nhpyrrole",
    "rdkit_2d.fr_SH",
    "rdkit_2d.fr_aldehyde",
    "rdkit_2d.fr_alkyl_carbamate",
    "rdkit_2d.fr_alkyl_halide",
    "rdkit_2d.fr_allylic_oxid",
    "rdkit_2d.fr_amide",
    "rdkit_2d.fr_amidine",
    "rdkit_2d.fr_aniline",
    "rdkit_2d.fr_aryl_methyl",
    "rdkit_2d.fr_azide",
    "rdkit_2d.fr_azo",
    "rdkit_2d.fr_barbitur",
    "rdkit_2d.fr_benzene",
    "rdkit_2d.fr_benzodiazepine",
    "rdkit_2d.fr_bicyclic",
    "rdkit_2d.fr_diazo",
    "rdkit_2d.fr_dihydropyridine",
    "rdkit_2d.fr_epoxide",
    "rdkit_2d.fr_ester",
    "rdkit_2d.fr_ether",
    "rdkit_2d.fr_furan",
    "rdkit_2d.fr_guanido",
    "rdkit_2d.fr_halogen",
    "rdkit_2d.fr_hdrzine",
    "rdkit_2d.fr_hdrzone",
    "rdkit_2d.fr_imidazole",
    "rdkit_2d.fr_imide",
    "rdkit_2d.fr_isocyan",
    "rdkit_2d.fr_isothiocyan",
    "rdkit_2d.fr_ketone",
    "rdkit_2d.fr_ketone_Topliss",
    "rdkit_2d.fr_lactam",
    "rdkit_2d.fr_lactone",
    "rdkit_2d.fr_methoxy",
    "rdkit_2d.fr_morpholine",
    "rdkit_2d.fr_nitrile",
    "rdkit_2d.fr_nitro",
    "rdkit_2d.fr_nitro_arom",
    "rdkit_2d.fr_nitro_arom_nonortho",
    "rdkit_2d.fr_nitroso",
    "rdkit_2d.fr_oxazole",
    "rdkit_2d.fr_oxime",
    "rdkit_2d.fr_para_hydroxylation",
    "rdkit_2d.fr_phenol",
    "rdkit_2d.fr_phenol_noOrthoHbond",
    "rdkit_2d.fr_phos_acid",
    "rdkit_2d.fr_phos_ester",
    "rdkit_2d.fr_piperdine",
    "rdkit_2d.fr_piperzine",
    "rdkit_2d.fr_priamide",
    "rdkit_2d.fr_prisulfonamd",
    "rdkit_2d.fr_pyridine",
    "rdkit_2d.fr_quatN",
    "rdkit_2d.fr_sulfide",
    "rdkit_2d.fr_sulfonamd",
    "rdkit_2d.fr_sulfone",
    "rdkit_2d.fr_term_acetylene",
    "rdkit_2d.fr_tetrazole",
    "rdkit_2d.fr_thiazole",
    "rdkit_2d.fr_thiocyan",
    "rdkit_2d.fr_thiophene",
    "rdkit_2d.fr_unbrch_alkane",
    "rdkit_2d.fr_urea",)


def select_contract_columns(
    matrix: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Return ``(matrix, names)`` restricted to the published descriptor contract.

    Columns are selected by name, so a newer RDKit that appends extra
    descriptors still yields the 210-column block the models were fitted on.
    """
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D descriptor matrix, got shape {array.shape}")
    positions: dict[str, int] = {}
    for index, name in enumerate(names):
        positions.setdefault(str(name), index)
    missing = [name for name in PAPER_DESCRIPTOR_NAMES if name not in positions]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"installed RDKit does not expose {len(missing)} contract descriptor(s): {preview}"
        )
    columns = [positions[name] for name in PAPER_DESCRIPTOR_NAMES]
    return array[:, columns], list(PAPER_DESCRIPTOR_NAMES)
