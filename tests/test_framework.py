from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_framework import (
    CallablePhysicsPrior,
    CompositePrior,
    FeatureRegistry,
    FeatureSpec,
    LinearAdditivePrior,
    MLPEncoder,
    MLPResidual,
    MoleculeFeatureBlock,
    NASA7OutputBlock,
    ResidualPropertyModel,
    ScalarOutputHead,
    ZeroPrior,
    diagnose_features,
    molecular_feature_registry,
)


def test_registry_builds_named_smiles_features() -> None:
    registry = molecular_feature_registry()
    registry.add_smarts("n_ester", "C(=O)O")
    matrix = MoleculeFeatureBlock(registry, ["n_C", "n_O", "n_ester"]).transform(
        ["CCO", "CC(=O)OC"]
    )
    assert matrix.names == ("n_C", "n_O", "n_ester")
    np.testing.assert_array_equal(matrix.values, [[2, 1, 0], [3, 2, 1]])


def test_registry_rejects_duplicate_and_invalid_features() -> None:
    registry = FeatureRegistry()
    registry.register(FeatureSpec("x", lambda _: 1.0, domain="custom"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FeatureSpec("x", lambda _: 2.0, domain="custom"))
    with pytest.raises(ValueError, match="invalid SMARTS"):
        registry.add_smarts("bad", "[")
    with pytest.raises(ValueError, match="invalid molecule"):
        MoleculeFeatureBlock(registry, ["x"]).transform(["not a smiles"])


def test_prior_and_residual_are_composed_before_scalar_head() -> None:
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    encoder = torch.nn.Identity()
    residual = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        residual.weight.copy_(torch.tensor([[0.5, -0.5]]))
    model = ResidualPropertyModel(
        encoder,
        LinearAdditivePrior([2.0, 1.0], intercept=3.0),
        residual,
        ScalarOutputHead(),
    )
    output = model(features)
    torch.testing.assert_close(output["baseline"], torch.tensor([[7.0], [13.0]]))
    torch.testing.assert_close(output["residual"], torch.tensor([[-0.5], [-0.5]]))
    torch.testing.assert_close(output["prediction"], torch.tensor([6.5, 12.5]))


def test_composite_physics_prior_adds_independent_terms() -> None:
    prior = CompositePrior(
        [
            LinearAdditivePrior([2.0], intercept=1.0),
            CallablePhysicsPrior(lambda x: x.square(), output_dim=1),
        ]
    )
    actual = prior(torch.tensor([[3.0], [4.0]]))
    torch.testing.assert_close(actual, torch.tensor([[16.0], [25.0]]))


def test_descriptor_model_runs_end_to_end() -> None:
    features = torch.randn(4, 3)
    model = ResidualPropertyModel(
        MLPEncoder(3, 8),
        LinearAdditivePrior([1.0, 0.0, -1.0]),
        MLPResidual(8),
        ScalarOutputHead(),
    )
    prediction = model(features)["prediction"]
    assert prediction.shape == (4,)


def test_diagnostics_report_extrapolation_and_collinearity() -> None:
    train = np.array([[0, 0, 1], [1, 2, 1], [2, 4, 1], [3, 6, 1]], dtype=float)
    test = np.array([[4, 8, 2], [1, 2, 1]], dtype=float)
    report = diagnose_features(train, test, ["x", "twice_x", "constant"])
    by_name = {item.name: item for item in report.features}
    assert by_name["x"].out_of_range_count == 1
    assert by_name["twice_x"].out_of_range_count == 1
    assert by_name["constant"].constant_in_train
    assert by_name["constant"].out_of_range_count == 1
    assert report.collinear_pairs == (("x", "twice_x", pytest.approx(1.0)),)
    assert "outside the training range" in report.format_report()


def test_nasa7_preserves_reference_values_and_midpoint_continuity() -> None:
    block = NASA7OutputBlock(t_ref=300.0, t_mid=1000.0)
    low = torch.tensor([[3.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    high = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    h_ref = torch.tensor([12000.0], dtype=torch.float64)
    s_ref = torch.tensor([220.0], dtype=torch.float64)
    result = block(
        low,
        high,
        h_ref,
        s_ref,
        torch.tensor([300.0, 1000.0], dtype=torch.float64),
    )
    torch.testing.assert_close(result["H"][:, 0], h_ref, atol=1e-9, rtol=1e-12)
    torch.testing.assert_close(result["S"][:, 0], s_ref, atol=1e-9, rtol=1e-12)
    a1, a2, a3, a4, a5, a6, a7 = result["coeffs_high"][0].unbind()
    midpoint = 1000.0
    gas_constant = NASA7OutputBlock.gas_constant
    cp_high = gas_constant * (
        a1 + a2 * midpoint + a3 * midpoint**2 + a4 * midpoint**3 + a5 * midpoint**4
    )
    h_high = gas_constant * midpoint * (
        a1 + a2 * midpoint / 2 + a3 * midpoint**2 / 3 + a4 * midpoint**3 / 4
        + a5 * midpoint**4 / 5 + a6 / midpoint
    )
    s_high = gas_constant * (
        a1 * torch.log(torch.tensor(midpoint, dtype=torch.float64))
        + a2 * midpoint + a3 * midpoint**2 / 2 + a4 * midpoint**3 / 3
        + a5 * midpoint**4 / 4 + a7
    )
    for key, high_value in (("cp", cp_high), ("H", h_high), ("S", s_high)):
        torch.testing.assert_close(
            result[key][:, 1], high_value.reshape(1), atol=1e-8, rtol=1e-10
        )
    assert result["coeffs_low"].shape == (1, 7)
    assert result["coeffs_high"].shape == (1, 7)


def test_nasa7_output_block_composes_with_property_model() -> None:
    temperatures = torch.tensor([300.0, 1000.0, 1200.0], dtype=torch.float64)
    output = NASA7OutputBlock(t_ref=300.0, t_mid=1000.0, temperatures=temperatures)
    residual = torch.nn.Linear(2, 12, bias=True)
    with torch.no_grad():
        residual.weight.zero_()
        residual.bias.zero_()
        residual.bias[0] = 3.5
        residual.bias[10] = 12000.0
        residual.bias[11] = 220.0
    model = ResidualPropertyModel(
        encoder=torch.nn.Identity(),
        prior=ZeroPrior(output_dim=12),
        residual=residual,
        output_head=output,
    )
    result = model(torch.ones(1, 2))
    assert result["cp"].shape == (1, 3)
    torch.testing.assert_close(result["H"][:, 0], torch.tensor([12000.0]))
    torch.testing.assert_close(result["S"][:, 0], torch.tensor([220.0]))
