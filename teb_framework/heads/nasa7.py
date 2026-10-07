"""Analytic NASA-7 thermochemistry output block."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class NASA7OutputBlock(nn.Module):
    """Convert Cp coefficients and reference H/S into continuous NASA-7 curves.

    Inputs use SI units: H in J/mol, S and Cp in J/(mol K), and temperatures in K.
    """

    gas_constant = 8.314462618

    def __init__(
        self,
        t_ref: float = 298.15,
        t_mid: float = 1000.0,
        temperatures: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        if t_ref <= 0 or t_mid <= t_ref:
            raise ValueError("temperatures must satisfy 0 < t_ref < t_mid")
        self.t_ref = float(t_ref)
        self.t_mid = float(t_mid)
        temperature_grid = (
            torch.as_tensor(temperatures, dtype=torch.float64)
            if temperatures is not None
            else None
        )
        if temperature_grid is not None and (
            temperature_grid.ndim != 1
            or not torch.isfinite(temperature_grid).all()
            or (temperature_grid <= 0).any()
        ):
            raise ValueError("temperatures must be a finite, positive 1-D grid")
        self.register_buffer("temperature_grid", temperature_grid)

    def forward(
        self,
        a1_5_low: Tensor,
        a1_5_high: Tensor | None = None,
        h_ref: Tensor | None = None,
        s_ref: Tensor | None = None,
        temperatures: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if a1_5_high is None:
            if a1_5_low.ndim != 2 or a1_5_low.shape[-1] != 12:
                raise ValueError(
                    "packed NASA input must have shape (batch, 12): "
                    "5 low coefficients, 5 high coefficients, H_ref, S_ref"
                )
            if self.temperature_grid is None and temperatures is None:
                raise ValueError(
                    "configure a temperature grid on NASA7OutputBlock or pass it explicitly"
                )
            packed = a1_5_low
            a1_5_low, a1_5_high = packed[:, :5], packed[:, 5:10]
            h_ref, s_ref = packed[:, 10], packed[:, 11]
        if h_ref is None or s_ref is None:
            raise ValueError("h_ref and s_ref are required")
        if temperatures is None:
            temperatures = self.temperature_grid
        if temperatures is None:
            raise ValueError("a temperature grid is required")
        if a1_5_low.ndim != 2 or a1_5_low.shape[-1] != 5:
            raise ValueError("a1_5_low must have shape (batch, 5)")
        if a1_5_high is None:
            raise ValueError("a1_5_high is required")
        if a1_5_high.shape != a1_5_low.shape:
            raise ValueError("a1_5_high must have the same shape as a1_5_low")
        batch_size = a1_5_low.shape[0]
        if h_ref.shape != (batch_size,) or s_ref.shape != (batch_size,):
            raise ValueError("h_ref and s_ref must each have shape (batch,)")
        if temperatures.ndim != 1 or not torch.isfinite(temperatures).all():
            raise ValueError("temperatures must be a finite 1-D tensor")
        if (temperatures <= 0).any():
            raise ValueError("temperatures must be positive")

        original_dtype = a1_5_low.dtype
        device = a1_5_low.device
        low = a1_5_low.double()
        high = a1_5_high.double()
        h_ref = h_ref.double()
        s_ref = s_ref.double()
        temperatures = temperatures.to(device=device, dtype=torch.float64)
        gas_constant = self.gas_constant

        a1_l, a2_l, a3_l, a4_l, a5_l = low.unbind(dim=-1)
        t_ref = self.t_ref
        t_mid = self.t_mid
        h_poly_ref = (
            a1_l + a2_l * t_ref / 2 + a3_l * t_ref**2 / 3
            + a4_l * t_ref**3 / 4 + a5_l * t_ref**4 / 5
        )
        a6_l = t_ref * (h_ref / (gas_constant * t_ref) - h_poly_ref)
        s_poly_ref = (
            a1_l * math.log(t_ref) + a2_l * t_ref + a3_l * t_ref**2 / 2
            + a4_l * t_ref**3 / 3 + a5_l * t_ref**4 / 4
        )
        a7_l = s_ref / gas_constant - s_poly_ref

        h_poly_mid_low = (
            a1_l + a2_l * t_mid / 2 + a3_l * t_mid**2 / 3
            + a4_l * t_mid**3 / 4 + a5_l * t_mid**4 / 5 + a6_l / t_mid
        )
        s_poly_mid_low = (
            a1_l * math.log(t_mid) + a2_l * t_mid + a3_l * t_mid**2 / 2
            + a4_l * t_mid**3 / 3 + a5_l * t_mid**4 / 4 + a7_l
        )
        cp_mid_low = (
            a1_l + a2_l * t_mid + a3_l * t_mid**2
            + a4_l * t_mid**3 + a5_l * t_mid**4
        )

        _, a2_h, a3_h, a4_h, a5_h = high.unbind(dim=-1)
        a1_h = cp_mid_low - (
            a2_h * t_mid + a3_h * t_mid**2 + a4_h * t_mid**3 + a5_h * t_mid**4
        )
        h_poly_mid_high = (
            a1_h + a2_h * t_mid / 2 + a3_h * t_mid**2 / 3
            + a4_h * t_mid**3 / 4 + a5_h * t_mid**4 / 5
        )
        a6_h = t_mid * (h_poly_mid_low - h_poly_mid_high)
        s_poly_mid_high = (
            a1_h * math.log(t_mid) + a2_h * t_mid + a3_h * t_mid**2 / 2
            + a4_h * t_mid**3 / 3 + a5_h * t_mid**4 / 4
        )
        a7_h = s_poly_mid_low - s_poly_mid_high

        cp_grid = torch.empty((batch_size, len(temperatures)), device=device, dtype=torch.float64)
        h_grid = torch.empty_like(cp_grid)
        s_grid = torch.empty_like(cp_grid)
        low_mask = temperatures <= t_mid

        def evaluate(mask: Tensor, coefficients: tuple[Tensor, ...]) -> None:
            if not mask.any():
                return
            a1, a2, a3, a4, a5, a6, a7 = coefficients
            t = temperatures[mask].unsqueeze(0)
            a1, a2, a3, a4, a5, a6, a7 = (
                value.unsqueeze(-1) for value in (a1, a2, a3, a4, a5, a6, a7)
            )
            cp_grid[:, mask] = gas_constant * (
                a1 + a2 * t + a3 * t**2 + a4 * t**3 + a5 * t**4
            )
            h_grid[:, mask] = gas_constant * t * (
                a1 + a2 * t / 2 + a3 * t**2 / 3 + a4 * t**3 / 4
                + a5 * t**4 / 5 + a6 / t
            )
            s_grid[:, mask] = gas_constant * (
                a1 * torch.log(t) + a2 * t + a3 * t**2 / 2 + a4 * t**3 / 3
                + a5 * t**4 / 4 + a7
            )

        evaluate(low_mask, (a1_l, a2_l, a3_l, a4_l, a5_l, a6_l, a7_l))
        evaluate(~low_mask, (a1_h, a2_h, a3_h, a4_h, a5_h, a6_h, a7_h))
        coeffs_low = torch.stack((a1_l, a2_l, a3_l, a4_l, a5_l, a6_l, a7_l), dim=-1)
        coeffs_high = torch.stack((a1_h, a2_h, a3_h, a4_h, a5_h, a6_h, a7_h), dim=-1)
        return {
            "cp": cp_grid.to(original_dtype),
            "H": h_grid.to(original_dtype),
            "S": s_grid.to(original_dtype),
            "coeffs_low": coeffs_low.to(original_dtype),
            "coeffs_high": coeffs_high.to(original_dtype),
            "h_ref": h_ref.to(original_dtype),
            "s_ref": s_ref.to(original_dtype),
        }
