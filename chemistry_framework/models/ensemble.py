from __future__ import annotations
from typing import Any, Sequence
import torch
from torch import Tensor, nn

class EnsembleModel(nn.Module):
    def __init__(self, models: Sequence[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("EnsembleModel requires at least one model")
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, inputs: Any = None, **kwargs: Any) -> dict[str, Tensor]:
        predictions: list[Tensor] = []
        variances: list[Tensor] = []
        for model in self.models:
            out = model(inputs, **kwargs)
            if isinstance(out, Tensor):
                pred = out
            elif isinstance(out, dict):
                pred = out.get("prediction", out.get("mean"))
                if "variance" in out:
                    variances.append(out["variance"])
            else:
                raise TypeError(f"model output must be Tensor or dict, got {type(out).__name__}")
            predictions.append(pred)
        stacked_preds = torch.stack(predictions, dim=0)
        mean_pred = stacked_preds.mean(dim=0)
        epistemic_var = stacked_preds.var(dim=0, unbiased=False)
        res: dict[str, Tensor] = {
            "prediction": mean_pred,
            "mean": mean_pred,
            "epistemic_variance": epistemic_var,
            "variance": epistemic_var,
            "std": torch.sqrt(epistemic_var.clamp(min=1e-12)),
        }
        if len(variances) == self.num_models:
            stacked_aleatoric = torch.stack(variances, dim=0)
            mean_aleatoric = stacked_aleatoric.mean(dim=0)
            total_var = epistemic_var + mean_aleatoric
            res["aleatoric_variance"] = mean_aleatoric
            res["total_variance"] = total_var
            res["variance"] = total_var
            res["std"] = torch.sqrt(total_var.clamp(min=1e-12))
        return res

__all__ = ["EnsembleModel"]
