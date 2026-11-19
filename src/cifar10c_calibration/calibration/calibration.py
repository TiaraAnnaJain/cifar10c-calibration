"""Temperature scaling, the number that softens or sharpens a model's confidence."""

import torch
import torch.nn.functional as F


class TemperatureScaler:
    """Divides logits by a single temperature, fitted on a validation set."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Find the temperature that minimises the NLL of the logits."""
        logits = logits.detach()
        log_temperature = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature], lr=0.1, max_iter=200, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / log_temperature.exp(), labels)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        self.temperature = log_temperature.exp().item()

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits divided by the fitted temperature."""
        scaled_logits = logits / self.temperature
        return scaled_logits
