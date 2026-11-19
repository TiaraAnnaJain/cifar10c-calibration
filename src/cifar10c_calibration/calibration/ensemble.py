"""Deep ensembles: combine several models by averaging their probabilities."""

import math

import torch


def ensemble_logits(member_logits: list[torch.Tensor]) -> torch.Tensor:
    """Log of the members' average probabilities, usable anywhere logits are."""
    stacked = torch.stack(member_logits)
    log_probs = stacked.log_softmax(dim=2)
    log_mean_prob = torch.logsumexp(log_probs, dim=0) - math.log(len(member_logits))
    return log_mean_prob
