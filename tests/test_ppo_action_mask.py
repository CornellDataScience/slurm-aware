import numpy as np
import torch
from torch.distributions import Categorical

from experiments.ppo.train import available_action_mask, mask_logits


def test_mask_allows_only_valid_actions():
    class FakeEnv:
        n_actions = 11

        def fitting_slots(self):
            return [2, 5]

    mask = available_action_mask(FakeEnv())

    # NOP (0) and fitting job slots (2, 5) are valid.
    assert np.flatnonzero(mask).tolist() == [0, 2, 5]


def test_invalid_actions_have_zero_probability():
    logits = torch.tensor([[1.0, 10.0, 2.0, 8.0]])
    mask = torch.tensor([[True, False, True, False]])

    masked = mask_logits(logits, mask)
    distribution = Categorical(logits=masked)

    assert distribution.probs[0, 1].item() == 0
    assert distribution.probs[0, 3].item() == 0

    for _ in range(100):
        assert distribution.sample().item() in (0, 2)
