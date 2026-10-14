"""Synthetic data with known rules, so we can check exactly what the model learned.

Each rule needs a different part of the architecture:

    next_text     = (prev_text + prev_user_q0)     mod V   big Transformer: past frames + user stream
    next_moshi_q0 = (next_text + prev_user_q1)     mod V   small Transformer: this frame's text
    next_moshi_q1 = (next_moshi_q0 + prev_moshi_q1) mod V  small Transformer: this frame's q0

User audio is random (it's "the microphone"). Frame 0 is random.
"""

import torch


def make_batch(B, T, vocab, generator=None):
    # returns codes [B, 5, T]: 0 text, 1 moshi q0, 2 moshi q1, 3 user q0, 4 user q1
    codes = torch.randint(0, vocab, (B, 5, T), generator=generator)
    for t in range(1, T):
        text = (codes[:, 0, t - 1] + codes[:, 3, t - 1]) % vocab
        q0 = (text + codes[:, 4, t - 1]) % vocab
        q1 = (q0 + codes[:, 2, t - 1]) % vocab
        codes[:, 0, t] = text
        codes[:, 1, t] = q0
        codes[:, 2, t] = q1
    return codes
