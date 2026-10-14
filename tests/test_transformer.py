import torch

from model import CausalTransformer


def make_transformer():
    torch.manual_seed(0)
    return CausalTransformer(dim=32, heads=4, layers=2, ffn_dim=64, max_seq_len=16)


def test_shape():
    m = make_transformer()
    x = torch.randn(2, 10, 32)
    assert m(x).shape == (2, 10, 32)


def test_causal():
    m = make_transformer().eval()
    x = torch.randn(2, 10, 32)

    # change everything after position t
    t = 4
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])

    y1, y2 = m(x), m(x2)
    # outputs at or before t must not change
    assert torch.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-6)
    # outputs after t should change (sanity check that the test can fail)
    assert not torch.allclose(y1[:, t + 1:], y2[:, t + 1:])


def test_gradients_finite():
    m = make_transformer()
    x = torch.randn(2, 10, 32)
    m(x).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
