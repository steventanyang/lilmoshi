import torch

from model import LilMoshi, LilMoshiConfig


def random_codes(cfg, B=2, T=10):
    # [B, K, T]: text IDs in row 0, audio IDs in rows 1..K-1
    text = torch.randint(0, cfg.text_vocab_size, (B, 1, T))
    audio = torch.randint(0, cfg.audio_vocab_size, (B, cfg.num_streams - 1, T))
    return torch.cat([text, audio], dim=1)


def make_model():
    torch.manual_seed(0)
    return LilMoshi(LilMoshiConfig())


def test_embed_shape():
    # output is [B, T, temporal_dim], not [B, K*T, temporal_dim]
    m = make_model()
    codes = random_codes(m.cfg)
    assert m.embed(codes).shape == (2, 10, m.cfg.temporal_dim)


def test_every_stream_matters():
    # changing any one stream changes the result
    m = make_model()
    codes = random_codes(m.cfg)
    base = m.embed(codes)
    for k in range(m.cfg.num_streams):
        changed = codes.clone()
        changed[:, k] = (changed[:, k] + 1) % m.cfg.audio_vocab_size
        assert not torch.allclose(m.embed(changed), base), f"stream {k} ignored"


def test_stream_identity():
    # swapping moshi q0 and user q0 changes the result (separate tables)
    m = make_model()
    codes = random_codes(m.cfg)
    swapped = codes.clone()
    swapped[:, 1], swapped[:, 3] = codes[:, 3], codes[:, 1]
    assert not torch.allclose(m.embed(swapped), m.embed(codes))


# --- Stage 3: temporal Transformer + text head ---


def test_temporal_shapes():
    m = make_model()
    codes = random_codes(m.cfg)
    h, text_logits = m.temporal_forward(codes)
    assert h.shape == (2, 10, m.cfg.temporal_dim)
    assert text_logits.shape == (2, 10, m.cfg.text_vocab_size)


def test_temporal_causal():
    # changing frames after t must not change h or text_logits at or before t
    m = make_model().eval()
    codes = random_codes(m.cfg)
    t = 4
    changed = codes.clone()
    changed[:, :, t + 1:] = random_codes(m.cfg)[:, :, t + 1:]

    h1, logits1 = m.temporal_forward(codes)
    h2, logits2 = m.temporal_forward(changed)
    assert torch.allclose(h1[:, : t + 1], h2[:, : t + 1], atol=1e-6)
    assert torch.allclose(logits1[:, : t + 1], logits2[:, : t + 1], atol=1e-6)


def test_gradients_reach_every_stream_embedding():
    m = make_model()
    codes = random_codes(m.cfg)
    _, text_logits = m.temporal_forward(codes)
    text_logits.sum().backward()
    assert m.text_emb.weight.grad is not None
    for k, emb in enumerate(m.audio_embs, start=1):
        assert emb.weight.grad is not None, f"no gradient for stream {k}"
        assert emb.weight.grad.abs().sum() > 0, f"zero gradient for stream {k}"


# --- Stage 4: depth Transformer (teacher-forced) ---


def depth_setup():
    m = make_model().eval()
    h = torch.randn(2, 10, m.cfg.temporal_dim)  # stand-in for the big Transformer's output
    target = random_codes(m.cfg)                 # the frames being predicted
    return m, h, target


def test_depth_shape():
    m, h, target = depth_setup()
    audio_logits = m.depth_forward_training(h, target)
    Q = m.cfg.num_moshi_audio_codebooks
    assert audio_logits.shape == (2, Q, 10, m.cfg.audio_vocab_size)


def test_text_affects_q0():
    m, h, target = depth_setup()
    changed = target.clone()
    changed[:, 0] = (changed[:, 0] + 1) % m.cfg.text_vocab_size
    a = m.depth_forward_training(h, target)
    b = m.depth_forward_training(h, changed)
    assert not torch.allclose(a[:, 0], b[:, 0])


def test_q0_affects_q1_but_not_q0():
    m, h, target = depth_setup()
    changed = target.clone()
    changed[:, 1] = (changed[:, 1] + 1) % m.cfg.audio_vocab_size  # moshi q0
    a = m.depth_forward_training(h, target)
    b = m.depth_forward_training(h, changed)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-6)  # q0 can't see its own answer
    assert not torch.allclose(a[:, 1], b[:, 1])          # q1 does see q0


def test_q1_does_not_affect_q0():
    m, h, target = depth_setup()
    changed = target.clone()
    changed[:, 2] = (changed[:, 2] + 1) % m.cfg.audio_vocab_size  # moshi q1
    a = m.depth_forward_training(h, target)
    b = m.depth_forward_training(h, changed)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-6)


def test_frames_are_independent():
    # changing frame 3's tokens only changes frame 3's audio logits
    m, h, target = depth_setup()
    changed = target.clone()
    changed[:, :, 3] = random_codes(m.cfg)[:, :, 3]
    a = m.depth_forward_training(h, target)
    b = m.depth_forward_training(h, changed)
    other = [t for t in range(10) if t != 3]
    assert torch.allclose(a[:, :, other], b[:, :, other], atol=1e-6)
    assert not torch.allclose(a[:, :, 3], b[:, :, 3])


# --- Stage 5: full training forward + loss ---


def test_loss_is_finite_scalar():
    m = make_model()
    out = m(random_codes(m.cfg))
    assert out.loss.dim() == 0
    assert torch.isfinite(out.loss)
    assert out.text_logits.shape == (2, 9, m.cfg.text_vocab_size)  # T-1 positions
    assert out.audio_logits.shape == (2, m.cfg.num_moshi_audio_codebooks, 9, m.cfg.audio_vocab_size)


def test_every_parameter_gets_a_gradient():
    m = make_model()
    m(random_codes(m.cfg)).loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} is disconnected from the loss"
        assert torch.isfinite(p.grad).all(), name


def test_shift_alignment():
    # the last frame is only ever a target, never an input:
    # changing it must not change any logits the temporal model produces
    m = make_model().eval()
    codes = random_codes(m.cfg)
    changed = codes.clone()
    changed[:, :, -1] = random_codes(m.cfg)[:, :, -1]
    assert torch.allclose(m(codes).text_logits, m(changed).text_logits, atol=1e-6)


def test_user_audio_is_never_a_target():
    # the last frame is never an input, so its user tokens could only affect the
    # loss by being graded; they aren't, so changing them leaves the loss unchanged
    m = make_model().eval()
    codes = random_codes(m.cfg)
    changed = codes.clone()
    u = m.cfg.user_audio_offset
    changed[:, u:, -1] = (changed[:, u:, -1] + 1) % m.cfg.audio_vocab_size
    assert torch.allclose(m(codes).loss, m(changed).loss)

    # sanity: changing a graded stream (moshi q1) in the last frame does change it
    changed = codes.clone()
    changed[:, 2, -1] = (changed[:, 2, -1] + 1) % m.cfg.audio_vocab_size
    assert not torch.allclose(m(codes).loss, m(changed).loss)


def test_overfits_tiny_batch():
    m = make_model()
    codes = random_codes(m.cfg, B=2, T=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    first = m(codes).loss.item()
    for _ in range(300):
        opt.zero_grad()
        loss = m(codes).loss
        loss.backward()
        opt.step()

    assert loss.item() < 0.1 * first, f"loss only went from {first:.2f} to {loss.item():.2f}"
