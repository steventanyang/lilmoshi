import torch

from model import LilMoshi, LilMoshiConfig
from tests.test_model import random_codes


def setup(B=2, T=6):
    torch.manual_seed(0)
    m = LilMoshi(LilMoshiConfig()).eval()
    history = random_codes(m.cfg, B=B, T=T)
    user = torch.randint(0, m.cfg.audio_vocab_size, (B, m.cfg.num_user_audio_codebooks))
    return m, history, user


def test_one_frame_shape():
    m, history, user = setup()
    frame = m.generate_frame(history, user)
    assert frame.shape == (2, m.cfg.num_streams, 1)


def test_ids_in_vocab():
    m, history, user = setup()
    frame = m.generate_frame(history, user)[:, :, 0]
    assert ((frame[:, 0] >= 0) & (frame[:, 0] < m.cfg.text_vocab_size)).all()
    assert ((frame[:, 1:] >= 0) & (frame[:, 1:] < m.cfg.audio_vocab_size)).all()


def test_user_audio_copied_exactly():
    m, history, user = setup()
    frame = m.generate_frame(history, user)[:, :, 0]
    assert torch.equal(frame[:, m.cfg.user_audio_offset:], user)


def test_greedy_is_deterministic():
    m, history, user = setup()
    assert torch.equal(m.generate_frame(history, user), m.generate_frame(history, user))


def test_generation_matches_training():
    # feed [history + generated frame] through the training path: its greedy
    # picks at the last position must be exactly the tokens generation chose
    m, history, user = setup()
    frame = m.generate_frame(history, user)
    out = m(torch.cat([history, frame], dim=2))

    assert torch.equal(out.text_logits[:, -1].argmax(-1), frame[:, 0, 0])
    for q in range(m.cfg.num_moshi_audio_codebooks):
        picked = out.audio_logits[:, q, -1].argmax(-1)
        assert torch.equal(picked, frame[:, m.cfg.moshi_audio_offset + q, 0]), f"q{q}"


def test_generate_many_frames():
    m, history, user = setup()
    user_seq = torch.randint(0, m.cfg.audio_vocab_size, (2, m.cfg.num_user_audio_codebooks, 5))
    out = m.generate(history, user_seq)
    assert out.shape == (2, m.cfg.num_streams, history.shape[2] + 5)
    assert torch.equal(out[:, :, : history.shape[2]], history)          # past untouched
    assert torch.equal(out[:, m.cfg.user_audio_offset:, history.shape[2]:], user_seq)
