import json
import os
import time

import torch

from model import LilMoshi, LilMoshiConfig
from synthetic_data import make_batch

VOCAB = 16
STEPS = 600
LOG_EVERY = 10
BATCH = 64
SEQ_LEN = 32
STREAMS = ["text", "moshi q0", "moshi q1"]


@torch.no_grad()
def accuracy(model, codes):
    out = model(codes)
    target = codes[:, :, 1:]
    accs = [(out.text_logits.argmax(-1) == target[:, 0]).float().mean().item()]
    for q in range(model.cfg.num_moshi_audio_codebooks):
        picked = out.audio_logits[:, q].argmax(-1)
        accs.append((picked == target[:, 1 + q]).float().mean().item())
    return accs


@torch.no_grad()
def rollout_accuracy(model, codes, prefix=8):
    user = codes[:, 3:, prefix:]
    generated = model.generate(codes[:, :, :prefix], user)
    gen, true = generated[:, :3, prefix:], codes[:, :3, prefix:]
    return [(gen[:, k] == true[:, k]).float().mean().item() for k in range(3)]


def fmt(accs):
    return "  ".join(f"{name} {a:6.1%}" for name, a in zip(STREAMS, accs))


def main():
    torch.manual_seed(0)
    cfg = LilMoshiConfig(text_vocab_size=VOCAB, audio_vocab_size=VOCAB)
    model = LilMoshi(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    held_out = make_batch(256, SEQ_LEN, VOCAB, generator=torch.Generator().manual_seed(123))
    print(f"chance = 1/{VOCAB} = {1 / VOCAB:.1%}")

    history = []
    start = time.time()
    for step in range(STEPS + 1):
        model.train()
        out = model(make_batch(BATCH, SEQ_LEN, VOCAB))
        opt.zero_grad()
        out.loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            model.eval()
            accs = accuracy(model, held_out)
            history.append({
                "step": step,
                "text_loss": out.text_loss.item(),
                "audio_loss": out.audio_loss.item(),
                "acc": dict(zip(STREAMS, accs)),
            })
            if step % 100 == 0:
                print(f"step {step:4d}  loss {out.loss.item():.3f}  held-out: {fmt(accs)}"
                      f"  ({time.time() - start:.0f}s)")

    model.eval()
    results = {
        "teacher-forced": accuracy(model, held_out),
        "free generation": rollout_accuracy(model, held_out),
    }
    print()
    for name, accs in results.items():
        print(f"{name:20s}: {fmt(accs)}")

    os.makedirs("runs", exist_ok=True)
    with open("runs/metrics.json", "w") as f:
        json.dump({
            "vocab": VOCAB,
            "streams": STREAMS,
            "history": history,
            "results": {k: dict(zip(STREAMS, v)) for k, v in results.items()},
        }, f, indent=1)
    print("\nwrote runs/metrics.json")


if __name__ == "__main__":
    main()
