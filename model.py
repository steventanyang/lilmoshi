from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LilMoshiConfig:
    text_vocab_size: int = 128
    audio_vocab_size: int = 64
    num_moshi_audio_codebooks: int = 2
    num_user_audio_codebooks: int = 2

    # Temporal Transformer (across time T)
    temporal_dim: int = 128
    temporal_heads: int = 4
    temporal_layers: int = 3
    temporal_ffn_dim: int = 512

    # Depth Transformer (across codebooks Q)
    depth_dim: int = 96
    depth_heads: int = 4
    depth_layers: int = 2
    depth_ffn_dim: int = 384

    max_seq_len: int = 128
    dropout: float = 0.0

    @property
    def num_streams(self) -> int:
        return 1 + self.num_moshi_audio_codebooks + self.num_user_audio_codebooks

    @property
    def moshi_audio_offset(self) -> int:
        return 1

    @property
    def user_audio_offset(self) -> int:
        return 1 + self.num_moshi_audio_codebooks


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) 

        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.dropout(scores.softmax(dim=-1))
        y = weights @ v

        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ffn_dim, dropout)

    def forward(self, x):
        # x: [B, T, D] -> [B, T, D]
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, dim, heads, layers, ffn_dim, max_seq_len, dropout=0.0):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ffn_dim, dropout) for _ in range(layers)]
        )

    def forward(self, x):
        # x: [B, T, D] -> [B, T, D], causal over T
        B, T, D = x.shape

        positions = torch.arange(T, device=x.device)
        x = x + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class LilMoshiOutput:
    text_logits: torch.Tensor   # [B, T-1, Vtext]
    audio_logits: torch.Tensor  # [B, Q, T-1, Vaudio]
    text_loss: torch.Tensor     # scalar
    audio_loss: torch.Tensor    # scalar, mean over the Q codebooks
    loss: torch.Tensor          # text_loss + audio_loss


class LilMoshi(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.text_emb = nn.Embedding(cfg.text_vocab_size, cfg.temporal_dim)

        self.audio_embs = nn.ModuleList(
            [nn.Embedding(cfg.audio_vocab_size, cfg.temporal_dim)
             for _ in range(cfg.num_streams - 1)]
        )

        self.temporal = CausalTransformer(
            dim=cfg.temporal_dim,
            heads=cfg.temporal_heads,
            layers=cfg.temporal_layers,
            ffn_dim=cfg.temporal_ffn_dim,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        self.temporal_norm = nn.LayerNorm(cfg.temporal_dim)
        self.text_head = nn.Linear(cfg.temporal_dim, cfg.text_vocab_size)

        Q = cfg.num_moshi_audio_codebooks
        self.depth_in = nn.ModuleList(
            [nn.Linear(cfg.temporal_dim, cfg.depth_dim) for _ in range(Q)]
        )

        self.depth_text_emb = nn.Embedding(cfg.text_vocab_size, cfg.depth_dim)
        self.depth_audio_embs = nn.ModuleList(
            [nn.Embedding(cfg.audio_vocab_size, cfg.depth_dim) for _ in range(Q - 1)]
        )

        self.depth = CausalTransformer(
            dim= cfg.depth_dim,
            heads=cfg.depth_heads,
            layers=cfg.depth_layers,
            ffn_dim=cfg.depth_ffn_dim,
            max_seq_len=Q,
            dropout=cfg.dropout
        )

        self.depth_norms = nn.ModuleList([nn.LayerNorm(cfg.depth_dim) for _ in range(Q)])
        self.audio_heads = nn.ModuleList(
            [nn.Linear(cfg.depth_dim, cfg.audio_vocab_size) for _ in range(Q)]
        )

    def embed(self, codes):
        # codes = [B, K, T]
        x = self.text_emb(codes[:, 0]) 
        for k, emb in enumerate(self.audio_embs, start=1):
            x = x + emb(codes[:, k])
        return x

    def temporal_forward(self, codes):
        # codes: [B, K, T] -> h: [B, T, temporal_dim], text_logits: [B, T, text_vocab_size]
        x = self.embed(codes)
        h = self.temporal(x)
        h = self.temporal_norm(h)
        text_logits = self.text_head(h)
        return h, text_logits

    def depth_step_input(self, q, h, prev_token):
        if q == 0:
            token_in = self.depth_text_emb(prev_token)
        else:
            token_in = self.depth_audio_embs[q - 1](prev_token)
        return token_in + self.depth_in[q](h)

    def depth_forward_training(self, h, target):
        # h: [B, T, Dt], target: [B, K, T] -> audio_logits: [B, Q, T, Va]
        B, T, _ = h.shape
        Q = self.cfg.num_moshi_audio_codebooks

        inputs = []
        for q in range(Q):
            # teacher forcing: the previous token comes from the real target frame
            prev = 0 if q == 0 else self.cfg.moshi_audio_offset + q - 1
            inputs.append(self.depth_step_input(q, h, target[:, prev]))

        x = torch.stack(inputs, dim=2)
        x = x.reshape(B * T, Q, -1)
        y = self.depth(x)       

        logits = []
        for q in range(Q):
            out = self.audio_heads[q](self.depth_norms[q](y[:, q])) 
            logits.append(out.view(B, T, -1))                      
        return torch.stack(logits, dim=1)                         

    def forward(self, codes):
        # codes: [B, K, T] -> LilMoshiOutput (logits over T-1 positions + losses)
        Q = self.cfg.num_moshi_audio_codebooks

        history = codes[:, :, :-1]   # [B, K, T-1]
        target = codes[:, :, 1:]     # [B, K, T-1]

        h, text_logits = self.temporal_forward(history)

        audio_logits = self.depth_forward_training(h, target)

        text_target = target[:, 0]
        start = self.cfg.moshi_audio_offset
        audio_target = target[:, start:start + Q]

        text_loss = F.cross_entropy(
            text_logits.reshape(-1, self.cfg.text_vocab_size),
            text_target.reshape(-1),
        )
        audio_loss = torch.stack([
            F.cross_entropy(
                audio_logits[:, q].reshape(-1, self.cfg.audio_vocab_size),
                audio_target[:, q].reshape(-1),
            )
            for q in range(Q)
        ]).mean()

        return LilMoshiOutput(
            text_logits=text_logits,
            audio_logits=audio_logits,
            text_loss=text_loss,
            audio_loss=audio_loss,
            loss=text_loss + audio_loss,
        )
    @torch.no_grad()
    def generate_frame(self, history, next_user_audio):
        # history: [B, K, T], next_user_audio: [B, Qu] -> new frame [B, K, 1]
        # greedy decoding, no cache: reruns both Transformers on their full prefix
        Q = self.cfg.num_moshi_audio_codebooks

        # 1. big Transformer over all past frames; keep only the last position
        h, text_logits = self.temporal_forward(history)
        h = h[:, -1]                              # [B, Dt]
        text = text_logits[:, -1].argmax(-1)      # [B]

        # 2. small Transformer, one codebook at a time, feeding back its own picks
        prev = text
        steps, audio = [], []
        for q in range(Q):
            steps.append(self.depth_step_input(q, h, prev))   # [B, Dd]
            y = self.depth(torch.stack(steps, dim=1))         # [B, q+1, Dd]
            logits = self.audio_heads[q](self.depth_norms[q](y[:, -1]))
            prev = logits.argmax(-1)                          # [B]
            audio.append(prev)

        # 3. assemble: [text, moshi q0..q(Q-1), user q0..] -> [B, K, 1]
        frame = torch.cat([text[:, None], torch.stack(audio, dim=1), next_user_audio], dim=1)
        return frame[:, :, None]

    @torch.no_grad()
    def generate(self, history, user_audio):
        # history: [B, K, T], user_audio: [B, Qu, N] -> [B, K, T + N]
        for n in range(user_audio.shape[2]):
            frame = self.generate_frame(history, user_audio[:, :, n])
            history = torch.cat([history, frame], dim=2)
        return history
