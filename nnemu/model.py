"""Bin-by-bin neural-network emulator of DRAGON BGO spectra (PyTorch, CPU).

Inputs per sample
  gammas  (B, 6) emitter-frame energies in MeV, sorted descending, 0 = absent
  context (B, C) vertex z, beta, log p0, acceptance parameters (+ open flag),
                 resolution scale, threshold shift   (see encode_context)
Outputs (log-probabilities per GENERATED event; each head's last entry is
"other": no hit / not transmitted / no deposit)
  e_max (B, 301), e_sum (B, 301), mult (B, N_MULT+1), crystal_max (B, 31),
  per_crystal (B, 30, 151)   [optionally a subset of crystals]

Architecture:
  * Encoder: per-gamma tokens, plus one context token, then a small
    transformer, then a masked pooled embedding h.
  * Decoders for the energy-binned heads: each bin is a query made of
    Fourier features of its energy plus RELATIVE features to every gamma
    (E_bin - E_i, - single/double escape, - total), passed through RBFs.
    The network knows where peaks *can* be, but still predicts every bin
    freely: bin-by-bin, with no imposed shape. Per-crystal queries also
    carry a learned crystal embedding.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .recoil import ACCEPTANCE_PARAMS, OPEN_ACCEPTANCE
from .trainset import C_EDGES, E_EDGES, MAX_GAMMAS, N_CRYSTALS, N_MULT, RANGES

ME = 0.51099895
RBF_SCALES = (0.02, 0.06, 0.2, 0.6)
N_FOURIER = 8


# ------------------------------------------------------------- inputs
def acceptance_norm():
    """(center, halfwidth) per acceptance parameter for input scaling."""
    lo, hi = RANGES["acc_lo"], RANGES["acc_hi"]
    return (lo + hi) / 2, (hi - lo) / 2


def encode_context(z, beta, p0, acceptance, res_scale, thr_shift) -> np.ndarray:
    """Numpy -> (N, C) float32 context features."""
    acceptance = np.atleast_2d(acceptance)
    is_open = np.all(np.isclose(acceptance, OPEN_ACCEPTANCE), axis=1)
    c, hw = acceptance_norm()
    acc = np.where(is_open[:, None], 0.0, np.clip((acceptance - c) / hw, -3, 3))
    return np.column_stack([
        np.asarray(z) / 8.0, (np.asarray(beta) - 0.03) / 0.02, np.log(np.asarray(p0) / 700.0),
        is_open.astype(float), acc, (np.asarray(res_scale) - 1.0) / 0.3, np.asarray(thr_shift) / 0.1,
    ]).astype(np.float32)


N_CONTEXT = 4 + len(ACCEPTANCE_PARAMS) + 2


def fourier(x, n=N_FOURIER, scale=15.0):
    k = torch.arange(n, dtype=x.dtype, device=x.device)
    w = math.pi * (2.0 ** k) / scale
    a = x[..., None] * w
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def relative_features(e_bin, gammas):
    """e_bin (Q,) bin centres; gammas (B, G) (0 = absent).
    Returns (B, Q, F) relative-energy features."""
    mask = (gammas > 0).float()                                    # (B,G)
    tot = gammas.sum(1, keepdim=True)                              # (B,1)
    shifts = torch.tensor([0.0, ME, 2 * ME], dtype=gammas.dtype)
    refs = (gammas[:, :, None] - shifts).reshape(gammas.shape[0], -1)          # (B, G*3)
    rmask = mask.repeat_interleave(3, dim=1)
    refs = torch.cat([refs, tot - shifts[None, :2]], dim=1)                     # + total, total-511
    rmask = torch.cat([rmask, torch.ones(gammas.shape[0], 2)], dim=1)
    d = e_bin[None, None, :] - refs[:, :, None]                                 # (B, R, Q)
    feats = [torch.exp(-(d / s) ** 2) for s in RBF_SCALES] + [torch.tanh(d / 1.0)]
    f = torch.stack(feats, dim=-1) * rmask[:, :, None, None]                    # (B,R,Q,K)
    f = f.permute(0, 2, 1, 3).reshape(gammas.shape[0], e_bin.shape[0], -1)     # (B,Q,R*K)
    return f


N_REL = (MAX_GAMMAS * 3 + 2) * (len(RBF_SCALES) + 1)


# ------------------------------------------------------------- model
class Emulator(nn.Module):
    def __init__(self, d=128, hidden=192, dec_hidden=128, n_layers=2, crystal_emb=16):
        super().__init__()
        self.e_centres = torch.tensor((E_EDGES[:-1] + E_EDGES[1:]) / 2, dtype=torch.float32)
        self.c_centres = torch.tensor((C_EDGES[:-1] + C_EDGES[1:]) / 2, dtype=torch.float32)
        self.tok = nn.Sequential(nn.Linear(2 * N_FOURIER + 2, d), nn.SiLU(), nn.Linear(d, d))
        self.ctx_tok = nn.Sequential(nn.Linear(N_CONTEXT, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.trunk = nn.Sequential(nn.Linear(2 * d + N_CONTEXT, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        q_in = N_REL + 2 * N_FOURIER
        self.dec = nn.ModuleDict()
        for name, extra in [("e_max", 0), ("e_sum", 0), ("per_crystal", crystal_emb)]:
            self.dec[name] = nn.ModuleDict(dict(
                q=nn.Linear(q_in + extra, dec_hidden), h=nn.Linear(hidden, dec_hidden),
                mlp=nn.Sequential(nn.SiLU(), nn.Linear(dec_hidden, dec_hidden), nn.SiLU(),
                                  nn.Linear(dec_hidden, dec_hidden), nn.SiLU(), nn.Linear(dec_hidden, 1))))
        self.other = nn.Linear(hidden, 2)                       # e_max, e_sum "other" logits
        self.crystal_emb = nn.Embedding(N_CRYSTALS, crystal_emb)
        self.pc_other = nn.Linear(hidden + crystal_emb, 1)
        self.mult = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, N_MULT + 1))
        # hit pattern: one query per crystal (crystal embedding + trunk), plus "other"
        self.cmax_q = nn.Sequential(nn.Linear(hidden + crystal_emb, dec_hidden), nn.SiLU(),
                                    nn.Linear(dec_hidden, dec_hidden), nn.SiLU(), nn.Linear(dec_hidden, 1))
        self.cmax_other = nn.Linear(hidden, 1)

    def embed(self, gammas, context):
        mask = gammas > 0
        g = gammas.clamp(min=1e-3)
        tok = self.tok(torch.cat([fourier(gammas), (gammas / 15.0)[..., None], torch.log(g)[..., None]], -1))
        ctx = self.ctx_tok(context)[:, None, :]
        x = torch.cat([ctx, tok], 1)
        pad = torch.cat([torch.zeros(len(gammas), 1, dtype=torch.bool), ~mask], 1)
        x = self.enc(x, src_key_padding_mask=pad)
        tokx = x[:, 1:]
        m = mask.float()[..., None]
        mean = (tokx * m).sum(1) / m.sum(1).clamp(min=1)
        mx = tokx.masked_fill(~mask[..., None], -1e4).max(1).values
        return self.trunk(torch.cat([mean, mx, context], -1))

    def _decode(self, name, h, q):
        """h (B,H), q (B,Q,Fq) -> logits (B,Q)."""
        dd = self.dec[name]
        return dd["mlp"](dd["q"](q) + dd["h"](h)[:, None, :]).squeeze(-1)

    def forward(self, gammas, context, crystals=None):
        B = len(gammas)
        h = self.embed(gammas, context)
        out = {}
        qe = torch.cat([relative_features(self.e_centres, gammas),
                        fourier(self.e_centres)[None].expand(B, -1, -1)], -1)
        oth = self.other(h)
        for k, name in enumerate(["e_max", "e_sum"]):
            logits = torch.cat([self._decode(name, h, qe), oth[:, k:k + 1]], 1)
            out[name] = F.log_softmax(logits, 1)
        crystals = torch.arange(N_CRYSTALS) if crystals is None else crystals
        qc = torch.cat([relative_features(self.c_centres, gammas),
                        fourier(self.c_centres)[None].expand(B, -1, -1)], -1)    # (B,150,F)
        emb = self.crystal_emb(crystals)                                       # (K,E)
        nC, nK = qc.shape[1], len(crystals)
        q = torch.cat([qc[:, None].expand(B, nK, nC, -1),
                       emb[None, :, None, :].expand(B, nK, nC, -1)], -1).reshape(B, nK * nC, -1)
        lc = self._decode("per_crystal", h, q).reshape(B, nK, nC)
        lo = self.pc_other(torch.cat([h[:, None].expand(B, nK, -1), emb[None].expand(B, nK, -1)], -1))
        out["per_crystal"] = F.log_softmax(torch.cat([lc, lo], 2), 2)
        out["mult"] = F.log_softmax(self.mult(h), 1)
        allc = self.crystal_emb(torch.arange(N_CRYSTALS))
        lq = self.cmax_q(torch.cat([h[:, None].expand(B, N_CRYSTALS, -1), allc[None].expand(B, -1, -1)], -1))
        out["crystal_max"] = F.log_softmax(torch.cat([lq.squeeze(-1), self.cmax_other(h)], 1), 1)
        return out


# ------------------------------------------------------------- targets / loss
def with_other(t, axis=-1):
    """Append 'other' = 1 - sum to a per-generated-event target array."""
    other = np.clip(1.0 - t.sum(axis=axis, keepdims=True), 0.0, 1.0)
    return np.concatenate([np.clip(t, 0, None), other], axis=axis)


HEAD_WEIGHTS = dict(e_max=1.0, e_sum=1.0, per_crystal=1.0, mult=0.3, crystal_max=0.3)


def loss_fn(out, tgt, crystals):
    """Cross-entropy (multinomial NLL per event) summed over heads.
    tgt: dict of torch tensors with 'other' already appended."""
    total, parts = 0.0, {}
    for name, w in HEAD_WEIGHTS.items():
        t = tgt[name] if name != "per_crystal" else tgt[name][:, crystals]
        l = -(t * out[name]).sum(-1)
        l = l.mean() if name != "per_crystal" else l.sum(1).mean()
        parts[name] = float(l.detach())
        total = total + w * l
    return total, parts
