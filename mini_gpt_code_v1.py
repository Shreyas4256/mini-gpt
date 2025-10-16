"""
mini_gpt_code_v1: A self-contained mini GPT-style coding assistant.

This script can train a small decoder-only Transformer on local code/text data, build a
byte-level BPE tokenizer, and serve an offline coding assistant with retrieval over a
project directory. Everything runs locally with PyTorch on CPU or CUDA (if available).

Quick start:
  * Train from scratch (auto-collects data from --project_dir if --data missing):
      python mini_gpt_code_v1.py --project_dir . --train --out out
  * Resume training:
      python mini_gpt_code_v1.py --project_dir . --resume --out out
  * Sample plain text from a prompt:
      python mini_gpt_code_v1.py --sample --start "def fibonacci(n):" --max_new_tokens 120
  * Launch the assistant with retrieval and optional live docs:
      python mini_gpt_code_v1.py --assistant --project_dir . --out out --enable_web true

Live data fetching uses plain HTTP via the optional 'requests' library. Only whitelisted
hosts are allowed, responses are cached under --out/web_cache, and no AI APIs are ever
called. If requests is unavailable or disabled, the assistant continues with local
retrieval only and prints a helpful message.
"""

from __future__ import annotations

import argparse
import ast
import base64
import collections
import dataclasses
import functools
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin1", errors="ignore")


def write_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Tokenizer: Byte-level BPE implemented in-file
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class ByteLevelBPETokenizer:
    """Simple byte-level BPE tokenizer with in-file training and persistence."""

    def __init__(self) -> None:
        self.special_tokens = SPECIAL_TOKENS
        self.byte_tokens = 256
        self.token_bytes: Dict[int, bytes] = {}
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[int, int]] = []
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.merge_map: Dict[Tuple[int, int], int] = {}
        self.merge_token_ids: List[int] = []
        self._init_base_vocab()

    # ------------------------------------------------------------------
    def _init_base_vocab(self) -> None:
        # Base vocabulary: bytes 0-255
        for i in range(256):
            token = f"<byte_{i}>"
            self.vocab[token] = i
            self.id_to_token[i] = token
            self.token_bytes[i] = bytes([i])
        self.next_token_id = 256
        for tok in self.special_tokens:
            self.vocab[tok] = self.next_token_id
            self.id_to_token[self.next_token_id] = tok
            self.token_bytes[self.next_token_id] = b""
            self.next_token_id += 1

    # ------------------------------------------------------------------
    def train(self, text: str, num_merges: int) -> None:
        logging.info("Training tokenizer on %d characters with %d merges", len(text), num_merges)
        data = text.encode("utf-8", errors="ignore")
        ids = list(data)
        if not ids:
            logging.warning("Tokenizer training received empty text; keeping byte-only vocab")
            return
        base_vocab = 256 + len(self.special_tokens)
        # reset any previously learned merges
        self.merges = []
        self.merge_ranks = {}
        self.merge_map = {}
        self.merge_token_ids = []
        self.next_token_id = base_vocab
        self.id_to_token = {i: tok for i, tok in self.id_to_token.items() if i < base_vocab}
        self.vocab = {tok: idx for idx, tok in self.id_to_token.items()}
        self.token_bytes = {i: b for i, b in self.token_bytes.items() if i < base_vocab}
        for merge_index in range(num_merges):
            stats = self._get_pair_stats(ids)
            if not stats:
                logging.info("No more pairs to merge at step %d", merge_index)
                break
            best_pair = max(stats.items(), key=lambda kv: kv[1])[0]
            new_token = self.next_token_id
            self.next_token_id += 1
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = merge_index
            self.merge_map[best_pair] = new_token
            self.merge_token_ids.append(new_token)
            self.id_to_token[new_token] = f"<merge_{merge_index}>"
            self.vocab[self.id_to_token[new_token]] = new_token
            b = self.token_bytes[best_pair[0]] + self.token_bytes[best_pair[1]]
            self.token_bytes[new_token] = b
            ids = self._merge_ids(ids, best_pair, new_token)
        logging.info("Tokenizer training complete; vocab size=%d", self.vocab_size)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_pair_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
        stats: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        if not ids:
            return stats
        prev = ids[0]
        for cur in ids[1:]:
            stats[(prev, cur)] += 1
            prev = cur
        return stats

    @staticmethod
    def _merge_ids(ids: List[int], pair: Tuple[int, int], token: int) -> List[int]:
        i = 0
        new_ids: List[int] = []
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(token)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ------------------------------------------------------------------
    def encode_bytes(self, byte_ids: List[int]) -> List[int]:
        if not self.merges:
            return byte_ids
        tokens = list(byte_ids)
        while True:
            best_rank = None
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_map:
                    rank = self.merge_ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair is None:
                break
            new_token = self.merge_map[best_pair]
            tokens = self._merge_ids(tokens, best_pair, new_token)
        return tokens

    # ------------------------------------------------------------------
    def encode(self, text: str, add_special: bool = False) -> List[int]:
        data = text.encode("utf-8", errors="ignore")
        tokens = self.encode_bytes(list(data))
        if add_special:
            bos = self.special_token_id("<bos>")
            eos = self.special_token_id("<eos>")
            tokens = [bos] + tokens + [eos]
        return tokens

    # ------------------------------------------------------------------
    def decode(self, tokens: Sequence[int]) -> str:
        pieces: List[bytes] = []
        for t in tokens:
            if t in self.token_bytes:
                pieces.append(self.token_bytes[t])
            elif t in self.special_token_ids().values():
                continue
            else:
                pieces.append(b"")
        try:
            return b"".join(pieces).decode("utf-8", errors="ignore")
        except Exception:
            return b"".join(pieces).decode("latin1", errors="ignore")

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.next_token_id

    # ------------------------------------------------------------------
    def special_token_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("<unk>", 0))

    def special_token_ids(self) -> Dict[str, int]:
        return {tok: self.vocab[tok] for tok in self.special_tokens}

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        data = {
            "special_tokens": self.special_tokens,
            "next_token_id": self.next_token_id,
            "id_to_token": self.id_to_token,
            "token_bytes": {str(k): base64.b64encode(v).decode("ascii") for k, v in self.token_bytes.items()},
            "merges": [[a, b] for a, b in self.merges],
            "merge_token_ids": self.merge_token_ids,
        }
        write_json(path, data)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPETokenizer":
        obj = cls()
        data = read_json(path)
        obj.special_tokens = data["special_tokens"]
        obj.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        obj.vocab = {v: int(k) for k, v in obj.id_to_token.items()}
        obj.token_bytes = {int(k): base64.b64decode(v) for k, v in data["token_bytes"].items()}
        obj.next_token_id = data["next_token_id"]
        obj.merges = [tuple(pair) for pair in data["merges"]]
        obj.merge_ranks = {tuple(pair): i for i, pair in enumerate(obj.merges)}
        merge_token_ids = data.get("merge_token_ids")
        if merge_token_ids is None:
            base_next = 256 + len(obj.special_tokens)
            merge_token_ids = [base_next + i for i in range(len(obj.merges))]
        obj.merge_token_ids = merge_token_ids
        obj.merge_map = {tuple(pair): token_id for pair, token_id in zip(obj.merges, obj.merge_token_ids)}
        return obj


# -----------------------------------------------------------------------------
# GPT model components
# -----------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 0
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    batch_size: int = 16
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    min_lr: float = 6e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    dtype: str = "float32"
    bpe_merges: int = 1000
    seed: int = 42


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer["wte"].weight
        self.apply(self._init_weights)
        self._init_resid_scale()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_resid_scale(self) -> None:
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------


def collect_corpus_from_project(project_dir: Path, max_bytes: int = 200 * 1024 * 1024) -> str:
    allowed_exts = {".py", ".md", ".txt", ".json", ".js", ".ts", ".java", ".cpp", ".c", ".h"}
    texts: List[str] = []
    total = 0
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_exts:
            continue
        if any(part.startswith('.') and part not in (".", "..") for part in path.relative_to(project_dir).parts[:-1]):
            continue
        try:
            data = read_text_file(path)
        except Exception:
            continue
        header = f"\n\n### File: {path.relative_to(project_dir)}\n"
        chunk = header + data
        total += len(chunk.encode("utf-8", errors="ignore"))
        if total > max_bytes:
            logging.warning("Corpus truncated at %s due to max_bytes", path)
            break
        texts.append(chunk)
    if not texts:
        raise RuntimeError("No suitable files found to build training corpus. Provide --data.")
    return "\n".join(texts)


def prepare_datasets(
    args: argparse.Namespace, tokenizer: ByteLevelBPETokenizer, train_tokenizer: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data {data_path} not found")
        text = read_text_file(data_path)
    else:
        logging.info("No --data provided. Auto-collecting corpus from %s", args.project_dir)
        text = collect_corpus_from_project(Path(args.project_dir))
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    if train_tokenizer:
        tokenizer.train(train_text, args.config.bpe_merges)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    if len(train_ids) <= args.config.block_size or len(val_ids) <= args.config.block_size:
        raise ValueError(
            "Encoded dataset shorter than block size. Provide more data or reduce block_size in GPTConfig."
        )
    return torch.tensor(train_ids, dtype=torch.long), torch.tensor(val_ids, dtype=torch.long)


def get_batch(data: torch.Tensor, config: GPTConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + config.block_size] for i in ix])
    return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# Optimizer setup and training loop
# -----------------------------------------------------------------------------


def configure_optimizer(model: GPT, config: GPTConfig) -> torch.optim.Optimizer:
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    decay = {pn for pn in decay if pn in param_dict}
    no_decay = {pn for pn in no_decay if pn in param_dict}
    remaining = set(param_dict.keys()) - decay - no_decay
    if remaining:
        logging.debug("Parameters defaulting to no_decay: %s", sorted(remaining))
        no_decay.update(remaining)
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))


def get_lr(iter_num: int, config: GPTConfig) -> float:
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / max(1, config.warmup_iters)
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iter_num - config.warmup_iters) / max(1, config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def estimate_loss(model: GPT, data_train: torch.Tensor, data_val: torch.Tensor, config: GPTConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    out = {}
    for split, data in [("train", data_train), ("val", data_val)]:
        losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(data, config, device)
            with torch.no_grad():
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train_model(args: argparse.Namespace, config: GPTConfig, tokenizer: ByteLevelBPETokenizer) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    tokenizer_path = out_dir / "tokenizer.json"
    resume_ckpt_path = find_latest_checkpoint(out_dir) if args.resume else None
    if resume_ckpt_path:
        logging.info("Found checkpoint %s for resuming", resume_ckpt_path)
        ckpt_tmp = torch.load(resume_ckpt_path, map_location="cpu")
        config.__dict__.update(ckpt_tmp["config"])
        del ckpt_tmp
    train_tokenizer = True
    if args.resume and tokenizer_path.exists():
        logging.info("Loading existing tokenizer from %s", tokenizer_path)
        existing = ByteLevelBPETokenizer.load(tokenizer_path)
        tokenizer.__dict__.update(existing.__dict__)
        train_tokenizer = False
    data_train, data_val = prepare_datasets(args, tokenizer, train_tokenizer)
    tokenizer.save(tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    seed_everything(config.seed)
    device = determine_device(args.device)
    model = GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %.2fM", total_params / 1e6)
    if args.compile:
        try:  # pragma: no cover - depends on PyTorch version
            model = torch.compile(model)
            logging.info("Compiled model with torch.compile")
        except Exception as exc:
            logging.warning("torch.compile failed: %s", exc)
    optimizer = configure_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def load_checkpoint_if_needed(ckpt_path: Optional[Path]) -> int:
        if not ckpt_path:
            return 0
        logging.info("Resuming from %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", scaler.state_dict()))
        config.__dict__.update(checkpoint["config"])
        tokenizer_loaded = ByteLevelBPETokenizer.load(tokenizer_path)
        tokenizer.__dict__.update(tokenizer_loaded.__dict__)
        return checkpoint.get("iter", 0)

    iter_num = load_checkpoint_if_needed(resume_ckpt_path)
    model.train()
    for iter_idx in range(iter_num, config.max_iters):
        lr = get_lr(iter_idx, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(config.grad_accum_steps):
            x, y = get_batch(data_train, config, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and config.dtype != "float32"):
                _, loss = model(x, y)
            loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (iter_idx + 1) % config.eval_interval == 0 or iter_idx == 0:
            losses = estimate_loss(model, data_train, data_val, config, device)
            logging.info("iter %d | lr %.3e | train %.4f | val %.4f", iter_idx, lr, losses["train"], losses["val"])
            save_checkpoint(out_dir, iter_idx, model, optimizer, scaler, config)
    save_checkpoint(out_dir, config.max_iters, model, optimizer, scaler, config)


def save_checkpoint(out_dir: Path, iter_num: int, model: GPT, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, config: GPTConfig) -> None:
    ckpt = {
        "iter": iter_num,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dataclasses.asdict(config),
    }
    path = out_dir / f"ckpt_step{iter_num:06d}.pt"
    torch.save(ckpt, path)
    logging.info("Saved checkpoint to %s", path)


def find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
    if not out_dir.exists():
        return None
    ckpts = sorted(out_dir.glob("ckpt_step*.pt"))
    return ckpts[-1] if ckpts else None


# -----------------------------------------------------------------------------
# Retrieval engine (TF-IDF/BM25-like)
# -----------------------------------------------------------------------------


TokenFreq = Dict[str, int]


def tokenize_for_index(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower())


@dataclass
class Chunk:
    id: int
    path: str
    start: int
    end: int
    text: str
    tokens: TokenFreq
    length: int
    mtime: float


class RetrievalIndex:
    def __init__(self, project_dir: Path, out_dir: Path, extensions: List[str], exclude_dirs: List[str]) -> None:
        self.project_dir = project_dir
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.extensions = {ext.strip().lower() for ext in extensions if ext.strip()}
        self.exclude_dirs = set(exclude_dirs)
        self.index_path = out_dir / "index.jsonl"
        self.chunks: Dict[int, Chunk] = {}
        self.doc_freq: Dict[str, int] = collections.Counter()
        self.next_id = 0
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def should_index(self, path: Path) -> bool:
        if path.suffix.lower() not in self.extensions:
            return False
        for part in path.parts:
            if part in self.exclude_dirs:
                return False
        return True

    # ------------------------------------------------------------------
    def index_project(self, force: bool = False) -> None:
        with self.lock:
            if not force and self.index_path.exists():
                self._load_index()
                return
            else:
                self.chunks.clear()
                self.doc_freq.clear()
                self.next_id = 0
            for path in self.project_dir.rglob("*"):
                if not path.is_file() or not self.should_index(path):
                    continue
                self._index_file(path)
            self._save_index()
            logging.info("Indexed %d chunks", len(self.chunks))

    # ------------------------------------------------------------------
    def _index_file(self, path: Path) -> None:
        rel = str(path.relative_to(self.project_dir))
        try:
            text = read_text_file(path)
        except Exception:
            return
        chunks = self._chunk_file(text, path.suffix.lower())
        lines = text.splitlines()
        for start, end in chunks:
            snippet = "\n".join(lines[start - 1 : end])
            tokens = collections.Counter(tokenize_for_index(snippet))
            chunk = Chunk(
                id=self.next_id,
                path=rel,
                start=start,
                end=end,
                text=snippet,
                tokens=tokens,
                length=len(snippet),
                mtime=path.stat().st_mtime,
            )
            self.chunks[self.next_id] = chunk
            for term in tokens:
                self.doc_freq[term] += 1
            self.next_id += 1

    # ------------------------------------------------------------------
    def _chunk_file(self, text: str, suffix: str) -> List[Tuple[int, int]]:
        if suffix == ".py":
            try:
                tree = ast.parse(text)
                chunks: List[Tuple[int, int]] = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        end = getattr(node, "end_lineno", None)
                        if end is None:
                            continue
                        chunks.append((node.lineno, end))
                if chunks:
                    return chunks
            except SyntaxError:
                pass
        lines = text.splitlines()
        window = 120
        chunks = []
        start = 1
        while start <= len(lines):
            end = min(len(lines), start + window)
            chunks.append((start, end))
            start = end + 1
        return chunks

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 8) -> List[Chunk]:
        terms = tokenize_for_index(query)
        if not terms:
            return []
        scores: Dict[int, float] = collections.defaultdict(float)
        N = len(self.chunks) + 1
        for term in terms:
            df = self.doc_freq.get(term, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for chunk_id, chunk in self.chunks.items():
                tf = chunk.tokens.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + 0.5 + 1.5 * (chunk.length / 400.0)
                scores[chunk_id] += idf * (tf * (1.5 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self.chunks[cid] for cid, _ in ranked]

    # ------------------------------------------------------------------
    def summarize_context(self, chunks: List[Chunk], tokenizer: ByteLevelBPETokenizer, budget_tokens: int) -> Tuple[str, List[Tuple[str, int, int]]]:
        context_lines = []
        used = 0
        sources: List[Tuple[str, int, int]] = []
        for chunk in chunks:
            prefix = f"\n# {chunk.path}:{chunk.start}-{chunk.end}\n"
            tokens_needed = len(tokenizer.encode(prefix + chunk.text))
            if used + tokens_needed > budget_tokens:
                continue
            context_lines.append(prefix + chunk.text)
            used += tokens_needed
            sources.append((chunk.path, chunk.start, chunk.end))
        return "".join(context_lines), sources

    # ------------------------------------------------------------------
    def refresh_changed(self) -> None:
        with self.lock:
            removed = []
            removed_paths: set[str] = set()
            for chunk_id, chunk in list(self.chunks.items()):
                full_path = self.project_dir / chunk.path
                if not full_path.exists():
                    removed.append(chunk_id)
                    removed_paths.add(chunk.path)
                else:
                    mtime = full_path.stat().st_mtime
                    if mtime != chunk.mtime:
                        removed.append(chunk_id)
                        removed_paths.add(chunk.path)
            for cid in removed:
                old = self.chunks.pop(cid, None)
                if old:
                    for term in old.tokens:
                        self.doc_freq[term] -= 1
                        if self.doc_freq[term] <= 0:
                            del self.doc_freq[term]
            if removed:
                logging.info("Removed %d stale chunks", len(removed))
            existing_paths = {chunk.path for chunk in self.chunks.values()}
            for path in self.project_dir.rglob("*"):
                if not path.is_file() or not self.should_index(path):
                    continue
                rel = str(path.relative_to(self.project_dir))
                if rel in existing_paths and rel not in removed_paths:
                    continue
                self._index_file(path)
                existing_paths.add(rel)
            self._save_index()

    # ------------------------------------------------------------------
    def _save_index(self) -> None:
        ensure_dir(self.out_dir)
        with self.index_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks.values():
                row = {
                    "id": chunk.id,
                    "path": chunk.path,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "tokens": dict(chunk.tokens),
                    "length": chunk.length,
                    "mtime": chunk.mtime,
                }
                f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        self.chunks.clear()
        self.doc_freq.clear()
        self.next_id = 0
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chunk = Chunk(
                    id=row["id"],
                    path=row["path"],
                    start=row["start"],
                    end=row["end"],
                    text=row["text"],
                    tokens=collections.Counter(row["tokens"]),
                    length=row["length"],
                    mtime=row.get("mtime", 0.0),
                )
                self.chunks[chunk.id] = chunk
                for term in chunk.tokens:
                    self.doc_freq[term] += 1
                self.next_id = max(self.next_id, chunk.id + 1)


# -----------------------------------------------------------------------------
# Web fetching helpers (optional)
# -----------------------------------------------------------------------------


class WebFetcher:
    def __init__(self, out_dir: Path, enabled: bool, timeout: int, allow_list: List[str]) -> None:
        if enabled and requests is None:
            logging.info("requests is not available; install it to enable web fetching (pip install requests)")
        self.enabled = enabled and requests is not None
        self.timeout = timeout
        self.allow_list = [h.lower() for h in allow_list]
        self.cache_dir = out_dir / "web_cache"
        ensure_dir(self.cache_dir)
        if not self.enabled:
            logging.info("Web fetching disabled (requests missing or flag false)")

    def _allowed(self, url: str) -> bool:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return any(host == allowed or host.endswith("." + allowed) for allowed in self.allow_list)

    def fetch(self, url: str) -> Optional[str]:
        if not self.enabled:
            return None
        if not self._allowed(url):
            logging.warning("URL %s not in whitelist", url)
            return None
        import hashlib

        key = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".txt"
        cache_path = self.cache_dir / key
        if cache_path.exists():
            return read_text_file(cache_path)
        try:
            resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "mini-gpt-code-assistant"})
            if resp.status_code != 200:
                logging.warning("Failed to fetch %s: status %s", url, resp.status_code)
                return None
            if resp.headers.get("Content-Length") and int(resp.headers["Content-Length"]) > 1_000_000:
                logging.warning("Response too large for %s", url)
                return None
            if len(resp.content) > 1_000_000:
                logging.warning("Response content exceeded 1MB for %s", url)
                return None
            text = self._extract_text(resp.text)
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:  # pragma: no cover - network dependent
            logging.warning("Web fetch error for %s: %s", url, exc)
            return None

    def _extract_text(self, html: str) -> str:
        from html.parser import HTMLParser

        class Stripper(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.result: List[str] = []
                self.skip = False

            def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
                if tag in ("script", "style"):
                    self.skip = True

            def handle_endtag(self, tag: str) -> None:
                if tag in ("script", "style"):
                    self.skip = False

            def handle_data(self, data: str) -> None:
                if not self.skip:
                    self.result.append(data)

        stripper = Stripper()
        stripper.feed(html)
        text = "\n".join(line.strip() for line in stripper.result if line.strip())
        return text[:20000]

    def fetch_docs(self, keyword: str) -> List[Tuple[str, str]]:
        keyword = keyword.lower()
        docs = {
            "python dict": "https://docs.python.org/3/library/stdtypes.html#dict",
            "pytest": "https://docs.python.org/3/library/unittest.mock.html",
            "asyncio": "https://docs.python.org/3/library/asyncio.html",
            "typing": "https://docs.python.org/3/library/typing.html",
        }
        results = []
        for key, url in docs.items():
            if key in keyword:
                text = self.fetch(url)
                if text:
                    excerpt = text[:1000]
                    results.append((url, excerpt))
        return results


# -----------------------------------------------------------------------------
# Assistant CLI
# -----------------------------------------------------------------------------


def determine_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_and_tokenizer(out_dir: Path, device: torch.device) -> Tuple[GPT, GPTConfig, ByteLevelBPETokenizer]:
    ckpt_path = find_latest_checkpoint(out_dir)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoints found. Train the model first with --train")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, tokenizer


def safe_git_info(project_dir: Path) -> str:
    try:
        import subprocess

        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=project_dir, text=True)
        head = subprocess.check_output(["git", "log", "-1", "--pretty=%h %s"], cwd=project_dir, text=True)
        return f"Git status:\n{status}\nLast commit: {head}"
    except Exception:
        return "Git info unavailable"


class AssistantLoop:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = determine_device(args.device)
        self.out_dir = Path(args.out)
        self.project_dir = Path(args.project_dir)
        self.model, self.config, self.tokenizer = load_model_and_tokenizer(self.out_dir, self.device)
        extensions = [ext.strip() for ext in args.index_extensions.split(",")]
        exclude = [d.strip() for d in args.exclude_dirs.split(",")]
        self.retriever = RetrievalIndex(self.project_dir, self.out_dir, extensions, exclude)
        self.retriever.index_project(force=False)
        allow_list = [host.strip() for host in args.web_allow.split(",") if host.strip()]
        web_enabled = bool(args.enable_web)
        self.web = WebFetcher(self.out_dir, web_enabled, args.web_timeout, allow_list)
        if web_enabled and not self.web.enabled:
            print("Web fetching disabled: install requests or rerun with --enable_web false.")
        self.running = True
        self.reindex_interval = args.reindex_interval
        self.thread = threading.Thread(target=self._background_reindex, daemon=True)
        self.thread.start()

    def _background_reindex(self) -> None:
        while self.running:
            try:
                self.retriever.refresh_changed()
            except Exception as exc:
                logging.warning("Background reindex failed: %s", exc)
            time.sleep(self.reindex_interval)

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)

    def run(self) -> None:
        print("mini-gpt assistant ready. Type :quit to exit, :reload to reindex.")
        print(safe_git_info(self.project_dir))
        while self.running:
            try:
                query = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query == ":quit":
                break
            if query == ":reload":
                self.retriever.index_project(force=True)
                print("Index rebuilt.")
                continue
            self.answer_query(query)
        self.stop()

    def answer_query(self, query: str) -> None:
        chunks = self.retriever.search(query, k=8)
        context_budget = max(0, min(self.args.max_context_tokens, self.config.block_size - 64))
        context, sources = self.retriever.summarize_context(chunks, self.tokenizer, context_budget)
        doc_results: List[Tuple[str, str]] = []
        if self.web.enabled:
            doc_results = self.web.fetch_docs(query)
        prompt = self._build_prompt(query, context, doc_results)
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) >= self.config.block_size:
            input_ids = input_ids[-self.config.block_size + 1 :]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            input_tensor,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
        )
        generated = output[0].tolist()[len(input_ids) :]
        answer = self.tokenizer.decode(generated)
        print("\n--- Answer ---\n")
        print(answer.strip())
        if sources or doc_results:
            print("\nSources:")
            for path, start, end in sources:
                print(f"- {path}:{start}-{end}")
            for url, _ in doc_results:
                print(f"- {url}")
        print("\n---------------\n")

    def _build_prompt(self, query: str, context: str, docs: List[Tuple[str, str]]) -> str:
        system = "You are a concise coding assistant. Use the provided context and answer clearly.\n"
        prompt = system
        if context:
            prompt += f"\n[Project Context]\n{context}\n"
        for url, text in docs:
            prompt += f"\n[Doc {url}]\n{text}\n"
        prompt += f"\n[Question]\n{query}\n[Answer]\n"
        return prompt


# -----------------------------------------------------------------------------
# Sampling mode
# -----------------------------------------------------------------------------


def sample_text(args: argparse.Namespace) -> None:
    device = determine_device(args.device)
    out_dir = Path(args.out)
    model, config, tokenizer = load_model_and_tokenizer(out_dir, device)
    start_text = args.start or ""
    start_ids = tokenizer.encode(start_text, add_special=False)
    if not start_ids:
        start_ids = [tokenizer.special_token_id("<bos>")]
    x = torch.tensor([start_ids], dtype=torch.long, device=device)
    y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
    text = tokenizer.decode(y[0].tolist())
    print(text)


# -----------------------------------------------------------------------------
# Argument parsing and main entry
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini GPT code assistant")
    parser.add_argument("--data", type=str, default="", help="Path to training text data")
    parser.add_argument("--project_dir", type=str, default=".", help="Project directory to index")
    parser.add_argument("--train", action="store_true", help="Train the model from scratch")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--sample", action="store_true", help="Sample text from the model")
    parser.add_argument("--assistant", action="store_true", help="Launch the interactive assistant")
    parser.add_argument("--out", type=str, default="out", help="Output directory for checkpoints and caches")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    parser.add_argument("--start", type=str, default="", help="Prompt start text for sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--index_extensions", type=str, default=".py,.js,.ts,.tsx,.java,.cs,.cpp,.h,.hpp,.go,.rs,.php,.sh,.html,.css,.md", help="File extensions to index")
    parser.add_argument("--exclude_dirs", type=str, default=".git,.venv,node_modules,dist,build,out,__pycache__", help="Directories to exclude from indexing")
    parser.add_argument("--reindex_interval", type=int, default=5, help="Polling interval (s) for background reindexing")
    parser.add_argument("--enable_web", type=lambda x: str(x).lower() == "true", default=True, help="Enable live web fetching if requests available")
    parser.add_argument("--web_timeout", type=int, default=8, help="Timeout for web requests")
    parser.add_argument("--web_allow", type=str, default="docs.python.org,peps.python.org,raw.githubusercontent.com,developer.mozilla.org", help="Whitelisted hosts for web fetch")
    parser.add_argument("--max_context_tokens", type=int, default=2048, help="Max tokens for retrieval context")
    args = parser.parse_args(argv)
    args.config = GPTConfig()
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    if args.train:
        tokenizer = ByteLevelBPETokenizer()
        train_model(args, args.config, tokenizer)
    elif args.sample:
        sample_text(args)
    elif args.assistant:
        loop = AssistantLoop(args)
        loop.run()
    else:
        print("Specify an action: --train, --sample, or --assistant")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Usage examples (commented)
# -----------------------------------------------------------------------------

# 1) Prepare a quick training corpus from your repo (auto-collect)
# python mini_gpt_code_v1.py --project_dir . --train --out out

# 2) Resume training
# python mini_gpt_code_v1.py --project_dir . --resume --out out

# 3) Sample plain text from the model (no retrieval)
# python mini_gpt_code_v1.py --sample --start "def fibonacci(n):" --max_new_tokens 120 --temperature 0.9

# 4) Launch assistant with retrieval + optional live docs
# python mini_gpt_code_v1.py --assistant --project_dir . --out out --enable_web true

# 5) Limit indexing & reindex polling
# python mini_gpt_code_v1.py --assistant --index_extensions ".py,.md" --reindex_interval 3
