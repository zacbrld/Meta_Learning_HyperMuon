import argparse
import csv
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_text(data_dir, dataset):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{dataset}.txt"
    if path.exists():
        return path
    if dataset != "tinyshakespeare":
        raise ValueError("Only tinyshakespeare is auto-downloadable for now")
    print(f"Downloading {dataset} to {path}")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def load_char_data(data_dir, dataset, val_fraction=0.1):
    path = download_text(data_dir, dataset)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int((1.0 - val_fraction) * len(data))
    return data[:split], data[split:], len(chars)


def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        return self.out(y), h


def param_role(name):
    if name.startswith("emb."):
        return "emb"
    if "weight_ih" in name or "bias_ih" in name:
        return "W_ih"
    if "weight_hh" in name or "bias_hh" in name:
        return "W_hh"
    if name.startswith("out."):
        return "out"
    return "other"


@dataclass
class HyperState:
    lr_raw: torch.Tensor
    beta1_raw: torch.Tensor
    ema_alignment_lr: float = 0.0
    ema_alignment_beta1: float = 0.0


def logit(x):
    x = min(max(x, 1e-12), 1.0 - 1e-12)
    return math.log(x / (1.0 - x))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class GDUOAdam:
    def __init__(
        self,
        named_params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        mode="adam",
        lr_hyper_lr=1e-3,
        beta1_hyper_lr=1e-3,
        ema_beta=0.9,
        min_lr_ratio=0.25,
        max_lr_ratio=4.0,
        beta1_min=0.0,
        beta1_max=0.99,
    ):
        self.params = [(n, p) for n, p in named_params if p.requires_grad]
        self.base_lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.mode = mode
        self.lr_hyper_lr = lr_hyper_lr
        self.beta1_hyper_lr = beta1_hyper_lr
        self.ema_beta = ema_beta
        self.min_lr_ratio = min_lr_ratio
        self.max_lr_ratio = max_lr_ratio
        self.beta1_min = beta1_min
        self.beta1_max = beta1_max
        self.step_count = 0
        self.state = {}
        self.hyper = {}

        if mode == "adam":
            scopes = ["global"]
        elif mode == "hyperadam_global":
            scopes = ["global"]
        elif mode == "hyperadam_groupwise":
            scopes = ["emb", "W_ih", "W_hh", "out", "other"]
        elif mode == "hyperadam_layerwise":
            scopes = [name for name, _ in self.params]
        else:
            raise ValueError(f"unknown optimizer mode {mode}")

        scaled = (beta1 - beta1_min) / max(1e-12, beta1_max - beta1_min)
        for scope in scopes:
            self.hyper[scope] = HyperState(
                lr_raw=torch.tensor(0.0, dtype=torch.float64),
                beta1_raw=torch.tensor(logit(scaled), dtype=torch.float64),
            )

        for name, p in self.params:
            self.state[p] = {
                "step": 0,
                "m": torch.zeros_like(p),
                "v": torch.zeros_like(p),
                "prev_direction": None,
                "prev_beta1_deriv": None,
                "prev_lr": None,
                "scope": self.scope_for(name),
            }

    def scope_for(self, name):
        if self.mode in {"adam", "hyperadam_global"}:
            return "global"
        if self.mode == "hyperadam_groupwise":
            return param_role(name)
        return name

    def lr_scale(self, scope):
        if self.mode == "adam":
            return 1.0
        raw = float(self.hyper[scope].lr_raw.item())
        return math.exp(raw)

    def actual_lr(self, scope):
        return self.base_lr * self.lr_scale(scope)

    def beta1_value(self, scope):
        if self.mode == "adam":
            return self.beta1
        scaled = sigmoid(float(self.hyper[scope].beta1_raw.item()))
        return self.beta1_min + (self.beta1_max - self.beta1_min) * scaled

    def dbeta1_draw(self, scope):
        if self.mode == "adam":
            return 0.0
        scaled = sigmoid(float(self.hyper[scope].beta1_raw.item()))
        return (self.beta1_max - self.beta1_min) * scaled * (1.0 - scaled)

    def clamp_lr(self, scope):
        raw = float(self.hyper[scope].lr_raw.item())
        raw = min(max(raw, math.log(self.min_lr_ratio)), math.log(self.max_lr_ratio))
        self.hyper[scope].lr_raw.fill_(raw)

    @torch.no_grad()
    def meta_update(self):
        if self.mode == "adam":
            return
        by_scope = {}
        for name, p in self.params:
            if p.grad is None:
                continue
            scope = self.state[p]["scope"]
            by_scope.setdefault(scope, []).append((name, p))

        for scope, items in by_scope.items():
            h = self.hyper[scope]
            dot_lr = 0.0
            dot_beta1 = 0.0
            n_lr = 0
            n_beta1 = 0
            prev_lr = None
            for _, p in items:
                st = self.state[p]
                grad = p.grad.detach().float()
                if st["prev_direction"] is not None and st["prev_lr"] is not None:
                    dot_lr += float((grad * st["prev_direction"].float()).sum().detach().cpu())
                    n_lr += grad.numel()
                    prev_lr = st["prev_lr"]
                if st["prev_beta1_deriv"] is not None and st["prev_lr"] is not None:
                    dot_beta1 += float((grad * st["prev_beta1_deriv"].float()).sum().detach().cpu())
                    n_beta1 += grad.numel()
                    prev_lr = st["prev_lr"]

            if n_lr > 0 and prev_lr is not None:
                alignment = dot_lr / math.sqrt(n_lr)
                h.ema_alignment_lr = self.ema_beta * h.ema_alignment_lr + (1.0 - self.ema_beta) * alignment
                hypgrad = -h.ema_alignment_lr * prev_lr
                h.lr_raw.sub_(self.lr_hyper_lr * hypgrad)
                self.clamp_lr(scope)

            if n_beta1 > 0 and prev_lr is not None:
                alignment = dot_beta1 / math.sqrt(n_beta1)
                h.ema_alignment_beta1 = (
                    self.ema_beta * h.ema_alignment_beta1 + (1.0 - self.ema_beta) * alignment
                )
                hypgrad = -h.ema_alignment_beta1 * prev_lr
                h.beta1_raw.sub_(self.beta1_hyper_lr * hypgrad)

    @torch.no_grad()
    def step(self):
        self.meta_update()
        self.step_count += 1
        for name, p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]
            st["step"] += 1
            scope = st["scope"]
            lr = self.actual_lr(scope)
            beta1 = self.beta1_value(scope)
            dbeta1 = self.dbeta1_draw(scope)
            grad = p.grad.detach()
            m = st["m"]
            v = st["v"]
            prev_m = m.clone() if self.mode != "adam" else None

            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = v.sqrt().add_(self.eps)
            bc1 = 1.0 - beta1 ** st["step"]
            bc2 = 1.0 - self.beta2 ** st["step"]
            direction = (m / denom) * (math.sqrt(bc2) / bc1)
            if self.weight_decay:
                direction = direction + p.detach() * self.weight_decay
            p.sub_(direction, alpha=lr)

            beta1_deriv = None
            if self.mode != "adam":
                eps_fd = 1e-3
                beta_plus = min(self.beta1_max, beta1 + eps_fd * dbeta1)
                beta_minus = max(self.beta1_min, beta1 - eps_fd * dbeta1)
                m_plus = prev_m * beta_plus + grad * (1.0 - beta_plus)
                m_minus = prev_m * beta_minus + grad * (1.0 - beta_minus)
                bc1_plus = 1.0 - beta_plus ** st["step"]
                bc1_minus = 1.0 - beta_minus ** st["step"]
                d_plus = (m_plus / denom) * (math.sqrt(bc2) / bc1_plus)
                d_minus = (m_minus / denom) * (math.sqrt(bc2) / bc1_minus)
                beta1_deriv = (d_plus - d_minus) / (2.0 * eps_fd)

            st["prev_direction"] = direction.detach().float().clone()
            st["prev_beta1_deriv"] = (
                beta1_deriv.detach().float().clone() if beta1_deriv is not None else None
            )
            st["prev_lr"] = lr

    def zero_grad(self):
        for _, p in self.params:
            p.grad = None

    def metrics(self):
        rows = []
        if self.mode == "adam":
            rows.append(
                dict(scope="global", lr_scale=1.0, actual_lr=self.base_lr, beta1=self.beta1)
            )
            return rows
        for scope in sorted(self.hyper):
            rows.append(
                dict(
                    scope=scope,
                    lr_scale=self.lr_scale(scope),
                    actual_lr=self.actual_lr(scope),
                    beta1=self.beta1_value(scope),
                    ema_alignment_lr=self.hyper[scope].ema_alignment_lr,
                    ema_alignment_beta1=self.hyper[scope].ema_alignment_beta1,
                )
            )
        return rows


@torch.no_grad()
def evaluate(model, data, batch_size, seq_len, eval_batches, device):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def write_scope_rows(path, step, elapsed, rows):
    exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        fieldnames = [
            "step",
            "elapsed_sec",
            "scope",
            "lr_scale",
            "actual_lr",
            "beta1",
            "ema_alignment_lr",
            "ema_alignment_beta1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"step": step, "elapsed_sec": elapsed, **row})


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    train_data, val_data, vocab_size = load_char_data(args.data_dir, args.dataset)
    model = CharGRU(
        vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    opt = GDUOAdam(
        model.named_parameters(),
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        mode=args.optimizer,
        lr_hyper_lr=args.lr_hyper_lr,
        beta1_hyper_lr=args.beta1_hyper_lr,
        ema_beta=args.ema_beta,
        min_lr_ratio=args.min_lr_ratio,
        max_lr_ratio=args.max_lr_ratio,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset}_{args.optimizer}_seed{args.seed}"
    train_csv = results_dir / f"{prefix}_train.csv"
    hyper_csv = results_dir / f"{prefix}_hyper.csv"
    start = time.time()

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.seq_len, args.eval_batches, device
            )
            elapsed = time.time() - start
            with open(train_csv, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "elapsed_sec", "val_loss", "val_bpc"]
                )
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(
                    {
                        "step": step,
                        "elapsed_sec": elapsed,
                        "val_loss": val_loss,
                        "val_bpc": val_loss / math.log(2),
                    }
                )
            write_scope_rows(hyper_csv, step, elapsed, opt.metrics())
            print(
                f"Eval step={step} val_loss={val_loss:.4f} "
                f"val_bpc={val_loss / math.log(2):.4f}"
            )
            for row in opt.metrics():
                if row["scope"] in {"global", "emb", "W_ih", "W_hh", "out"}:
                    print(
                        f"  {row['scope']}: lr_scale={row['lr_scale']:.4g} "
                        f"beta1={row['beta1']:.4g}"
                    )

        if step == args.steps:
            break

        x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tinyshakespeare")
    parser.add_argument("--data_dir", default="datasets_rnn")
    parser.add_argument("--results_dir", default="results_rnn_gduo")
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "hyperadam_global", "hyperadam_groupwise", "hyperadam_layerwise"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_hyper_lr", type=float, default=1.0)
    parser.add_argument("--beta1_hyper_lr", type=float, default=10.0)
    parser.add_argument("--ema_beta", type=float, default=0.9)
    parser.add_argument("--min_lr_ratio", type=float, default=0.25)
    parser.add_argument("--max_lr_ratio", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
