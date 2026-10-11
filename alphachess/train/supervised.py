"""Supervised trainer (Phase 1): fit the policy-value net to encoded Lichess shards.

Data are INTERFACES §5 ``.npz`` shards (``planes``, ``pi_idx``, ``pi_p``, ``z``). For Lichess the
policy target is one-hot on the human move and ``z = tanh(cp / 400)`` from the Stockfish eval.

Loss for one batch (AlphaZero's loss without the L2 term, which AdamW's weight decay covers)::

    policy_loss = mean over positions of  -sum_a pi(a) * log softmax(logits)(a)   # cross-entropy
    value_loss  = mean over positions of  (v - z)^2                                # MSE
    loss        = policy_loss + value_weight * value_loss

Illegal moves are not masked in the loss: the targets only ever put mass on legal moves, so the
network learns to push illegal logits down on its own, and MCTS masks them anyway at play time.

Shards are streamed one at a time (shuffled order, shuffled within each shard), so memory stays
at about one shard however large the dataset is.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alphachess.common.checkpoint import save_checkpoint
from alphachess.encoding.board import NUM_MOVES
from alphachess.model.net import PolicyValueNet, count_parameters


@dataclass
class TrainConfig:
    run_dir: str
    train_shards: list[str]
    val_shards: list[str]
    blocks: int = 6
    channels: int = 64
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    epochs: int = 10
    max_steps: int | None = None
    eval_every: int = 200
    checkpoint_minutes: float = 20.0
    seed: int = 0
    device: str = "auto"
    notes: str = ""
    extra: dict = field(default_factory=dict)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- data --------------------------------------------------------------------------------------


def load_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in ("planes", "pi_idx", "pi_p", "z")}


def count_positions(paths: list[str]) -> int:
    total = 0
    for p in paths:
        with np.load(p) as data:
            total += len(data["z"])
    return total


def iterate_batches(
    paths: list[str], batch_size: int, rng: np.random.Generator | None = None
) -> Iterator[dict[str, np.ndarray]]:
    """Yield batches from shards. With ``rng``: shuffled shard order and rows; else in order."""
    order = list(paths)
    if rng is not None:
        rng.shuffle(order)
    for path in order:
        shard = load_shard(path)
        n = len(shard["z"])
        rows = rng.permutation(n) if rng is not None else np.arange(n)
        for start in range(0, n, batch_size):
            take = rows[start : start + batch_size]
            yield {k: v[take] for k, v in shard.items()}


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planes": torch.from_numpy(batch["planes"].astype(np.float32)).to(device),
        "pi_idx": torch.from_numpy(batch["pi_idx"].astype(np.int64)).to(device),
        "pi_p": torch.from_numpy(batch["pi_p"].astype(np.float32)).to(device),
        "z": torch.from_numpy(batch["z"].astype(np.float32)).to(device),
    }


def dense_policy_target(pi_idx: torch.Tensor, pi_p: torch.Tensor) -> torch.Tensor:
    """(B, K) sparse targets padded with -1 -> dense (B, 4672) distribution."""
    valid = pi_idx >= 0
    target = torch.zeros(pi_idx.shape[0], NUM_MOVES, device=pi_idx.device)
    target.scatter_add_(1, pi_idx.clamp(min=0), pi_p * valid)
    return target


# --- loss and metrics --------------------------------------------------------------------------


def compute_losses(
    model: PolicyValueNet, batch: dict[str, torch.Tensor], value_weight: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, value = model(batch["planes"])
    target = dense_policy_target(batch["pi_idx"], batch["pi_p"])
    policy_loss = -(target * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value.float(), batch["z"])
    loss = policy_loss + value_weight * value_loss
    return loss, {
        "logits": logits,
        "value": value,
        "target": target,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


@torch.no_grad()
def evaluate(model: PolicyValueNet, paths: list[str], device: torch.device, batch_size: int = 1024):
    """Policy top-1/top-5 accuracy, policy CE, value MSE/MAE, and the predict-0 value baseline."""
    model.eval()
    n = 0
    sums = {"policy_loss": 0.0, "value_mse": 0.0, "value_mae": 0.0, "top1": 0.0, "top5": 0.0}
    z_sq = 0.0
    for raw in iterate_batches(paths, batch_size):
        batch = to_tensors(raw, device)
        _, out = compute_losses(model, batch)
        b = batch["z"].shape[0]
        best = out["target"].argmax(dim=1)
        top5 = out["logits"].topk(5, dim=1).indices
        err = out["value"].float() - batch["z"]
        sums["policy_loss"] += out["policy_loss"].item() * b
        sums["value_mse"] += (err**2).sum().item()
        sums["value_mae"] += err.abs().sum().item()
        sums["top1"] += (out["logits"].argmax(dim=1) == best).sum().item()
        sums["top5"] += (top5 == best[:, None]).any(dim=1).sum().item()
        z_sq += (batch["z"] ** 2).sum().item()
        n += b
    model.train()
    if n == 0:
        return {"positions": 0}
    metrics = {
        "positions": n,
        "policy_loss": sums["policy_loss"] / n,
        "policy_top1": sums["top1"] / n,
        "policy_top5": sums["top5"] / n,
        "value_mse": sums["value_mse"] / n,
        "value_mae": sums["value_mae"] / n,
        "value_mse_baseline_zero": z_sq / n,  # error of always predicting 0
    }
    return metrics


# --- training loop -----------------------------------------------------------------------------


def train(cfg: TrainConfig, config_dict: dict | None = None) -> dict:
    """Train, checkpoint and evaluate. Returns the final summary (also saved as summary.json)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = PolicyValueNet(blocks=cfg.blocks, channels=cfg.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_train = count_positions(cfg.train_shards)
    steps_per_epoch = math.ceil(n_train / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max(total_steps, 1), pct_start=0.05
    )
    use_bf16 = device.type == "cuda"
    saved_config = {"train": asdict(cfg), **(config_dict or {})}

    def checkpoint(step: int, notes: str) -> Path:
        path = run_dir / f"ckpt_{step:08d}.pt"
        save_checkpoint(path, model, optimizer, step, saved_config, notes=notes)
        return path

    log_path = run_dir / "metrics.jsonl"
    log = log_path.open("a", encoding="utf-8")

    def write(record: dict) -> None:
        log.write(json.dumps(record) + "\n")
        log.flush()

    print(
        f"device={device} params={count_parameters(model):,} train_positions={n_train:,} "
        f"steps={total_steps:,} ({steps_per_epoch:,}/epoch)"
    )
    initial = evaluate(model, cfg.val_shards, device)
    write({"step": 0, "split": "val", **initial})
    print(f"step 0 val {format_metrics(initial)}")

    step, start, last_ckpt = 0, time.time(), time.time()
    running: dict[str, float] = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "n": 0}
    model.train()
    done = False
    for epoch in range(cfg.epochs):
        for raw in iterate_batches(cfg.train_shards, cfg.batch_size, rng):
            batch = to_tensors(raw, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, out = compute_losses(model, batch, cfg.value_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            step += 1

            running["loss"] += loss.item()
            running["policy_loss"] += out["policy_loss"].item()
            running["value_loss"] += out["value_loss"].item()
            running["n"] += 1

            if step % cfg.eval_every == 0 or step == total_steps:
                n = running.pop("n")
                train_rec = {k: v / n for k, v in running.items()}
                running = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "n": 0}
                val = evaluate(model, cfg.val_shards, device)
                elapsed = time.time() - start
                write({"step": step, "epoch": epoch, "split": "train", "seconds": elapsed,
                       "lr": scheduler.get_last_lr()[0], **train_rec})
                write({"step": step, "epoch": epoch, "split": "val", "seconds": elapsed, **val})
                print(
                    f"step {step} epoch {epoch} train_loss {train_rec['loss']:.3f} "
                    f"val {format_metrics(val)} ({elapsed:.0f}s)"
                )
            if time.time() - last_ckpt > cfg.checkpoint_minutes * 60:
                checkpoint(step, "periodic")
                last_ckpt = time.time()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    log.close()
    final_path = checkpoint(step, "final " + cfg.notes)
    final = evaluate(model, cfg.val_shards, device)
    summary = {
        "steps": step,
        "seconds": time.time() - start,
        "device": str(device),
        "parameters": count_parameters(model),
        "train_positions": n_train,
        "val_initial": initial,
        "val_final": final,
        "checkpoint": str(final_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        plot_curves(log_path, run_dir / "curves.png")
    except Exception as exc:  # noqa: BLE001 — plotting must never fail a run
        print(f"(could not plot curves: {exc})")
    return summary


def format_metrics(m: dict) -> str:
    if not m.get("positions"):
        return "(no val data)"
    return (
        f"top1 {m['policy_top1']:.3f} top5 {m['policy_top5']:.3f} "
        f"policy_ce {m['policy_loss']:.3f} value_mse {m['value_mse']:.4f} "
        f"(predict-0 baseline {m['value_mse_baseline_zero']:.4f})"
    )


def plot_curves(log_path: Path, out_path: Path) -> None:
    """Training/validation curves for the slides: policy CE, top-1 accuracy, value MSE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    train_r = [r for r in records if r["split"] == "train"]
    val_r = [r for r in records if r["split"] == "val" and r.get("positions")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot([r["step"] for r in train_r], [r["policy_loss"] for r in train_r], label="train")
    axes[0].plot([r["step"] for r in val_r], [r["policy_loss"] for r in val_r], label="val")
    axes[0].set_title("Policy cross-entropy")
    axes[1].plot([r["step"] for r in val_r], [r["policy_top1"] for r in val_r], label="top-1")
    axes[1].plot([r["step"] for r in val_r], [r["policy_top5"] for r in val_r], label="top-5")
    axes[1].set_title("Val move accuracy (human move)")
    axes[2].plot([r["step"] for r in train_r], [r["value_loss"] for r in train_r], label="train")
    axes[2].plot([r["step"] for r in val_r], [r["value_mse"] for r in val_r], label="val")
    if val_r:
        axes[2].axhline(val_r[0]["value_mse_baseline_zero"], ls="--", c="gray", label="predict 0")
    axes[2].set_title("Value MSE")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
