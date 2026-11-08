"""Supervised training on encoded Lichess shards.

    python -m scripts.train_supervised --month 2024-01 --run-name sl_trial
    python -m scripts.train_supervised --months 2024-01,2024-02 --run-name sl_2m_6x64 \\
        --batch-size 1024 --eval-every 1000 --device cuda

Reads ``<paths.lichess_dir>/encoded/<month>/{train,val}-*.npz`` (from
``scripts/encode_lichess_dataset.py``), trains the policy-value net, and writes to
``runs/<run_name>/``:

- ``ckpt_<step:08d>.pt``   checkpoints (INTERFACES §6), the last one at the end of the run;
- ``metrics.jsonl``        train and val metrics every ``--eval-every`` steps;
- ``summary.json``         val metrics before and after training, run time, parameter count;
- ``curves.png``           policy loss, move accuracy and value MSE curves (for the slides).

Defaults come from ``config.yaml`` (``model`` and ``train`` sections). For the 5,000-position
smoke data, a few epochs on a laptop CPU or Apple GPU take a minute or two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphachess.common.config import load_config, repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", default=None, help="YYYY-MM (default: lichess.month)")
    parser.add_argument("--months", default=None, help="several months, e.g. 2024-01,2024-02")
    parser.add_argument(
        "--data-dir", action="append", default=None,
        help="folder of encoded .npz shards (repeat the flag for several folders)",
    )
    parser.add_argument("--run-name", default="sl_trial")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--val-periodic-positions", type=int, default=200_000,
        help="validation positions used for the checks during training (0 = all); "
        "the final check always uses every validation shard",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def pick_periodic_val(val_by_dir: list[list[str]], cap: int) -> list[str] | None:
    """Fixed validation subset for checks during training: shards taken round-robin across
    months until ``cap`` positions. None (use all) when ``cap`` is 0 or the total is small."""
    if cap <= 0:
        return None
    import numpy as np

    picked, total = [], 0
    for i in range(max((len(v) for v in val_by_dir), default=0)):
        for shards in val_by_dir:
            if i < len(shards) and total < cap:
                picked.append(shards[i])
                with np.load(shards[i]) as data:
                    total += len(data["z"])
    if sum(len(v) for v in val_by_dir) == len(picked):
        return None
    print(f"periodic validation: {total:,} positions in {len(picked)} shards "
          "(final check uses all validation shards)")
    return picked


def main() -> None:
    args = parse_args()
    cfg = load_config()
    from alphachess.train.supervised import TrainConfig, train

    if args.data_dir:
        months = []
        data_dirs = [Path(d) for d in args.data_dir]
    else:
        months = (args.months or args.month or cfg["lichess"]["month"]).split(",")
        base = repo_path(cfg["paths"]["lichess_dir"]) / "encoded"
        data_dirs = [base / m.strip() for m in months]
    train_shards, val_shards, val_by_dir = [], [], []
    for data_dir in data_dirs:
        found = sorted(str(p) for p in data_dir.glob("train-*.npz"))
        if not found:
            raise SystemExit(
                f"no train-*.npz shards in {data_dir}; run scripts/build_lichess_dataset.py and "
                "scripts/encode_lichess_dataset.py first"
            )
        train_shards += found
        found_val = sorted(str(p) for p in data_dir.glob("val-*.npz"))
        val_shards += found_val
        val_by_dir.append(found_val)
    print(f"{len(train_shards)} train / {len(val_shards)} val shards from "
          f"{', '.join(str(d) for d in data_dirs)}")
    if not val_shards:
        print("warning: no val-*.npz shards; val metrics will be empty")
    periodic = pick_periodic_val(val_by_dir, args.val_periodic_positions)

    tcfg = TrainConfig(
        run_dir=str(repo_path(cfg["paths"]["runs_dir"]) / args.run_name),
        train_shards=train_shards,
        val_shards=val_shards,
        val_periodic_shards=periodic,
        blocks=args.blocks or cfg["model"]["blocks"],
        channels=args.channels or cfg["model"]["channels"],
        batch_size=args.batch_size or cfg["train"]["batch_size"],
        lr=args.lr or cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        value_weight=args.value_weight,
        epochs=args.epochs,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        checkpoint_minutes=cfg["train"]["checkpoint_minutes"],
        seed=args.seed,
        device=args.device,
        notes=args.notes,
        extra={"months": months, "data_dirs": [str(d) for d in data_dirs]},
    )
    summary = train(tcfg, config_dict={"model": cfg["model"], "lichess": cfg["lichess"]})
    print(json.dumps(summary, indent=2))
    print(f"\nrun folder: {tcfg.run_dir}  (summary.json, metrics.jsonl, curves.png, checkpoints)")


if __name__ == "__main__":
    main()
