"""Supervised training on encoded Lichess shards (Phase 1 trial run).

    python -m scripts.train_supervised --month 2024-01 --run-name sl_trial

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
    parser.add_argument("--data-dir", default=None, help="folder of encoded .npz shards")
    parser.add_argument("--run-name", default="sl_trial")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    from alphachess.train.supervised import TrainConfig, train

    month = args.month or cfg["lichess"]["month"]
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else repo_path(cfg["paths"]["lichess_dir"]) / "encoded" / month
    )
    train_shards = sorted(str(p) for p in data_dir.glob("train-*.npz"))
    val_shards = sorted(str(p) for p in data_dir.glob("val-*.npz"))
    if not train_shards:
        raise SystemExit(
            f"no train-*.npz shards in {data_dir}; run scripts/build_lichess_dataset.py and "
            "scripts/encode_lichess_dataset.py first"
        )
    if not val_shards:
        print(f"warning: no val-*.npz shards in {data_dir}; val metrics will be empty")

    tcfg = TrainConfig(
        run_dir=str(repo_path(cfg["paths"]["runs_dir"]) / args.run_name),
        train_shards=train_shards,
        val_shards=val_shards,
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
        extra={"month": month, "data_dir": str(data_dir)},
    )
    summary = train(tcfg, config_dict={"model": cfg["model"], "lichess": cfg["lichess"]})
    print(json.dumps(summary, indent=2))
    print(f"\nrun folder: {tcfg.run_dir}  (summary.json, metrics.jsonl, curves.png, checkpoints)")


if __name__ == "__main__":
    main()
