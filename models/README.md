# Released checkpoints

| File | What | Size |
|---|---|---|
| `sl_jan-feb2024_6x64.pt` | Supervised policy-value ResNet (6 blocks x 64 channels, 505k params) trained on Lichess Jan + Feb 2024 (28.6M positions, 10 epochs). Val top-1 51.1%, top-5 88.9%, value MSE 0.061 vs 0.294 predict-0. Details: `reports/results/sl_2024-01_2024-02_6x64/`. | 5.9 MB |

Format: INTERFACES §6 (`model`, `optimizer`, `step`, `config`, ...). Load it anywhere (CPU is fine):

```python
from alphachess.model.evaluator import load_model, make_evaluator
net = load_model("models/sl_jan-feb2024_6x64.pt")            # PolicyValueNet, eval mode
evaluator = make_evaluator("models/sl_jan-feb2024_6x64.pt")  # for alphachess.mcts.search
```

With the CLI tools (UCI engine, arena, eval_positions):

```bash
ALPHACHESS_CHECKPOINT=models/sl_jan-feb2024_6x64.pt \
  python -m scripts.run_arena --evaluator alphachess.model.evaluator:make_evaluator ...
```

Quick checks (known positions, games vs a random mover; add `--months` on the HPC for the
held-out test split):

```bash
python -m scripts.check_model --checkpoint models/sl_jan-feb2024_6x64.pt
```
