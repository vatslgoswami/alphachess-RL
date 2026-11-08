# Supervised run: Lichess 2024-01 + 2024-02, 6x64 ResNet

- Data: every rated game with Stockfish evals, both players >= 1800, no bullet, from the
  Hugging Face copy of the Lichess database (`scripts/build_lichess_hf.py`), 8 positions per
  game, split by game. 28,567,802 train / 1,586,377 val positions (~4M games).
- Model: `PolicyValueNet(blocks=6, channels=64)`, 505,228 parameters.
- Training: 10 epochs, batch 1024, AdamW lr 1e-3 (one-cycle), weight decay 1e-4, bf16,
  one H100 NVL, 278,990 steps in 48 min (~98,500 positions/s).
- Command: `INSECURE=1 TARGET=0 GPU=0 bash scripts/run_supervised_hpc.sh all`
- Checkpoint (on the HPC): `runs/sl_2024-01_2024-02_6x64/ckpt_00278990.pt`

| Validation (all 1.59M positions) | Start | End | Midsem trial (5k positions) |
|---|---:|---:|---:|
| top-1 move accuracy | 0% | **51.1%** | 11.3% |
| top-5 move accuracy | 0% | **88.9%** | 31.9% |
| policy cross-entropy | 8.43 | **1.52** | 4.95 |
| value MSE (predict-0 baseline 0.294) | 0.294 | **0.061** | 0.330 (baseline 0.312) |
| value R^2 = 1 - MSE / baseline | 0.00 | **0.79** | -0.06 |

Train and validation losses end almost equal (no overfitting). Both curves are still
creeping down at the end, so the next gains are more likely from a larger network than
from more data. Periodic checks during training used a fixed 200k-position validation
subset; the final row uses every validation position.
