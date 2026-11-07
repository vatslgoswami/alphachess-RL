#!/usr/bin/env bash
# Supervised training on several months of Lichess, end to end, on a GPU workstation.
#
#   bash scripts/run_supervised_hpc.sh check     # environment, GPU, disk, network (1 min)
#   bash scripts/run_supervised_hpc.sh build     # stream + sample each month, in parallel
#   bash scripts/run_supervised_hpc.sh encode    # Parquet -> .npz training shards, in parallel
#   bash scripts/run_supervised_hpc.sh train     # train the ResNet on all months together
#   bash scripts/run_supervised_hpc.sh all       # build, encode, train
#
# Run long steps inside tmux (or with nohup) so they survive a disconnect. Logs: runs/logs/.
# Settings are environment variables, e.g.
#   MONTHS="2024-01 2024-02" TARGET=4000000 GPU=0 bash scripts/run_supervised_hpc.sh all
#
# Rough timings (from the 5k smoke run: ~1,400 games scanned/s per process, network-bound):
#   build (SOURCE=hf, default): Parquet files fetched by WORKERS processes per month, so speed
#           is mostly your network; TARGET=0 takes every eval game in the month
#   build (SOURCE=lichess): ~1,400 games/s single stream, TARGET=4,000,000 ~ 5 h per month
#   encode: ~11,000 positions/s per process, ~6 min per 4M-position month
#   train:  see the pos/s and eta printed every --eval-every steps
set -euo pipefail
cd "$(dirname "$0")/.."

MONTHS=${MONTHS:-"2024-01 2024-02"}
TARGET=${TARGET:-4000000}            # positions per month (8 per game)
INSECURE=${INSECURE:-0}              # 1 = skip TLS checks for the data host (firewalls)
SOURCE=${SOURCE:-hf}                 # hf = Hugging Face Parquet copy (parallel) | lichess = stream
WORKERS=${WORKERS:-24}               # download+parse processes per month (SOURCE=hf)
export INSECURE
GPU=${GPU:-0}
BLOCKS=${BLOCKS:-6}
CHANNELS=${CHANNELS:-64}
EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-1024}
LR=${LR:-0.001}
EVAL_EVERY=${EVAL_EVERY:-1000}
RUN=${RUN:-sl_$(echo "$MONTHS" | tr ' ' '_')_${BLOCKS}x${CHANNELS}}
LOGS=runs/logs
mkdir -p "$LOGS"
LICHESS_URL="https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"

# Old system certificate stores can't verify database.lichess.org; use certifi's up-to-date
# bundle for Python's urllib, which the Lichess streamer uses.
if [ -z "${SSL_CERT_FILE:-}" ] && python -c "import certifi" 2>/dev/null; then
  SSL_CERT_FILE=$(python -m certifi)
  export SSL_CERT_FILE
fi

check() {
  echo "== python";   python --version
  python - <<'EOF'
import torch, chess, numpy
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  gpu", i, torch.cuda.get_device_name(i))
EOF
  echo "== GPU usage (pick a free one with GPU=...)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
  echo "== CPU cores: $(nproc)"
  echo "== disk free here"; df -h . | tail -1
  echo "== Hugging Face copy of the Lichess database"
  python -c '
import os
from alphachess.lichess.hf import list_month_files
files = list_month_files("2024-01", insecure=os.environ.get("INSECURE") == "1")
print(f"Python can reach Hugging Face: {len(files)} Parquet files for 2024-01")
' || echo "NO ACCESS to the Hugging Face dataset from Python"
  echo "== network to Lichess (certificates: ${SSL_CERT_FILE:-system default})"
  python -c '
import os, ssl, sys, urllib.request
ctx = ssl.create_default_context()
if os.environ.get("INSECURE") == "1":
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    print("(INSECURE=1: certificate check skipped)")
req = urllib.request.Request(sys.argv[1], method="HEAD")
with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
    size = int(r.headers.get("Content-Length", 0)) / 1e9
    print(f"Python can reach Lichess: HTTP {r.status}, file {size:.1f} GB")
' "$LICHESS_URL" || echo "NO ACCESS to database.lichess.org from Python"
  echo "== tests"; python -m pytest -q tests/test_encoding.py tests/test_model.py 2>&1 | tail -1
}

build() {
  for m in $MONTHS; do
    if ls "data/lichess/raw/$m"/*.parquet >/dev/null 2>&1; then
      echo "data/lichess/raw/$m already has shards; move them away first to rebuild"; exit 1
    fi
    echo "building $m -> $LOGS/build_$m.log"
    insecure_flag=""
    [ "$INSECURE" = "1" ] && insecure_flag="--insecure-download"
    if [ "$SOURCE" = "hf" ]; then
      python -m scripts.build_lichess_hf --month "$m" --target "$TARGET" \
        --workers "$WORKERS" $insecure_flag > "$LOGS/build_$m.log" 2>&1 &
    else
      python -m scripts.build_lichess_dataset --month "$m" --target "$TARGET" $insecure_flag \
        > "$LOGS/build_$m.log" 2>&1 &
    fi
  done
  wait
  for m in $MONTHS; do tail -n 3 "$LOGS/build_$m.log"; done
}

encode() {
  for m in $MONTHS; do
    echo "encoding $m -> $LOGS/encode_$m.log"
    python -m scripts.encode_lichess_dataset --month "$m" > "$LOGS/encode_$m.log" 2>&1 &
  done
  wait
  for m in $MONTHS; do tail -n 1 "$LOGS/encode_$m.log"; done
}

train() {
  echo "training $RUN on GPU $GPU -> $LOGS/train_$RUN.log"
  CUDA_VISIBLE_DEVICES=$GPU python -m scripts.train_supervised \
    --months "$(echo "$MONTHS" | tr ' ' ',')" --run-name "$RUN" \
    --blocks "$BLOCKS" --channels "$CHANNELS" --epochs "$EPOCHS" \
    --batch-size "$BATCH" --lr "$LR" --eval-every "$EVAL_EVERY" --device cuda \
    2>&1 | tee "$LOGS/train_$RUN.log"
}

case "${1:-}" in
  check) check ;;
  build) build ;;
  encode) encode ;;
  train) train ;;
  all) build && encode && train ;;
  *) sed -n 2,20p "$0"; exit 1 ;;
esac
