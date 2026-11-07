"""Read Lichess games from the official Hugging Face copy of the database (Parquet, per month).

``Lichess/standard-chess-games`` holds the same games as database.lichess.org, split into many
Parquet files per month (``data/year=YYYY/month=MM/train-NNNNN-of-NNNNN.parquet``). Each row is
one game: header columns (``Event``, ``Site``, ``WhiteElo``, ``TimeControl``, ...) plus the PGN
``movetext``, including the ``[%eval ...]`` comments we use for value targets.

Because the month is split into independent files, ``process_parquet`` can run in many worker
processes at once, unlike the single ``.pgn.zst`` stream. Each worker downloads one file, keeps
games that pass ``keep_headers`` and carry evals, samples positions with ``sample_game`` (the same
code as the stream path) and deletes the file.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import chess.pgn

from alphachess.lichess.sample import SampleConfig, keep_headers, sample_game

DATASET = "Lichess/standard-chess-games"
# HF_ENDPOINT is the standard Hugging Face setting for mirrors (and lets tests use a local server).
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
API_TREE = ENDPOINT + "/api/datasets/{dataset}/tree/main/data/year%3D{year}/month%3D{month}"
FILE_URL = ENDPOINT + "/datasets/{dataset}/resolve/main/{path}"
INSECURE_HOSTS = ("huggingface.co", urllib.parse.urlparse(ENDPOINT).hostname or "")
USER_AGENT = "alphachess-rl (supervised training data)"


def _open(url: str, insecure: bool = False, timeout: float = 60):
    """urlopen with a User-Agent; ``insecure`` skips TLS checks, for huggingface.co only."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if not insecure:
        return urllib.request.urlopen(request, timeout=timeout)
    host = urllib.parse.urlparse(url).hostname or ""
    if not any(host == h or host.endswith("." + h) for h in INSECURE_HOSTS):
        raise ValueError(f"insecure download is only allowed for {INSECURE_HOSTS}")
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def list_month_files(month: str, insecure: bool = False, dataset: str = DATASET) -> list[str]:
    """Paths of one month's Parquet files, e.g. ``list_month_files("2024-01")``."""
    year, mon = month.split("-")
    url = API_TREE.format(dataset=dataset, year=year, month=mon)
    with _open(url, insecure) as response:
        entries = json.load(response)
    paths = sorted(e["path"] for e in entries if e.get("type") == "file"
                   and e["path"].endswith(".parquet"))
    if not paths:
        raise RuntimeError(f"no Parquet files listed for {month} at {url}")
    return paths


def file_url(path: str, dataset: str = DATASET) -> str:
    return FILE_URL.format(dataset=dataset, path=urllib.parse.quote(path, safe="/="))


def download(url: str, dest: Path, insecure: bool = False, retries: int = 3) -> Path:
    """Download to ``dest`` (via a ``.part`` file), retrying transient failures."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _open(url, insecure, timeout=120) as response, open(part, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 22)
            part.rename(dest)
            return dest
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    return dest


def row_to_game(row: dict) -> chess.pgn.Game | None:
    """Rebuild a PGN game from one Parquet row (header columns + ``movetext``)."""
    movetext = row.get("movetext")
    if not movetext:
        return None
    # Only tags that change how moves are read go into the PGN text; the rest are copied in
    # afterwards, so player names with quotes or backslashes can't break parsing.
    setup = [f'[{k} "{row[k]}"]' for k in ("Variant", "SetUp", "FEN") if row.get(k)]
    game = chess.pgn.read_game(io.StringIO("\n".join(setup) + "\n\n" + movetext + "\n\n"))
    if game is None:
        return None
    for key, value in row.items():
        if key != "movetext" and value is not None:
            game.headers[key] = str(value)
    return game


def process_parquet(path: str | Path, cfg: SampleConfig, batch_size: int = 50_000) -> dict:
    """Sample positions from one Parquet file. Returns rows plus per-file counts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    stats = {"games_seen": 0, "games_kept_headers": 0, "games_with_evals": 0}
    rows: list[dict] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        stats["games_seen"] += batch.num_rows
        if cfg.require_eval:
            # Cheap string test before any PGN parsing: most games carry no engine evals.
            has_eval = pc.match_substring(batch.column("movetext"), "%eval")
            batch = batch.filter(pc.fill_null(has_eval, False))
        for row in batch.to_pylist():
            headers = chess.pgn.Headers({k: str(v) for k, v in row.items()
                                         if k != "movetext" and v is not None})
            if not keep_headers(headers, cfg):
                continue
            stats["games_kept_headers"] += 1
            game = row_to_game(row)
            if game is None:
                continue
            sampled = sample_game(game, cfg)
            if sampled:
                stats["games_with_evals"] += 1
                rows.append(sampled)
    return {"rows": rows, **stats}


def fetch_and_process(job: tuple) -> dict:
    """Worker: download one file, sample it, delete it. ``job`` is picklable for a Pool."""
    path, cache_dir, cfg_dict, insecure = job
    cfg = SampleConfig(**cfg_dict)
    local = Path(cache_dir) / Path(path).name
    started = time.perf_counter()
    try:
        download(file_url(path), local, insecure)
        out = process_parquet(local, cfg)
    finally:
        local.unlink(missing_ok=True)
    out["file"] = path
    out["seconds"] = time.perf_counter() - started
    return out
