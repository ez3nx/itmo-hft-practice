from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

LOGGER = logging.getLogger("chronos_eval")


@dataclass(frozen=True)
class ChronosConfig:
    seed: int
    model_path: str
    device_map: str
    local_files_only: bool
    context_length: int
    num_samples: int
    batch_size: int
    horizons: list[int]
    quantiles: list[float]
    target_prefix: str
    include_symbols: list[str]
    max_points_per_symbol_per_horizon: int
    stride_minutes: int
    processed_dir: Path
    raw_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    logs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chronos zero-shot baseline.")
    parser.add_argument("--config", type=str, default="configs/chronos.yaml", help="Path to Chronos YAML config.")
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional path to log file. If empty, writes to timestamped file in logs_dir.",
    )
    return parser.parse_args()


def load_config(path: str) -> ChronosConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ChronosConfig(
        seed=int(cfg["experiment"]["seed"]),
        model_path=str(cfg["model"]["model_path"]),
        device_map=str(cfg["model"].get("device_map", "cpu")),
        local_files_only=bool(cfg["model"].get("local_files_only", True)),
        context_length=int(cfg["model"]["context_length"]),
        num_samples=int(cfg["model"]["num_samples"]),
        batch_size=int(cfg["model"]["batch_size"]),
        horizons=[int(x) for x in cfg["targets"]["horizons"]],
        quantiles=[float(x) for x in cfg["targets"]["quantiles"]],
        target_prefix=str(cfg["targets"]["target_prefix"]),
        include_symbols=[str(x).upper() for x in cfg["targets"].get("include_symbols", [])],
        max_points_per_symbol_per_horizon=int(cfg["runtime"]["max_points_per_symbol_per_horizon"]),
        stride_minutes=int(cfg["runtime"].get("stride_minutes", 1)),
        processed_dir=Path(cfg["paths"]["processed_dir"]),
        raw_dir=Path(cfg["paths"]["raw_dir"]),
        metrics_dir=Path(cfg["paths"]["metrics_dir"]),
        predictions_dir=Path(cfg["paths"]["predictions_dir"]),
        logs_dir=Path(cfg["paths"].get("logs_dir", "artifacts/logs")),
    )


def configure_logging(logs_dir: Path, log_file: str = "") -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_file.strip():
        path = Path(log_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = logs_dir / f"chronos_eval_{ts}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    LOGGER.addHandler(sh)
    LOGGER.addHandler(fh)
    return path


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def even_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idx = np.linspace(0, len(df) - 1, num=n, dtype=int)
    return df.iloc[idx].copy()


def quantiles_to_numpy(q_output: Any) -> np.ndarray:
    """
    Normalize outputs of predict_quantiles to shape:
    [batch, prediction_length, num_quantiles]
    Supports Chronos v1 (tensor output) and Chronos-2 (list output).
    """
    if isinstance(q_output, torch.Tensor):
        arr = q_output.detach().cpu().numpy()
        if arr.ndim != 3:
            raise ValueError(f"Unexpected tensor shape from predict_quantiles: {arr.shape}")
        return arr

    if isinstance(q_output, list):
        rows: list[np.ndarray] = []
        for item in q_output:
            if isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy()
            else:
                arr = np.asarray(item)

            # Chronos-2 typically returns per-item shape (n_variates, horizon, n_quantiles).
            if arr.ndim == 3:
                if arr.shape[0] != 1:
                    raise ValueError(
                        "Multivariate outputs are not expected in this script. "
                        f"Got shape={arr.shape} for one forecast item."
                    )
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"Unexpected per-item quantile shape: {arr.shape}")
            rows.append(arr)

        if not rows:
            raise ValueError("Empty quantile list returned by pipeline.")
        return np.stack(rows, axis=0)

    raise TypeError(f"Unsupported quantile output type: {type(q_output)}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg.metrics_dir, cfg.predictions_dir, cfg.logs_dir)
    log_path = configure_logging(cfg.logs_dir, log_file=args.log_file)
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info("Config path: %s", args.config)

    model_path = Path(cfg.model_path)
    if model_path.exists():
        model_ref = str(model_path.resolve())
        LOGGER.info("Loading Chronos model from local path: %s", model_ref)
    else:
        model_ref = cfg.model_path
        LOGGER.info("Loading Chronos model from HuggingFace id: %s", model_ref)
        if cfg.local_files_only:
            raise FileNotFoundError(
                f"Model path does not exist locally and local_files_only=true: {cfg.model_path}"
            )

    pipeline = BaseChronosPipeline.from_pretrained(
        model_ref,
        device_map=cfg.device_map,
        local_files_only=cfg.local_files_only,
    )
    LOGGER.info("Chronos loaded successfully.")
    LOGGER.info("Pipeline class: %s", pipeline.__class__.__name__)
    model_tag = Path(model_ref).name.replace(" ", "_")

    target_cols = [f"{cfg.target_prefix}{h}" for h in cfg.horizons]
    read_cols = ["open_time", "symbol", "close"] + target_cols
    valid_df = pd.read_parquet(cfg.processed_dir / "valid.parquet", columns=read_cols)
    valid_df["open_time"] = pd.to_datetime(valid_df["open_time"], utc=True)
    valid_df = valid_df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    if cfg.include_symbols:
        valid_df = valid_df[valid_df["symbol"].isin(cfg.include_symbols)].copy()
        LOGGER.info("Symbol filter applied: %s", cfg.include_symbols)
    symbols = sorted(valid_df["symbol"].unique().tolist())
    LOGGER.info("Valid rows=%d symbols=%d", len(valid_df), len(symbols))

    raw_close_by_symbol: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for symbol in symbols:
        raw_path = cfg.raw_dir / f"{symbol}.parquet"
        raw = pd.read_parquet(raw_path, columns=["open_time", "close"]).sort_values("open_time")
        raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
        raw_close_by_symbol[symbol] = (pd.DatetimeIndex(raw["open_time"]), raw["close"].to_numpy(dtype=np.float64))

    all_metrics: list[dict[str, Any]] = []
    horizon_summary: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []

    for h in cfg.horizons:
        target_col = f"{cfg.target_prefix}{h}"
        LOGGER.info("Horizon %d: preparing contexts.", h)
        horizon_rows = []

        for symbol in symbols:
            sym = valid_df[["open_time", "symbol", "close", target_col]].loc[valid_df["symbol"] == symbol].copy()
            sym = sym.dropna(subset=[target_col])
            if cfg.stride_minutes > 1:
                sym = sym.iloc[:: cfg.stride_minutes].copy()
            sym = even_sample(sym, cfg.max_points_per_symbol_per_horizon)

            raw_index, raw_close = raw_close_by_symbol[symbol]
            pos = raw_index.get_indexer(sym["open_time"].to_numpy())
            keep = pos >= (cfg.context_length - 1)
            if not np.any(keep):
                continue

            sym = sym.iloc[np.where(keep)[0]].reset_index(drop=True)
            pos = pos[keep]
            contexts = np.stack(
                [raw_close[p - cfg.context_length + 1 : p + 1].astype(np.float32) for p in pos],
                axis=0,
            )
            sym["raw_pos"] = pos
            sym["current_close"] = sym["close"].astype(np.float64)
            sym["y_true"] = sym[target_col].astype(np.float64)
            sym["context_np"] = list(contexts)
            horizon_rows.append(sym[["open_time", "symbol", "current_close", "y_true", "context_np"]])

        if not horizon_rows:
            LOGGER.warning("Horizon %d: no rows to evaluate.", h)
            continue

        eval_df = pd.concat(horizon_rows, ignore_index=True)
        LOGGER.info("Horizon %d: eval rows=%d", h, len(eval_df))

        pred_q = {q: np.zeros(len(eval_df), dtype=np.float64) for q in cfg.quantiles}
        start_infer = time.perf_counter()
        use_num_samples = True
        for start in range(0, len(eval_df), cfg.batch_size):
            end = min(start + cfg.batch_size, len(eval_df))
            batch_context = np.stack(eval_df["context_np"].iloc[start:end].to_numpy())
            # A list of 1D tensors is accepted by both Chronos and Chronos-2 APIs.
            batch_inputs = [torch.from_numpy(x) for x in batch_context]
            with torch.no_grad():
                if use_num_samples:
                    try:
                        q_output, _ = pipeline.predict_quantiles(
                            batch_inputs,
                            prediction_length=h,
                            quantile_levels=cfg.quantiles,
                            num_samples=cfg.num_samples,
                        )
                    except TypeError as exc:
                        msg = str(exc)
                        if "num_samples" not in msg and "Unexpected keyword arguments" not in msg:
                            raise
                        use_num_samples = False
                        LOGGER.warning(
                            "Pipeline %s does not support num_samples argument. "
                            "Retrying without num_samples for the rest of this run.",
                            pipeline.__class__.__name__,
                        )
                        q_output, _ = pipeline.predict_quantiles(
                            batch_inputs,
                            prediction_length=h,
                            quantile_levels=cfg.quantiles,
                        )
                else:
                    q_output, _ = pipeline.predict_quantiles(
                        batch_inputs,
                        prediction_length=h,
                        quantile_levels=cfg.quantiles,
                    )
            # q_tensor shape: [batch, prediction_length, num_quantiles]
            q_tensor = quantiles_to_numpy(q_output)
            q_step = q_tensor[:, h - 1, :]
            current_close = eval_df["current_close"].iloc[start:end].to_numpy(dtype=np.float64)
            for q_idx, q in enumerate(cfg.quantiles):
                pred_price = np.clip(q_step[:, q_idx], 1e-12, None)
                pred_q[q][start:end] = np.log(pred_price) - np.log(np.clip(current_close, 1e-12, None))
        infer_seconds = time.perf_counter() - start_infer

        y_true = eval_df["y_true"].to_numpy(dtype=np.float64)
        metric_rows_for_h = []
        for q in cfg.quantiles:
            y_pred = pred_q[q]
            row: dict[str, Any] = {
                "model": model_tag,
                "horizon": h,
                "quantile": q,
                "pinball_valid": float(mean_pinball_loss(y_true, y_pred, alpha=q)),
                "pinball_zero_valid": float(mean_pinball_loss(y_true, np.zeros_like(y_true), alpha=q)),
                "infer_seconds_total": float(infer_seconds),
                "infer_rows_per_sec": float(len(eval_df) / max(infer_seconds, 1e-12)),
                "valid_rows_used": int(len(eval_df)),
            }
            row["pinball_improvement_vs_zero_pct"] = float(
                100.0 * (row["pinball_zero_valid"] - row["pinball_valid"]) / max(row["pinball_zero_valid"], 1e-12)
            )
            if abs(q - 0.5) < 1e-9:
                row["mae_q50_valid"] = float(mean_absolute_error(y_true, y_pred))
                row["rmse_q50_valid"] = rmse(y_true, y_pred)
                row["direction_acc_q50_valid"] = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
            metric_rows_for_h.append(row)
            all_metrics.append(row)

        pred_df = eval_df[["open_time", "symbol", "y_true"]].copy()
        for q in cfg.quantiles:
            pred_df[f"pred_q{int(round(q * 100)):02d}"] = pred_q[q]
        pred_df["horizon"] = h
        pred_df.to_parquet(cfg.predictions_dir / f"pred_valid_h{h}.parquet", index=False)
        all_preds.append(pred_df)

        q10 = pred_q.get(0.1)
        q50 = pred_q.get(0.5)
        q90 = pred_q.get(0.9)
        if q10 is not None and q50 is not None and q90 is not None:
            horizon_summary.append(
                {
                    "model": model_tag,
                    "horizon": h,
                    "interval_80_coverage_valid": float(np.mean((y_true >= q10) & (y_true <= q90))),
                    "interval_80_width_mean_valid": float(np.mean(q90 - q10)),
                    "q50_mae_valid": float(mean_absolute_error(y_true, q50)),
                    "q50_rmse_valid": rmse(y_true, q50),
                    "q50_direction_acc_valid": float(np.mean(np.sign(q50) == np.sign(y_true))),
                    "infer_rows_per_sec": float(len(eval_df) / max(infer_seconds, 1e-12)),
                    "valid_rows_used": int(len(eval_df)),
                }
            )
        LOGGER.info("Horizon %d: done.", h)

    metrics_df = pd.DataFrame(all_metrics).sort_values(["horizon", "quantile"]).reset_index(drop=True)
    summary_df = pd.DataFrame(horizon_summary).sort_values("horizon").reset_index(drop=True)
    metrics_df.to_csv(cfg.metrics_dir / "chronos_metrics.csv", index=False)
    summary_df.to_csv(cfg.metrics_dir / "chronos_horizon_summary.csv", index=False)
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_parquet(cfg.predictions_dir / "pred_valid_all_horizons.parquet", index=False)

    LOGGER.info("Chronos evaluation complete.")
    LOGGER.info("Saved metrics: %s", (cfg.metrics_dir / "chronos_metrics.csv").resolve())
    LOGGER.info("Saved summary: %s", (cfg.metrics_dir / "chronos_horizon_summary.csv").resolve())
    LOGGER.info("Saved predictions dir: %s", cfg.predictions_dir.resolve())
    print("Chronos evaluation complete.")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    if not summary_df.empty:
        print("\nChronos horizon summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
