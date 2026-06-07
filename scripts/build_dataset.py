from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from math import pi
from pathlib import Path

import polars as pl
import yaml


@dataclass(frozen=True)
class FeatureConfig:
    horizons: list[int]
    target_prefix: str
    return_lags: list[int]
    volume_lags: list[int]
    rolling_vol_windows: list[int]
    rolling_mean_windows: list[int]
    range_windows: list[int]
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    raw_dir: Path
    processed_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minute-level features and targets.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/features.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path: str) -> FeatureConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return FeatureConfig(
        horizons=[int(x) for x in cfg["targets"]["horizons"]],
        target_prefix=str(cfg["targets"]["target_prefix"]),
        return_lags=[int(x) for x in cfg["features"]["return_lags"]],
        volume_lags=[int(x) for x in cfg["features"]["volume_lags"]],
        rolling_vol_windows=[int(x) for x in cfg["features"]["rolling_vol_windows"]],
        rolling_mean_windows=[int(x) for x in cfg["features"]["rolling_mean_windows"]],
        range_windows=[int(x) for x in cfg["features"]["range_windows"]],
        train_start=cfg["split"]["train_start"],
        train_end=cfg["split"]["train_end"],
        valid_start=cfg["split"]["valid_start"],
        valid_end=cfg["split"]["valid_end"],
        raw_dir=Path(cfg["paths"]["raw_dir"]),
        processed_dir=Path(cfg["paths"]["processed_dir"]),
    )


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_symbol_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    df = df.sort("open_time").unique(subset=["open_time"], keep="last").sort("open_time")

    minute_of_day = pl.col("open_time").dt.hour() * 60 + pl.col("open_time").dt.minute()
    day_of_week = pl.col("open_time").dt.weekday()
    pi_expr = pl.lit(2.0 * pi)

    df = df.with_columns(
        [
            pl.col("close").clip(lower_bound=1e-12).log().cast(pl.Float32).alias("log_close"),
            pl.col("volume").clip(lower_bound=0.0).log1p().cast(pl.Float32).alias("log_volume"),
        ]
    ).with_columns(
        [
            pl.col("log_close").diff().cast(pl.Float32).alias("logret_1"),
            ((pl.col("high") - pl.col("low")) / pl.col("close").clip(lower_bound=1e-12))
            .cast(pl.Float32)
            .alias("intrabar_range"),
            ((pl.col("close") - pl.col("open")) / pl.col("open").clip(lower_bound=1e-12))
            .cast(pl.Float32)
            .alias("close_open_spread"),
            (pi_expr * minute_of_day / pl.lit(1440.0)).sin().cast(pl.Float32).alias("minute_sin"),
            (pi_expr * minute_of_day / pl.lit(1440.0)).cos().cast(pl.Float32).alias("minute_cos"),
            (pi_expr * day_of_week / pl.lit(7.0)).sin().cast(pl.Float32).alias("dow_sin"),
            (pi_expr * day_of_week / pl.lit(7.0)).cos().cast(pl.Float32).alias("dow_cos"),
        ]
    )

    for lag in cfg.return_lags:
        df = df.with_columns(pl.col("logret_1").shift(lag).cast(pl.Float32).alias(f"logret_lag_{lag}"))
    for lag in cfg.volume_lags:
        df = df.with_columns(pl.col("log_volume").shift(lag).cast(pl.Float32).alias(f"logvol_lag_{lag}"))
    for window in cfg.rolling_vol_windows:
        df = df.with_columns(
            pl.col("logret_1")
            .rolling_std(window_size=window, min_samples=window)
            .cast(pl.Float32)
            .alias(f"logret_std_{window}")
        )
    for window in cfg.rolling_mean_windows:
        df = df.with_columns(
            pl.col("logret_1")
            .rolling_mean(window_size=window, min_samples=window)
            .cast(pl.Float32)
            .alias(f"logret_mean_{window}")
        )
    for window in cfg.range_windows:
        df = df.with_columns(
            pl.col("intrabar_range")
            .rolling_mean(window_size=window, min_samples=window)
            .cast(pl.Float32)
            .alias(f"range_mean_{window}")
        )
    for h in cfg.horizons:
        df = df.with_columns(
            (pl.col("log_close").shift(-h) - pl.col("log_close"))
            .cast(pl.Float32)
            .alias(f"{cfg.target_prefix}{h}")
        )
    return df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    target_cols = [f"{cfg.target_prefix}{h}" for h in cfg.horizons]
    train_start = parse_utc(cfg.train_start)
    train_end = parse_utc(cfg.train_end)
    valid_start = parse_utc(cfg.valid_start)
    valid_end = parse_utc(cfg.valid_end)
    train_parts_dir = cfg.processed_dir / "train_parts"
    valid_parts_dir = cfg.processed_dir / "valid_parts"
    train_parts_dir.mkdir(parents=True, exist_ok=True)
    valid_parts_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted([p for p in cfg.raw_dir.glob("*.parquet") if not p.name.startswith("_")])
    if not raw_paths:
        raise FileNotFoundError(f"No parquet files found in: {cfg.raw_dir.resolve()}")

    row_count_train = 0
    row_count_valid = 0
    symbols: list[str] = []
    per_symbol_rows: dict[str, dict[str, int]] = {}

    target_non_null = pl.all_horizontal([pl.col(c).is_not_null() for c in target_cols])
    for path in raw_paths:
        symbol = path.stem.upper()
        symbols.append(symbol)

        df = pl.read_parquet(path)
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

        feat_df = build_symbol_features(df=df, cfg=cfg).filter(target_non_null)
        train_df = feat_df.filter((pl.col("open_time") >= train_start) & (pl.col("open_time") <= train_end))
        valid_df = feat_df.filter((pl.col("open_time") >= valid_start) & (pl.col("open_time") <= valid_end))

        train_df.write_parquet(train_parts_dir / f"{symbol}.parquet")
        valid_df.write_parquet(valid_parts_dir / f"{symbol}.parquet")

        n_train = int(train_df.height)
        n_valid = int(valid_df.height)
        row_count_train += n_train
        row_count_valid += n_valid
        per_symbol_rows[symbol] = {"train_rows": n_train, "valid_rows": n_valid}

    train_scan = pl.scan_parquet(str(train_parts_dir / "*.parquet"))
    valid_scan = pl.scan_parquet(str(valid_parts_dir / "*.parquet"))
    train_scan.sink_parquet(str(cfg.processed_dir / "train.parquet"))
    valid_scan.sink_parquet(str(cfg.processed_dir / "valid.parquet"))

    row_count_full = row_count_train + row_count_valid

    metadata = {
        "target_columns": target_cols,
        "row_count_full": row_count_full,
        "row_count_train": row_count_train,
        "row_count_valid": row_count_valid,
        "symbols": sorted(symbols),
        "per_symbol_rows": per_symbol_rows,
    }
    with open(cfg.processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Dataset build complete.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
