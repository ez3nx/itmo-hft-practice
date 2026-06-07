from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from tqdm import tqdm

BINANCE_REST = "https://api.binance.com"
KLINE_INTERVAL_MS = {"1m": 60_000}
STABLE_EXCLUDE_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
EXCLUDED_BASE_ASSETS = {
    "USDT",
    "USDC",
    "FDUSD",
    "BUSD",
    "TUSD",
    "USDP",
    "DAI",
    "EUR",
    "EURC",
}
LOGGER = logging.getLogger("binance_downloader")


@dataclass(frozen=True)
class DownloadConfig:
    quote_asset: str
    top_n_symbols: int
    fixed_symbols: list[str]
    timeframe: str
    start: str
    end: str
    sleep_seconds: float
    limit_per_request: int
    strict_history_coverage: bool
    min_coverage_ratio: float
    log_every_batches: int
    skip_existing: bool
    raw_dir: Path
    logs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download minute Binance OHLCV data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated list of symbols. Overrides top-N selection.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional path to log file. If empty, uses timestamped file under logs_dir.",
    )
    return parser.parse_args()


def load_config(path: str) -> DownloadConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fixed_symbols = [str(x).upper() for x in cfg["binance"].get("symbols", [])]
    return DownloadConfig(
        quote_asset=cfg["binance"]["quote_asset"],
        top_n_symbols=int(cfg["binance"]["top_n_symbols"]),
        fixed_symbols=fixed_symbols,
        timeframe=cfg["binance"]["timeframe"],
        start=cfg["binance"]["start"],
        end=cfg["binance"]["end"],
        sleep_seconds=float(cfg["binance"]["sleep_seconds"]),
        limit_per_request=int(cfg["binance"]["limit_per_request"]),
        strict_history_coverage=bool(cfg["binance"].get("strict_history_coverage", True)),
        min_coverage_ratio=float(cfg["binance"].get("min_coverage_ratio", 0.995)),
        log_every_batches=int(cfg["binance"].get("log_every_batches", 50)),
        skip_existing=bool(cfg["binance"].get("skip_existing", True)),
        raw_dir=Path(cfg["paths"]["raw_dir"]),
        logs_dir=Path(cfg["paths"].get("logs_dir", "artifacts/logs")),
    )


def configure_logging(logs_dir: Path, log_file: str = "") -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_file.strip():
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"download_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return log_path


def to_unix_ms(iso_ts: str) -> int:
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def request_json(endpoint: str, params: dict[str, Any] | None = None, retries: int = 5) -> Any:
    url = f"{BINANCE_REST}{endpoint}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            sleep_s = 1.0 * (attempt + 1)
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {retries} retries: {url}") from last_error


def is_allowed_symbol(symbol: str, quote_asset: str) -> bool:
    if not symbol.endswith(quote_asset):
        return False
    base = symbol[: -len(quote_asset)]
    if base.endswith(STABLE_EXCLUDE_SUFFIXES):
        return False
    if base in EXCLUDED_BASE_ASSETS:
        return False
    return True


def fetch_symbols_by_24h_volume(quote_asset: str) -> list[str]:
    tickers = request_json("/api/v3/ticker/24hr")
    exchange_info = request_json("/api/v3/exchangeInfo")
    tradable = {
        s["symbol"]
        for s in exchange_info["symbols"]
        if s.get("status") == "TRADING" and s.get("isSpotTradingAllowed", True)
    }

    rows = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol not in tradable:
            continue
        if not is_allowed_symbol(symbol, quote_asset=quote_asset):
            continue
        rows.append((symbol, float(t.get("quoteVolume", 0.0))))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows]


def fetch_top_symbols(quote_asset: str, top_n: int) -> list[str]:
    symbols = fetch_symbols_by_24h_volume(quote_asset=quote_asset)
    if len(symbols) < top_n:
        raise RuntimeError(f"Found only {len(symbols)} eligible symbols for quote asset {quote_asset}.")
    return symbols[:top_n]


def fetch_symbol_bounds(symbol: str, timeframe: str, end_ms: int) -> tuple[int, int] | None:
    first_payload = request_json(
        "/api/v3/klines",
        params={"symbol": symbol, "interval": timeframe, "startTime": 0, "limit": 1},
    )
    last_payload = request_json(
        "/api/v3/klines",
        params={"symbol": symbol, "interval": timeframe, "endTime": end_ms, "limit": 1},
    )
    if not first_payload or not last_payload:
        return None
    oldest_open_ms = int(first_payload[0][0])
    latest_open_ms = int(last_payload[-1][0])
    return oldest_open_ms, latest_open_ms


def fetch_klines(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit_per_request: int,
    sleep_seconds: float,
    log_every_batches: int = 50,
) -> pd.DataFrame:
    if timeframe not in KLINE_INTERVAL_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    step_ms = KLINE_INTERVAL_MS[timeframe]

    all_rows: list[list[Any]] = []
    cursor = start_ms
    batch_count = 0

    while cursor < end_ms:
        payload = request_json(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": timeframe,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit_per_request,
            },
        )
        if not payload:
            break
        all_rows.extend(payload)
        batch_count += 1
        last_open = int(payload[-1][0])
        next_cursor = last_open + step_ms
        if log_every_batches > 0 and batch_count % log_every_batches == 0:
            last_ts = datetime.fromtimestamp(last_open / 1000, tz=timezone.utc).isoformat()
            LOGGER.info(
                "PROGRESS symbol=%s batches=%d rows=%d last_open_utc=%s",
                symbol,
                batch_count,
                len(all_rows),
                last_ts,
            )
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(sleep_seconds)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df = df.drop(columns=["ignore"])
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype("Int64")
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df


def save_symbol_data(df: pd.DataFrame, out_dir: Path, symbol: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    df.to_parquet(out_path, index=False)


def save_manifest(symbols: list[str], out_dir: Path, reports: list[dict[str, Any]] | None = None) -> None:
    if reports:
        manifest = pd.DataFrame(reports)
    else:
        manifest = pd.DataFrame({"symbol": symbols})
    manifest["downloaded_at_utc"] = datetime.now(timezone.utc)
    manifest.to_csv(out_dir / "_manifest.csv", index=False)


def assess_coverage(
    df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
    timeframe: str,
    min_coverage_ratio: float,
) -> dict[str, Any]:
    if timeframe not in KLINE_INTERVAL_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    step_ms = KLINE_INTERVAL_MS[timeframe]

    expected_rows = max(0, ((end_ms - start_ms) // step_ms) + 1)
    if df.empty or expected_rows == 0:
        return {
            "coverage_ratio": 0.0,
            "has_full_bounds": False,
            "expected_rows": expected_rows,
            "actual_rows": int(len(df)),
            "coverage_ok": False,
        }

    work = df.sort_values("open_time_ms").drop_duplicates(subset=["open_time_ms"], keep="last")
    in_range = work[(work["open_time_ms"] >= start_ms) & (work["open_time_ms"] <= end_ms)]

    actual_rows = int(len(in_range))
    coverage_ratio = actual_rows / expected_rows if expected_rows > 0 else 0.0
    first_open = int(in_range["open_time_ms"].min()) if actual_rows else None
    last_open = int(in_range["open_time_ms"].max()) if actual_rows else None
    target_last_open = end_ms - step_ms

    has_full_bounds = (
        first_open is not None
        and last_open is not None
        and first_open <= start_ms
        and last_open >= target_last_open
    )
    coverage_ok = has_full_bounds and (coverage_ratio >= min_coverage_ratio)
    return {
        "coverage_ratio": float(coverage_ratio),
        "has_full_bounds": bool(has_full_bounds),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "coverage_ok": bool(coverage_ok),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    log_path = configure_logging(logs_dir=cfg.logs_dir, log_file=args.log_file)
    start_ms = to_unix_ms(cfg.start)
    end_ms = to_unix_ms(cfg.end)
    target_last_open = end_ms - KLINE_INTERVAL_MS[cfg.timeframe]
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info(
        "Config timeframe=%s start=%s end=%s strict=%s min_coverage_ratio=%.4f skip_existing=%s",
        cfg.timeframe,
        cfg.start,
        cfg.end,
        cfg.strict_history_coverage,
        cfg.min_coverage_ratio,
        cfg.skip_existing,
    )

    source_mode = "dynamic_top_n"
    if args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        source_mode = "cli_symbols"
    elif cfg.fixed_symbols:
        symbols = cfg.fixed_symbols
        source_mode = "config_symbols"
    else:
        symbols = fetch_symbols_by_24h_volume(cfg.quote_asset)
        source_mode = "dynamic_top_n"

    reports: list[dict[str, Any]] = []
    accepted: list[str] = []
    max_required = cfg.top_n_symbols if source_mode == "dynamic_top_n" else len(symbols)
    LOGGER.info(
        "Selecting symbols mode=%s strict=%s required=%d",
        source_mode,
        cfg.strict_history_coverage,
        max_required,
    )

    for symbol in tqdm(symbols, desc="Symbols"):
        if source_mode == "dynamic_top_n" and len(accepted) >= cfg.top_n_symbols:
            break

        out_path = cfg.raw_dir / f"{symbol}.parquet"
        if cfg.skip_existing and out_path.exists():
            LOGGER.info("SKIP symbol=%s reason=existing_file path=%s", symbol, out_path)
            accepted.append(symbol)
            reports.append({"symbol": symbol, "status": "skipped_existing"})
            continue

        LOGGER.info("START symbol=%s", symbol)
        bounds = fetch_symbol_bounds(symbol=symbol, timeframe=cfg.timeframe, end_ms=end_ms)
        if bounds is None:
            LOGGER.warning("SKIP symbol=%s reason=no_bounds", symbol)
            reports.append({"symbol": symbol, "status": "skipped_no_bounds"})
            continue
        oldest_open_ms, latest_open_ms = bounds

        has_precheck_bounds = oldest_open_ms <= start_ms and latest_open_ms >= target_last_open
        if cfg.strict_history_coverage and not has_precheck_bounds:
            LOGGER.warning(
                "SKIP symbol=%s reason=precheck_bounds oldest_ms=%d latest_ms=%d",
                symbol,
                oldest_open_ms,
                latest_open_ms,
            )
            reports.append(
                {
                    "symbol": symbol,
                    "status": "skipped_precheck_bounds",
                    "oldest_open_ms": oldest_open_ms,
                    "latest_open_ms": latest_open_ms,
                    "coverage_ratio": 0.0,
                }
            )
            continue

        df = fetch_klines(
            symbol=symbol,
            timeframe=cfg.timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            limit_per_request=cfg.limit_per_request,
            sleep_seconds=cfg.sleep_seconds,
            log_every_batches=cfg.log_every_batches,
        )
        if df.empty:
            LOGGER.warning("SKIP symbol=%s reason=empty", symbol)
            reports.append({"symbol": symbol, "status": "skipped_empty", "coverage_ratio": 0.0})
            continue

        coverage = assess_coverage(
            df=df,
            start_ms=start_ms,
            end_ms=end_ms,
            timeframe=cfg.timeframe,
            min_coverage_ratio=cfg.min_coverage_ratio,
        )
        if cfg.strict_history_coverage and not coverage["coverage_ok"]:
            LOGGER.warning(
                "SKIP symbol=%s reason=coverage ratio=%.6f expected=%d actual=%d full_bounds=%s",
                symbol,
                coverage["coverage_ratio"],
                coverage["expected_rows"],
                coverage["actual_rows"],
                coverage["has_full_bounds"],
            )
            reports.append(
                {
                    "symbol": symbol,
                    "status": "skipped_coverage",
                    "oldest_open_ms": oldest_open_ms,
                    "latest_open_ms": latest_open_ms,
                    **coverage,
                }
            )
            continue

        save_symbol_data(df, cfg.raw_dir, symbol=symbol)
        accepted.append(symbol)
        LOGGER.info(
            "SAVED symbol=%s coverage_ratio=%.6f rows=%d",
            symbol,
            coverage["coverage_ratio"],
            coverage["actual_rows"],
        )
        reports.append(
            {
                "symbol": symbol,
                "status": "saved",
                "oldest_open_ms": oldest_open_ms,
                "latest_open_ms": latest_open_ms,
                **coverage,
            }
        )

    save_manifest(symbols=accepted, out_dir=cfg.raw_dir, reports=reports)

    if source_mode == "dynamic_top_n" and len(accepted) < cfg.top_n_symbols:
        raise RuntimeError(
            f"Could not collect {cfg.top_n_symbols} symbols with required history. "
            f"Collected only {len(accepted)}."
        )
    if source_mode in {"cli_symbols", "config_symbols"} and cfg.strict_history_coverage:
        if len(accepted) < len(symbols):
            missing = [s for s in symbols if s not in accepted]
            raise RuntimeError(
                "Some requested symbols do not satisfy strict history coverage: "
                f"{missing}"
            )

    LOGGER.info("Done. Saved %d symbols to: %s", len(accepted), cfg.raw_dir.resolve())
    LOGGER.info("Accepted symbols: %s", accepted)


if __name__ == "__main__":
    main()
