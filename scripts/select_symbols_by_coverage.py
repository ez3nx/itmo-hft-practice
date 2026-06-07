from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import download_binance as dl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top-N Binance symbols by 24h volume with full period bounds."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many symbols to select.",
    )
    parser.add_argument(
        "--quote-asset",
        type=str,
        default="",
        help="Override quote asset from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dl.load_config(args.config)

    quote_asset = args.quote_asset.strip().upper() if args.quote_asset.strip() else cfg.quote_asset
    top_n = int(args.top_n)
    start_ms = dl.to_unix_ms(cfg.start)
    end_ms = dl.to_unix_ms(cfg.end)
    target_last_open = end_ms - dl.KLINE_INTERVAL_MS[cfg.timeframe]

    ranked = dl.fetch_symbols_by_24h_volume(quote_asset=quote_asset)

    selected: list[str] = []
    checked: list[dict[str, Any]] = []
    for symbol in ranked:
        bounds = dl.fetch_symbol_bounds(symbol=symbol, timeframe=cfg.timeframe, end_ms=end_ms)
        if bounds is None:
            checked.append({"symbol": symbol, "bounds_ok": False, "reason": "no_bounds"})
            continue

        oldest_open_ms, latest_open_ms = bounds
        bounds_ok = oldest_open_ms <= start_ms and latest_open_ms >= target_last_open
        checked.append(
            {
                "symbol": symbol,
                "bounds_ok": bounds_ok,
                "oldest_open_ms": oldest_open_ms,
                "latest_open_ms": latest_open_ms,
            }
        )
        if bounds_ok:
            selected.append(symbol)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        raise RuntimeError(
            f"Could not find {top_n} symbols with full bounds. Found only {len(selected)}."
        )

    result = {
        "quote_asset": quote_asset,
        "timeframe": cfg.timeframe,
        "start": cfg.start,
        "end": cfg.end,
        "top_n": top_n,
        "selected_symbols": selected,
        "checked_count": len(checked),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
