from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from chronos import ChronosPipeline


@dataclass(frozen=True)
class FinetuneConfig:
    model_id: str
    processed_dir: Path
    output_dir: Path
    symbols: list[str]
    context_length: int
    prediction_length: int
    train_stride: int
    valid_stride: int
    max_windows_per_symbol_train: int
    max_windows_per_symbol_valid: int
    tokenizer_batch_size: int
    seed: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    fp16: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Chronos T5 fine-tune on selected symbols.")
    parser.add_argument("--model-id", type=str, default="amazon/chronos-t5-tiny")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="models/chronos-t5-tiny-ft-kaggle")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--valid-stride", type=int, default=15)
    parser.add_argument("--max-windows-per-symbol-train", type=int, default=12000)
    parser.add_argument("--max-windows-per-symbol-valid", type=int, default=3000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        model_id=str(args.model_id),
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        symbols=[s.upper() for s in args.symbols],
        context_length=int(args.context_length),
        prediction_length=int(args.prediction_length),
        train_stride=int(args.train_stride),
        valid_stride=int(args.valid_stride),
        max_windows_per_symbol_train=int(args.max_windows_per_symbol_train),
        max_windows_per_symbol_valid=int(args.max_windows_per_symbol_valid),
        tokenizer_batch_size=int(args.tokenizer_batch_size),
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        warmup_steps=int(args.warmup_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        eval_steps=int(args.eval_steps),
        save_steps=int(args.save_steps),
        logging_steps=int(args.logging_steps),
        fp16=bool(args.fp16),
    )


def even_sample_2d(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = np.linspace(0, len(x) - 1, num=n, dtype=int)
    return x[idx]


def load_close_series(split_path: Path, symbols: list[str]) -> pd.DataFrame:
    df = (
        pl.scan_parquet(str(split_path))
        .filter(pl.col("symbol").is_in(symbols))
        .select(["open_time", "symbol", "close"])
        .collect()
        .to_pandas()
    )
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return df


def build_windows_from_close(
    close_values: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(close_values) < context_length + prediction_length + 1:
        return (
            np.zeros((0, context_length), dtype=np.float32),
            np.zeros((0, prediction_length), dtype=np.float32),
        )

    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    max_end = len(close_values) - prediction_length
    for end_idx in range(context_length, max_end + 1, max(1, stride)):
        contexts.append(close_values[end_idx - context_length : end_idx])
        futures.append(close_values[end_idx : end_idx + prediction_length])

    return np.asarray(contexts, dtype=np.float32), np.asarray(futures, dtype=np.float32)


def build_split_windows(
    split_df: pd.DataFrame,
    symbols: list[str],
    context_length: int,
    prediction_length: int,
    stride: int,
    max_windows_per_symbol: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_ctx: list[np.ndarray] = []
    all_fut: list[np.ndarray] = []
    for symbol in symbols:
        chunk = split_df.loc[split_df["symbol"] == symbol, "close"].to_numpy(dtype=np.float64)
        ctx, fut = build_windows_from_close(
            close_values=chunk,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
        )
        if len(ctx) == 0:
            continue
        if len(ctx) > max_windows_per_symbol:
            idx = np.linspace(0, len(ctx) - 1, num=max_windows_per_symbol, dtype=int)
            ctx = ctx[idx]
            fut = fut[idx]
        all_ctx.append(ctx)
        all_fut.append(fut)

    if not all_ctx:
        raise RuntimeError("No windows were generated. Decrease context/prediction lengths or adjust symbols.")
    return np.concatenate(all_ctx, axis=0), np.concatenate(all_fut, axis=0)


class TokenizedSeq2SeqDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def tokenize_windows(
    pipeline: ChronosPipeline,
    contexts: np.ndarray,
    futures: np.ndarray,
    batch_size: int,
) -> TokenizedSeq2SeqDataset:
    tokenizer = pipeline.tokenizer
    ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for start in range(0, len(contexts), batch_size):
        end = min(start + batch_size, len(contexts))
        context_batch = torch.from_numpy(contexts[start:end])
        future_batch = torch.from_numpy(futures[start:end])

        input_ids, attention_mask, tokenizer_state = tokenizer.context_input_transform(context_batch)
        label_ids, label_mask = tokenizer.label_input_transform(future_batch, tokenizer_state)

        labels = label_ids.clone()
        labels[~label_mask] = -100

        ids_list.append(input_ids.to(dtype=torch.long, device="cpu"))
        mask_list.append(attention_mask.to(dtype=torch.long, device="cpu"))
        labels_list.append(labels.to(dtype=torch.long, device="cpu"))

    input_ids = torch.cat(ids_list, dim=0)
    attention_mask = torch.cat(mask_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return TokenizedSeq2SeqDataset(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_close_series(cfg.processed_dir / "train.parquet", cfg.symbols)
    valid_df = load_close_series(cfg.processed_dir / "valid.parquet", cfg.symbols)

    train_ctx, train_fut = build_split_windows(
        split_df=train_df,
        symbols=cfg.symbols,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        stride=cfg.train_stride,
        max_windows_per_symbol=cfg.max_windows_per_symbol_train,
    )
    valid_ctx, valid_fut = build_split_windows(
        split_df=valid_df,
        symbols=cfg.symbols,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        stride=cfg.valid_stride,
        max_windows_per_symbol=cfg.max_windows_per_symbol_valid,
    )

    # Keep train size manageable for quick Kaggle runs.
    train_ctx = even_sample_2d(train_ctx, n=len(train_ctx))
    train_fut = even_sample_2d(train_fut, n=len(train_fut))

    pipeline = ChronosPipeline.from_pretrained(
        cfg.model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=False,
    )
    hf_model = pipeline.model.model

    train_ds = tokenize_windows(pipeline, train_ctx, train_fut, batch_size=cfg.tokenizer_batch_size)
    valid_ds = tokenize_windows(pipeline, valid_ctx, valid_fut, batch_size=cfg.tokenizer_batch_size)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        run_name="chronos_t5_tiny_ft_minimal",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=hf_model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))

    metadata = {
        "model_id": cfg.model_id,
        "symbols": cfg.symbols,
        "context_length": cfg.context_length,
        "prediction_length": cfg.prediction_length,
        "train_windows": len(train_ds),
        "valid_windows": len(valid_ds),
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.learning_rate,
        "fp16": cfg.fp16,
    }
    with open(cfg.output_dir / "finetune_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Fine-tuning complete. Model saved to: {cfg.output_dir.resolve()}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
