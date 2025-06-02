#!/usr/bin/env python3
"""
semi‑structured 2 : 4 pruning

Usage
-----
```bash
python train_bigvul_modernbert_pruned.py \
       --cfg config.yaml \
       --out results/ \
       --prune-amount 0.5          # 50 % 2:4 sparsity (default)
```
`--prune-amount` must be **≤0.5** for the 2‑out‑of‑4 pattern.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from datetime import datetime
from typing import Iterable

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch.ao.pruning import AmperePruningMethod  # NEW
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
from codecarbon import EmissionsTracker

# --------------------------------------------------------------------------- #
# 1. Logging
hf_logging.set_verbosity_info()
logger = hf_logging.get_logger(__name__)

# --------------------------------------------------------------------------- #
# 2. Dataset helpers

def load_bigvul():
    ds = load_dataset("bstee615/bigvul")
    return ds["train"], ds["validation"], ds["test"]


def prep_dataset(dataset, tok, text_col: str, label_col: str, max_len: int):
    def tok_fn(batch):
        enc = tok(batch[text_col], truncation=True, max_length=max_len)
        enc["labels"] = np.int64(batch[label_col])
        return enc

    keep_cols = [text_col, label_col]
    ds_tok = dataset.map(tok_fn, remove_columns=[c for c in dataset.column_names if c not in keep_cols], batched=False)
    ds_tok.set_format("torch")
    return ds_tok

# --------------------------------------------------------------------------- #
# 3. Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="weighted")}

# --------------------------------------------------------------------------- #
# 4. Pruning helpers (Ampere 2:4)

def iter_prunable_modules(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """Yield all Linear layers that accept 2 : 4 pruning."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            yield m


def prune_model_ampere(model: torch.nn.Module, amount: float):
    """Apply global 2 : 4 *structured* pruning.

    Parameters
    ----------
    model  : ModernBertForSequenceClassification
    amount : fraction of weights to prune (≤0.5 for 2‑out‑of‑4 pattern)
    """
    params_to_prune = [(m, "weight") for m in iter_prunable_modules(model)]
    logger.info(f"Applying 2:4 Ampere pruning globally – sparsity {amount:.0%}")

    AmperePruningMethod.apply_global(params_to_prune, amount=amount)

    # Convert masks to permanent pruning so state_dict is clean
    for mod, _ in params_to_prune:
        prune.remove(mod, "weight")

    total = sum(p.numel() for p in model.parameters())
    dense = sum(torch.count_nonzero(p) for p in model.parameters()).item()
    logger.info(f"Effective density after pruning: {dense/total:.2%}")

# --------------------------------------------------------------------------- #
# 5. Main pipeline with CodeCarbon tasks

def main(cfg_path: Path, output_root: Path, prune_amount: float):
    cfg = OmegaConf.load(cfg_path)

    run_name = f"ampere_pruned_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(project_name="modernbert_ampere_train",
                               output_dir=str(out_dir), log_level="error",
                               save_to_file=True, measure_power_secs=1.0,
                               tracking_mode="process", gpu_ids=[0])
    tracker.start()

    try:
        # 1. Load dataset
        tracker.start_task("load_dataset")
        train_raw, val_raw, test_raw = load_bigvul()
        if cfg.dummy_mode.enabled:
            n = cfg.dummy_mode.sample_size
            train_raw, val_raw, test_raw = (train_raw.select(range(n)), val_raw.select(range(n//2)), test_raw.select(range(n//2)))
        tracker.stop_task()

        # 2. Tokenise
        tracker.start_task("tokenize_dataset")
        tok = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=False)
        vcfg = cfg.data.versions.default
        train_ds = prep_dataset(train_raw, tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        val_ds   = prep_dataset(val_raw,   tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        test_ds  = prep_dataset(test_raw,  tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        collator = DataCollatorWithPadding(tok, return_tensors="pt")
        tracker.stop_task()

        # 3. Load model
        tracker.start_task("load_model")
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, num_labels=cfg.model.num_labels, attn_implementation="eager")
        tracker.stop_task()

        # 4. Structured pruning (Ampere)
        tracker.start_task("prune_model")
        if prune_amount > 0:
            prune_model_ampere(model, prune_amount)
        else:
            logger.info("Pruning skipped – amount=0")
        tracker.stop_task()

        # 5. Training
        tcfg = cfg.training.versions.default
        train_args = TrainingArguments(output_dir=out_dir,
                                       num_train_epochs=tcfg.num_epochs,
                                       per_device_train_batch_size=tcfg.batch_size,
                                       per_device_eval_batch_size=tcfg.eval_batch_size,
                                       gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
                                       learning_rate=tcfg.learning_rate,
                                       warmup_ratio=tcfg.warmup_ratio,
                                       weight_decay=tcfg.weight_decay,
                                       eval_strategy=tcfg.eval_strategy,
                                       save_strategy=tcfg.save_strategy,
                                       save_total_limit=tcfg.save_total_limit,
                                       logging_steps=tcfg.logging_steps,
                                       fp16=tcfg.fp16,
                                       gradient_checkpointing=tcfg.gradient_checkpointing,
                                       load_best_model_at_end=True,
                                       metric_for_best_model=tcfg.metric_for_best_model,
                                       report_to="none")

        trainer = Trainer(model=model, args=train_args,
                          train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=collator, compute_metrics=compute_metrics)

        tracker.start_task("train_model")
        trainer.train()
        tracker.stop_task()

        # 6. Save
        tracker.start_task("save_model")
        trainer.save_model(out_dir / "model")
        tok.save_pretrained(out_dir / "model")
        tracker.stop_task()

        # 7. Evaluate
        tracker.start_task("evaluate_model")
        test_metrics = trainer.evaluate(test_ds)
        tracker.stop_task()

        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics: {test_metrics}")

    finally:
        tracker.stop()

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default="config.yaml")
    parser.add_argument("--out", type=Path, default=Path("./results"))
    parser.add_argument("--prune-amount", type=float, default=0.5,
                        help="Fraction (≤0.5) of weights to prune in 2:4 structured pattern")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args.cfg, args.out, prune_amount=args.prune_amount)
