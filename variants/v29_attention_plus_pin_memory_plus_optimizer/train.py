#!/usr/bin/env python3
"""
Fine-tune ModernBERT-base on the BigVul dataset (binary classification).

Usage:
    python train_bigvul_modernbert.py --cfg config/config.yaml --out results/
"""
import os, sys, time
from pathlib import Path
from datetime import datetime
import json

import torch, numpy as np
from omegaconf import OmegaConf
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    logging as hf_logging,
)
from codecarbon import EmissionsTracker

# --------------------------------------------------------------------------- #
# 1.  Logging
hf_logging.set_verbosity_info()
logger = hf_logging.get_logger(__name__)

# --------------------------------------------------------------------------- #
# 2.  Dataset helpers
def load_bigvul():
    """Load BigVul splits exactly as hosted on Hugging Face."""
    ds = load_dataset("bstee615/bigvul")          # train / validation / test :contentReference[oaicite:6]{index=6}
    return ds["train"], ds["validation"], ds["test"]

def prep_dataset(dataset, tok, text_col, label_col, max_len):
    """Tokenise one split and attach integer labels."""
    def tok_fn(batch):
        enc = tok(batch[text_col],
                  truncation=True,
                  max_length=max_len)
        enc["labels"] = np.int64(batch[label_col])   # ensure int64 for PyTorch
        return enc

    keep_cols = [text_col, label_col]
    ds_tok = dataset.map(tok_fn,
                         remove_columns=[c for c in dataset.column_names if c not in keep_cols],
                         batched=False)
    ds_tok.set_format("torch")
    return ds_tok

# --------------------------------------------------------------------------- #
# 3.  Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="weighted")}   
# --------------------------------------------------------------------------- #
# 4.  Main
def main(cfg_path: Path, output_root: Path):
    cfg = OmegaConf.load(cfg_path)
    variant = "dummy" if cfg.dummy_mode.enabled else "default"

    # Get variant name from the directory structure
    variant_name = Path(__file__).parent.parent.name  

    run_name = f"{variant}_{datetime.now():%Y%m%d_%H%M%S}" 

    # Create output directory
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"{variant_name}_train",
        output_dir=str(output_dir),
        log_level="error",
        save_to_file=True,
        measure_power_secs=1.0,
        tracking_mode="process",
        gpu_ids=[0]
    )
    tracker.start()

    try:
        # ---- Data
        tracker.start_task("load_dataset")
        train_raw, val_raw, test_raw = load_bigvul()
        if cfg.dummy_mode.enabled:
            n = cfg.dummy_mode.sample_size
            train_raw, val_raw, test_raw = (train_raw.select(range(n)),
                                            val_raw.select(range(n//2)),
                                            test_raw.select(range(n//2)))
        tracker.stop_task()

        tracker.start_task("tokenize_dataset")
        tok = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=False)
        vcfg = cfg.data.versions[variant]
        train_ds = prep_dataset(train_raw, tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        val_ds   = prep_dataset(val_raw,   tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        test_ds  = prep_dataset(test_raw,  tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        tracker.stop_task()

        collator = DataCollatorWithPadding(tok, return_tensors="pt")  # dynamic padding

        # ---- Model
        tracker.start_task("load_model")
        model = AutoModelForSequenceClassification.from_pretrained(
                    cfg.model.name, num_labels=cfg.model.num_labels, attn_implementation="sdpa")
        tracker.stop_task()

        # ---- TrainingArguments
        tcfg = cfg.training.versions[variant]
        training_args = TrainingArguments(
            output_dir          = output_dir,
            num_train_epochs    = tcfg.num_epochs,
            per_device_train_batch_size = tcfg.batch_size,
            per_device_eval_batch_size  = tcfg.eval_batch_size,
            gradient_accumulation_steps = tcfg.gradient_accumulation_steps,
            learning_rate               = tcfg.learning_rate,
            warmup_ratio                = tcfg.warmup_ratio,
            weight_decay                = tcfg.weight_decay,
            eval_strategy               = tcfg.eval_strategy,
            save_strategy               = tcfg.save_strategy,
            save_total_limit            = tcfg.save_total_limit,
            logging_steps               = tcfg.logging_steps,
            fp16                        = tcfg.fp16,
            gradient_checkpointing      = tcfg.gradient_checkpointing,
            load_best_model_at_end      = True,
            metric_for_best_model       = tcfg.metric_for_best_model,
            report_to                   = "none",
            optim                       = tcfg.optimizer,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
        )

        # ---- Trainer
        trainer = Trainer(
            model               = model,
            args                = training_args,
            train_dataset       = train_ds,
            eval_dataset        = val_ds,
            data_collator       = collator,
            compute_metrics     = compute_metrics,
        )

        tracker.start_task("train_model")
        trainer.train()
        tracker.stop_task()

        tracker.start_task("save_model")
        trainer.save_model(output_dir / "model")
        tok.save_pretrained(output_dir / "model")
        tracker.stop_task()

        # ---- Test set evaluation
        tracker.start_task("evaluate_model")
        test_metrics = trainer.evaluate(test_ds)
        tracker.stop_task()
        
        # Get emissions data
        emissions = tracker.stop()
        
        # Save test metrics to file
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
            
        # Save energy stats to file
        with open(output_dir / "energy_stats_train.json", "w") as f:
            json.dump(json.loads(tracker.final_emissions_data.toJSON()), f, indent=2)
            
        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Energy stats: {emissions}")
        for task_name, task in tracker._tasks.items():
            logger.info(f"Emissions for {task_name}: {1000 * task.emissions_data.emissions} g CO₂")

    except Exception as e:
        # Ensure we stop tracking even if there's an error
        if tracker:
            tracker.stop()
        logger.error(f"Pipeline failed: {str(e)}")
        raise

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default="config.yaml")
    parser.add_argument("--out", type=Path, default=Path("./results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args.cfg, args.out)
