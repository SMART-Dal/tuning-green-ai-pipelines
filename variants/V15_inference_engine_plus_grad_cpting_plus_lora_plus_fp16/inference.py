#!/usr/bin/env python3
"""
Run inference with ModernBERT-base on the BigVul dataset.

Usage:
    python inference_bigvul_modernbert.py --cfg config/config.yaml --out results/
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
def main(cfg_path: Path, model_path: Path, output_root: Path):
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
        project_name=f"{variant_name}_inference",
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
        train_raw, val_raw, test_raw = load_bigvul()
        if cfg.dummy_mode.enabled:
            n = cfg.dummy_mode.sample_size
            train_raw, val_raw, test_raw = (train_raw.select(range(n)),
                                            val_raw.select(range(n//2)),
                                            test_raw.select(range(n//2)))
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        vcfg = cfg.data.versions[variant]
        test_ds  = prep_dataset(test_raw,  tok, vcfg.text_column, vcfg.label_column, vcfg.max_length)

        collator = DataCollatorWithPadding(tok, return_tensors="pt")  # dynamic padding

        # ---- Model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # ---- TrainingArguments
        tcfg = cfg.training.versions[variant]
        training_args = TrainingArguments(
            output_dir          = output_dir,
            per_device_eval_batch_size  = tcfg.eval_batch_size,
            fp16                        = tcfg.fp16,
            gradient_checkpointing      = tcfg.gradient_checkpointing,
            report_to                   = "none",
        )

        # ---- Trainer
        trainer = Trainer(
            model               = model,
            args                = training_args,
            data_collator       = collator,
            compute_metrics     = compute_metrics,
        )

        # ---- Test set evaluation
        test_metrics = trainer.evaluate(test_ds)
        
        # Get emissions data
        emissions = tracker.stop()
        
        
        # Save test metrics to file
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
            
        # Save energy stats to file
        with open(output_dir / "energy_stats_inference.json", "w") as f:
            json.dump(json.loads(tracker.final_emissions_data.toJSON()), f, indent=2)
            
        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Energy stats: {emissions}")

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
    main(args.cfg, args.model, args.out) 