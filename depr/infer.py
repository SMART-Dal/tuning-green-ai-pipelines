#!/usr/bin/env python3
"""
Run inference using a fine-tuned ModernBERT-base model on the BigVul dataset.

Usage:
    python infer.py [--cfg config/config.yaml] [--out results/]
"""
import os, sys
from pathlib import Path
from datetime import datetime
import json
import glob

import torch, wandb, numpy as np
from omegaconf import OmegaConf
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, DataCollatorWithPadding,
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
    ds = load_dataset("bstee615/bigvul")
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
# 4.  Helper functions
def find_latest_model(results_dir: Path) -> Path:
    """Find the latest model directory in the results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Path to the latest model directory
        
    Raises:
        FileNotFoundError: If no model directories are found
    """
    # Find all directories that match the variant_timestamp pattern
    timestamp_dirs = []
    for d in results_dir.iterdir():
        if d.is_dir() and "_" in d.name:
            parts = d.name.split("_")
            if len(parts) >= 3:  # variant_YYYYMMDD_HHMMSS
                try:
                    # Try to parse the timestamp part
                    datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
                    timestamp_dirs.append(d)
                except ValueError:
                    continue
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No variant_timestamp directories found in {results_dir}")
    
    # Sort by modification time (newest first)
    latest_dir = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / "model"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {latest_dir}")
    
    logger.info(f"Using latest model from: {model_path}")
    return model_path

# --------------------------------------------------------------------------- #
# 5.  Main
def main(cfg_path: Path, output_root: Path):
    cfg = OmegaConf.load(cfg_path)
    variant = "dummy" if cfg.dummy_mode.enabled else "default"

    # Find latest model
    try:
        model_path = find_latest_model(output_root)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # ---- WandB offline initialisation
    os.environ["WANDB_MODE"] = "offline"            
    run_name = f"inference_{variant}_{datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project="bigvul-modernbert",
               name=run_name,
               dir=str(output_root / run_name / "wandb"),
               config=OmegaConf.to_container(cfg),
               tags=["modernbert", "bigvul", "inference"])

    # Get variant name from the directory structure
    variant_name = Path(__file__).parent.parent.name  

    run_name = f"{variant}_{datetime.now():%Y%m%d_%H%M%S}" # Gets the variant folder name (e.g., V0_baseline)
    # Create output directory
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"{variant_name}_inference",
        output_dir=str(output_root / run_name),
        log_level="error",
        save_to_file=True,
        measure_power_secs=1.0,
        tracking_mode="process",
        gpu_ids=[0]
    )
    tracker.start()

    try:
        # ---- Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # ---- Data
        train_raw, val_raw, test_raw = load_bigvul()
        if cfg.dummy_mode.enabled:
            n = cfg.dummy_mode.sample_size
            test_raw = test_raw.select(range(n//2))
        
        vcfg = cfg.data.versions[variant]
        test_ds = prep_dataset(test_raw, tokenizer, vcfg.text_column, vcfg.label_column, vcfg.max_length)
        collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

        # ---- Trainer for evaluation
        trainer = Trainer(
            model=model,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        # ---- Run inference
        logger.info("Running inference on test set...")
        test_metrics = trainer.evaluate(test_ds)
        
        # Get emissions data
        emissions = tracker.stop()
        
        
        # Log to wandb
        wandb.log({
            "test_metrics": test_metrics,
            "energy_consumption": emissions,
            "carbon_emissions": emissions,
            "duration_seconds": tracker.duration,
            "cpu_power": tracker.cpu_power,
            "gpu_power": tracker.gpu_power,
            "ram_power": tracker.ram_power
        })
        
        # Save energy stats to file
        with open(output_root / run_name / "energy_stats_inference.json", "w") as f:
            json.dump(json.loads(tracker.final_emissions_data.toJSON()), f, indent=2)
            
        wandb.finish()
        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Energy stats: {emissions}")

    except Exception as e:
        # Ensure we stop tracking even if there's an error
        if tracker:
            tracker.stop()
        logger.error(f"Inference failed: {str(e)}")
        raise

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default="config.yaml")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args.cfg, args.out)
