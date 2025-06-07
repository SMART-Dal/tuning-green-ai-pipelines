#!/usr/bin/env python3
"""
Fine‑tune ModernBERT‑base on BigVul **with LoRA**, reading all LoRA
hyper‑parameters from `config.yaml` instead of the CLI so that every variant
remains fully reproducible from a single YAML file.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
from peft import LoraConfig, get_peft_model
from codecarbon import EmissionsTracker
import torch

# --------------------------------------------------------------------------- #
# 1. Logging
hf_logging.set_verbosity_info()
logger = hf_logging.get_logger(__name__)

# --------------------------------------------------------------------------- #
# 2. Helpers

def load_bigvul():
    ds = load_dataset("bstee615/bigvul")
    return ds["train"], ds["validation"], ds["test"]


def prep_dataset(ds, tok, text_col, label_col, max_len):
    def tok_fn(batch):
        enc = tok(batch[text_col], truncation=True, max_length=max_len)
        enc["labels"] = np.int64(batch[label_col])
        return enc

    keep = [text_col, label_col]
    ds_tok = ds.map(tok_fn, remove_columns=[c for c in ds.column_names if c not in keep], batched=False)
    ds_tok.set_format("torch")
    return ds_tok


def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="weighted")}

# --------------------------------------------------------------------------- #
# 3. Main

def main(cfg_path: Path, out_root: Path):
    cfg = OmegaConf.load(cfg_path)

    # ------------------------------------------------- LoRA hyper‑params ---- #
    lora_cfg = cfg.lora
    R           = lora_cfg.r
    ALPHA       = lora_cfg.alpha
    DROPOUT     = lora_cfg.dropout
    LOAD_4BIT   = lora_cfg.load_in_4bit

    run_name = f"lora_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(project_name="modernbert_lora_train",
                               output_dir=str(out_dir), log_level="error",
                               save_to_file=True, measure_power_secs=1.0,
                               tracking_mode="process", gpu_ids=[0])
    tracker.start()

    try:
        # ------------------------------ 1. Load dataset
        tracker.start_task("load_dataset")
        train_raw, val_raw, test_raw = load_bigvul()
        if cfg.dummy_mode.enabled:
            n = cfg.dummy_mode.sample_size
            train_raw, val_raw, test_raw = (
                train_raw.select(range(n)),
                val_raw.select(range(n // 2)),
                test_raw.select(range(n // 2)),
            )
        tracker.stop_task()

        # ------------------------------ 2. Tokenisation & model load
        tracker.start_task("tokenize_dataset")
        model_name   = cfg.model.name
        NUM_LABELS   = cfg.model.num_labels
        max_len      = cfg.data.versions.default.max_length

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(model_name)

        tracker.stop_task()

        tracker.start_task("load_model")
        
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        train_ds = prep_dataset(train_raw, tok, cfg.data.versions.default.text_column,
                                cfg.data.versions.default.label_column, max_len)
        val_ds   = prep_dataset(val_raw,  tok, cfg.data.versions.default.text_column,
                                cfg.data.versions.default.label_column, max_len)
        test_ds  = prep_dataset(test_raw, tok, cfg.data.versions.default.text_column,
                                cfg.data.versions.default.label_column, max_len)
        collator = DataCollatorWithPadding(tok, return_tensors="pt")

        # ------------------------------ 3. Inject LoRA adapters
       
        lora_config = LoraConfig(
            r=R,
            lora_alpha=ALPHA,
            lora_dropout=DROPOUT,
            bias="none",
            target_modules=["query", "key", "value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        logger.info(model.print_trainable_parameters())
        tracker.stop_task()

        # ------------------------------ 4. Training
        tracker.start_task("train_model")
        tcfg = cfg.training.versions.default
        train_args = TrainingArguments(
            output_dir=out_dir,
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
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=tcfg.gradient_checkpointing,
            load_best_model_at_end=True,
            metric_for_best_model=tcfg.metric_for_best_model,
            report_to="none",
        )

        trainer = Trainer(model=model, tokenizer=tok,
                          args=train_args, train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=collator, compute_metrics=compute_metrics)
        trainer.train()
        tracker.stop_task()

        # ------------------------------ 5. Save
        tracker.start_task("save_model")
        trainer.save_model(out_dir / "model")
        tok.save_pretrained(out_dir / "model")
        tracker.stop_task()

        # ------------------------------ 6. Evaluate
        tracker.start_task("evaluate_model")
        test_metrics = trainer.evaluate(test_ds)
        tracker.stop_task()

        emissions = tracker.stop()

        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics: {test_metrics}")

         with open(output_dir / "energy_stats_train.json", "w") as f:
            json.dump(json.loads(tracker.final_emissions_data.toJSON()), f, indent=2)

        
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("./results"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args.cfg, args.out)
