#!/usr/bin/env python3
"""
Fine-tune ModernBERT-base on the BigVul dataset using Hugging Face Optimum’s ORTTrainer
(ONNX Runtime Training) as the optimization engine. All energy accounting tasks are
partitioned exactly as in baseline so task-level emissions sum to the run total.

Usage:
    python train_bigvul_modernbert_optimized.py \
           --cfg config.yaml \
           --out results/
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as hf_logging,
)
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments, ORTSeq2SeqTrainer  # replaces Trainer
from optimum.onnxruntime.modeling_outputs import ORTGenerationOutput
from codecarbon import EmissionsTracker

# --------------------------------------------------------------------------- #
# 1. Logging
hf_logging.set_verbosity_info()
logger = hf_logging.get_logger(__name__)

# --------------------------------------------------------------------------- #
# 2. Dataset & Preprocessing helpers

def load_bigvul():
    """
    Load BigVul dataset from Hugging Face.
    Returns: train_raw, val_raw, test_raw
    """
    ds = load_dataset("bstee615/bigvul")  # HuggingFace dataset repository :contentReference[oaicite:13]{index=13}
    return ds["train"], ds["validation"], ds["test"]

def prep_dataset(dataset, tokenizer, text_col: str, label_col: str, max_len: int):
    """
    Tokenize one split and attach labels as int64 tensors.
    """
    def tok_fn(batch):
        enc = tokenizer(batch[text_col], truncation=True, max_length=max_len)
        enc["labels"] = np.int64(batch[label_col])
        return enc

    keep_cols = [text_col, label_col]
    ds_tok = dataset.map(
        tok_fn,
        remove_columns=[c for c in dataset.column_names if c not in keep_cols],
        batched=False
    )
    ds_tok.set_format("torch")
    return ds_tok

# --------------------------------------------------------------------------- #
# 3. Metrics

def compute_metrics(eval_pred):
    """
    Compute weighted F1 score from logits and labels.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="weighted")}

# --------------------------------------------------------------------------- #
# 4. Main function

def main(cfg_path: Path, out_root: Path):
    # 4.1 Load configuration
    cfg = OmegaConf.load(cfg_path)

    # 4.2 Build run name and output directory
    run_name = f"modernbert_ort_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = out_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4.3 Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="modernbert_ort_train",
        output_dir=str(output_dir),
        log_level="error",
        save_to_file=True,
        measure_power_secs=1.0,
        tracking_mode="process",
        gpu_ids=[0],
    )
    tracker.start()

    try:
        # 4.4 Task: Load dataset
        tracker.start_task("load_dataset")
        train_raw, val_raw, test_raw = load_bigvul()
        if cfg.dummy_mode.enabled:
            n = cfg.dummy_mode.sample_size
            train_raw = train_raw.select(range(n))
            val_raw = val_raw.select(range(n // 2))
            test_raw = test_raw.select(range(n // 2))
        tracker.stop_task()

        # 4.5 Task: Tokenization
        tracker.start_task("tokenize_dataset")
        # We still use AutoTokenizer (the ONNX model will use the same tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=False)
        max_len = cfg.data.versions.default.max_length

        train_ds = prep_dataset(
            train_raw,
            tokenizer,
            cfg.data.versions.default.text_column,
            cfg.data.versions.default.label_column,
            max_len
        )
        val_ds = prep_dataset(
            val_raw,
            tokenizer,
            cfg.data.versions.default.text_column,
            cfg.data.versions.default.label_column,
            max_len
        )
        test_ds = prep_dataset(
            test_raw,
            tokenizer,
            cfg.data.versions.default.text_column,
            cfg.data.versions.default.label_column,
            max_len
        )
        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
        tracker.stop_task()

        # 4.6 Task: Configure & load ONNX-optimized model
        tracker.start_task("load_model")
        # Ensure use_cache=False for export
        config = AutoConfig.from_pretrained(cfg.model.name)
        config.use_cache = False  # required to export the full graph :contentReference[oaicite:14]{index=14}
        config.attn_implementation = "eager"
        config.output_attentions = False
        config.output_hidden_states = False

        # Export to ONNX and then load via ORTModelForSequenceClassification
        from optimum.onnxruntime import ORTModelForSequenceClassification
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            cfg.model.name,
            config=config,
            export=True,               # perform ONNX export + graph optimize :contentReference[oaicite:15]{index=15}
            use_io_binding=False
        )
        # ORTModelForSequenceClassification wraps the ONNX graph inside a PyTorch-like interface
        # so all Trainer code can remain nearly identical
        tracker.stop_task()

        # 4.7 Task: (Optional) Inject LoRA via FastLanguageModel if cfg.lora.enabled
        # If you want to combine ORT with LoRA, you can do it here; otherwise skip.
        if cfg.get("lora", {}).get("enabled", False):
            tracker.start_task("inject_lora")
            from unsloth import FastLanguageModel
            ort_model_wrapped = FastLanguageModel.get_peft_model(
                ort_model,
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                bias="none",
                target_modules="all"
            )
            logger.info(ort_model_wrapped.print_trainable_parameters())
            ort_model = ort_model_wrapped
            tracker.stop_task()

        # 4.8 Task: Training
        tracker.start_task("train_model")
        tcfg = cfg.training.versions.default
        # Use ORTTrainingArguments instead of standard TrainingArguments
        ort_training_args = ORTTrainingArguments(
            output_dir=output_dir,
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
            report_to="none",
        )

        # Instantiate ORTTrainer with the ONNX-optimized model
        ort_trainer = ORTTrainer(
            model=ort_model,
            args=ort_training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        ort_trainer.train()
        tracker.stop_task()

        # 4.9 Task: Save the fine-tuned ONNX model
        tracker.start_task("save_model")
        # .save_model() will save the ORT-wrapped model; to re-export the graph for inference, do:
        ort_trainer.save_model(output_dir / "model_onnx")
        tokenizer.save_pretrained(output_dir / "model_onnx")
        tracker.stop_task()

        # 4.10 Task: Evaluate on test split
        tracker.start_task("evaluate_model")
        test_metrics = ort_trainer.evaluate(test_ds)
        tracker.stop_task()

        # Persist test metrics
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics: {test_metrics}")

    finally:
        tracker.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default="config.yaml")
    parser.add_argument("--out", type=Path, default=Path("./results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args.cfg, args.out)
