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
import sys
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

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from peft import LoraConfig, get_peft_model
from codecarbon import EmissionsTracker
import torch
from vllm import LLM, SamplingParams
from common.layer_drop import layer_drop



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

    variant = "dummy" if cfg.dummy_mode.enabled else "default"

    # Get variant name from the directory structure
    variant_name = Path(__file__).parent.parent.name  

    run_name = f"{variant}_{datetime.now():%Y%m%d_%H%M%S}" 
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
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        tracker.stop_task()

        tracker.start_task("load_model")
        
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto",
            attn_implementation="sdpa"
        )

        vcfg = cfg.data.versions[variant]

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

        if cfg.layer_pruning.enabled:
            layer_drop(
                model.model.layers, 
                N=cfg.layer_pruning.num_layers,
                position=cfg.layer_pruning.position
        )
        model = get_peft_model(model, lora_config)
        logger.info(model.print_trainable_parameters())
        
        
        tracker.stop_task()

        # ------------------------------ 4. Training
        tracker.start_task("train_model")
        tcfg = cfg.training.versions.default
        train_args = TrainingArguments(
            output_dir          = out_dir,
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
            fp16                        = not torch.cuda.is_bf16_supported(),  # Use fp16 only if bf16 is not supported
            bf16                        = torch.cuda.is_bf16_supported(),      # Use bf16 if supported
            gradient_checkpointing      = tcfg.gradient_checkpointing,
            load_best_model_at_end      = True,
            metric_for_best_model       = tcfg.metric_for_best_model,
            optim                       = tcfg.optimizer,
            report_to                   = "none",
            dataloader_pin_memory       = True,
            dataloader_num_workers      = 4,
        )

        trainer = Trainer(model=model, tokenizer=tok,
                          args=train_args, train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=collator, compute_metrics=compute_metrics)
        trainer.train()
        tracker.stop_task()

        # ------------------------------ 5. Save
        tracker.start_task("save_model")
        logger.info("Merging LoRA adapters into base weights …")
        model = model.merge_and_unload()
        model_save_path = out_dir / "model"
        model.save_pretrained(model_save_path)
        tok.save_pretrained(model_save_path)
        # Initialize vLLM with the trained model
        vllm_model = LLM(
            model=str(model_save_path),  # Use the locally saved model
            tensor_parallel_size=1,  # Adjust based on available GPUs
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            task="classify"
        )

        # Prepare test data for vLLM
        test_texts = [tok.decode(tok.encode(text, truncation=True, max_length=vcfg.max_length)) 
                     for text in test_raw[vcfg.text_column]]
        test_labels = test_raw[vcfg.label_column]

        tracker.stop_task()

        # ---- Test set evaluation using vLLM
        tracker.start_task("evaluate_model")
        # Run inference with vLLM
        outputs = vllm_model.classify(test_texts)
        print(outputs)
        predictions = [int(np.argmax(out.outputs.probs)) for out in outputs]
        tracker.stop_task()


        # Calculate metrics
        test_metrics = {
            "eval_f1": f1_score(test_labels, predictions, average="weighted")
        }
        
        # Get emissions data
        emissions = tracker.stop()
        
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics: {test_metrics}")
            
        with open(out_dir / "energy_stats_train.json", "w") as f:
            json.dump(json.loads(tracker.final_emissions_data.toJSON()), f, indent=2)

    except Exception as e:
        # Ensure we stop tracking even if there's an error
        if tracker:
            tracker.stop()
        logger.error(f"Pipeline failed: {str(e)}")
        raise

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("./results"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args.cfg, args.out)
