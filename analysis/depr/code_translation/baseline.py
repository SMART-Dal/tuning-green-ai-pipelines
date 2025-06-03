#!/usr/bin/env python3
import os
import sys
import json
import time
import torch
import wandb
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as transformers_logging
from datasets import load_dataset

# Set logging verbosity
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

from common.energy import EnergyMonitor
from common.utils import get_system_info

def load_codexglue_dataset():
    """Load and preprocess the CodexGLUE dataset for code-to-code translation."""
    logger.info("Loading CodexGLUE dataset...")
    
    # Load from HuggingFace
    dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")
    
    # Get the splits
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(val_dataset)} validation examples")
    logger.info(f"Loaded {len(test_dataset)} test examples")
    
    return train_dataset, val_dataset, test_dataset

def prepare_dataset_for_model(dataset, tokenizer, max_length=512):
    """Prepare dataset for model input."""
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["java"],  # Java source code
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        targets = tokenizer(
            examples["cs"],  # C# target code
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Convert to PyTorch tensors
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset

def fine_tune_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    """Fine-tune a model for translation and return training metrics."""
    model.train()
    best_val_loss = float('inf')
    training_metrics = {
        "train_losses": [],
        "val_losses": [],
        "best_epoch": 0,
        "learning_rates": []
    }
    
    total_batches = len(train_loader)
    print(f"Total batches: {total_batches}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Calculate progress and estimated time
                progress = (epoch * total_batches + batch_idx) / (num_epochs * total_batches) * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (progress / 100) if progress > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                
                # Log progress every 10% or at the start/end of each epoch
                if batch_idx == 0 or batch_idx == total_batches - 1 or batch_idx % (total_batches // 10) == 0:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx + 1}/{total_batches} "
                              f"({progress:.1f}%) - Est. remaining: {remaining_time/60:.1f} min")
                
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Log batch metrics to wandb
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/progress": progress,
                    "train/estimated_remaining_minutes": remaining_time/60
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning(f"GPU OOM in batch. Skipping batch. Error: {str(e)}")
                    continue
                raise e
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / len(train_loader)
        training_metrics["train_losses"].append(avg_train_loss)
        training_metrics["learning_rates"].append(scheduler.get_last_lr()[0])
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(f"GPU OOM in validation batch. Skipping batch. Error: {str(e)}")
                        continue
                    raise e
        
        avg_val_loss = val_loss / len(val_loader)
        training_metrics["val_losses"].append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            training_metrics["best_epoch"] = epoch + 1
            
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/epoch_time_minutes": epoch_time/60,
            "train/best_val_loss": best_val_loss
        })
            
        # Log progress
        log_msg = f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time/60:.1f} min - "
        log_msg += f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        log_msg += f" - LR: {scheduler.get_last_lr()[0]:.2e}"
        logger.info(log_msg)
    
    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time/60:.1f} minutes")
    
    # Log final training metrics
    wandb.log({
        "train/total_time_minutes": total_training_time/60,
        "train/best_epoch": training_metrics["best_epoch"],
        "train/final_train_loss": training_metrics["train_losses"][-1],
        "train/final_val_loss": training_metrics["val_losses"][-1]
    })
    
    return training_metrics

def evaluate_model(model, test_loader, device):
    """Evaluate a model for translation and return inference metrics."""
    model.eval()
    total_loss = 0
    total_samples = 0
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Measure inference time
                if torch.cuda.is_available():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                
                outputs = model(**batch)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_time.elapsed_time(end_time) / 1000)  # Convert to seconds
                
                loss = outputs.loss
                total_loss += loss.item()
                total_samples += batch["input_ids"].size(0)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning(f"GPU OOM in inference batch. Skipping batch. Error: {str(e)}")
                    continue
                raise e
    
    metrics = {
        "avg_loss": total_loss / len(test_loader),
        "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else None,
        "total_samples": total_samples
    }
    
    return metrics

def run_pipeline(output_dir: Path, cfg: DictConfig) -> None:
    """Run complete pipeline for code translation task."""
    logger.info("Starting pipeline for translation task")
    
    # Create output directory with dummy indicator if in dummy mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "dummy" if cfg.dummy_mode.enabled else "baseline"
    stage_dir = output_dir / f"{prefix}_{timestamp}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create wandb directory
    wandb_dir = stage_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb in offline mode
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(wandb_dir)
    
    wandb.init(
        project="greenai-pipeline",
        name=f"{prefix}_{timestamp}",
        config=OmegaConf.to_container(cfg),
        tags=["translation", "baseline"],
        dir=str(wandb_dir)
    )
    
    # Save configuration
    with open(stage_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=2)
    
    # Initialize energy monitor
    energy_monitor = EnergyMonitor(timestamp=timestamp)
    energy_monitor.start()
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Log system info to wandb
        system_info = get_system_info()
        wandb.log({"system_info": system_info})
        
        # Select configuration version based on dummy mode
        config_version = "dummy" if cfg.dummy_mode.enabled else "default"
        logger.info(f"Using configuration version: {config_version}")
        
        # 1. Data Stage
        logger.info("Loading dataset...")
        data_cfg = cfg.data.versions[config_version]
        
        # Load dataset
        train_dataset, val_dataset, test_dataset = load_codexglue_dataset()
        
        # If in dummy mode, take only a small sample
        if cfg.dummy_mode.enabled:
            train_dataset = train_dataset.select(range(cfg.dummy_mode.sample_size))
            val_dataset = val_dataset.select(range(cfg.dummy_mode.sample_size // 2))
            test_dataset = test_dataset.select(range(cfg.dummy_mode.sample_size // 2))
            logger.info(f"Using dummy sample sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Log dataset stats to wandb
        dataset_stats = {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset)
        }
        wandb.log({"dataset_stats": dataset_stats})
        
        # Prepare datasets for model
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            trust_remote_code=True  # Required for Qwen models
        )
        train_dataset = prepare_dataset_for_model(train_dataset, tokenizer, max_length=data_cfg.max_length)
        val_dataset = prepare_dataset_for_model(val_dataset, tokenizer, max_length=data_cfg.max_length)
        test_dataset = prepare_dataset_for_model(test_dataset, tokenizer, max_length=data_cfg.max_length)
        
        # Save dataset statistics
        with open(stage_dir / "dataset_stats.json", "w") as f:
            json.dump(dataset_stats, f, indent=2)
        
        # 2. Architecture Stage
        logger.info("Loading and preparing model for fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
        model = model.to(device)
        
        # Log model stats to wandb
        model_stats = {
            "name": cfg.model.name,
            "type": cfg.model.type,
            "parameters": sum(p.numel() for p in model.parameters()),
            "precision": "fp32",
            "device": device
        }
        wandb.log({"model_stats": model_stats})
        
        # Save model statistics
        with open(stage_dir / "model_stats.json", "w") as f:
            json.dump(model_stats, f, indent=2)
        
        # 3. Training Stage
        logger.info("Starting fine-tuning...")
        training_cfg = cfg.training.versions.default
        
        # Reduce batch size if CUDA is available to prevent OOM
        if torch.cuda.is_available():
            training_cfg.batch_size = min(training_cfg.batch_size, 8)  # Reduce batch size
            training_cfg.eval_batch_size = min(training_cfg.eval_batch_size, 8)
            logger.info(f"Reduced batch sizes to prevent OOM - Train: {training_cfg.batch_size}, Eval: {training_cfg.eval_batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=training_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_cfg.eval_batch_size)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.learning_rate)
        num_training_steps = len(train_loader) * training_cfg.num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps
        )
        
        # Watch model gradients
        wandb.watch(model, log="all")
        
        training_metrics = fine_tune_model(
            model, train_loader, val_loader, optimizer, scheduler,
            training_cfg.num_epochs, device
        )
        
        # Log training metrics to wandb
        wandb.log({"training_metrics": training_metrics})
        
        # Save model and training metrics
        model_path = stage_dir / "model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        with open(stage_dir / "training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        # 4. Inference Stage
        logger.info("Running inference...")
        inference_cfg = cfg.inference.versions.default
        test_loader = DataLoader(test_dataset, batch_size=inference_cfg.batch_size)
        inference_metrics = evaluate_model(model, test_loader, device)
        
        # Log inference metrics to wandb
        wandb.log({"inference_metrics": inference_metrics})
        
        with open(stage_dir / "inference_metrics.json", "w") as f:
            json.dump(inference_metrics, f, indent=2)
        
        # 5. System Stage
        logger.info("Collecting system information...")
        system_info = get_system_info()
        with open(stage_dir / "system_info.json", "w") as f:
            json.dump(system_info, f, indent=2)
        
        # Get and save energy consumption
        emissions = energy_monitor.stop()
        
        # Log energy stats to wandb
        wandb.log({
            "energy_consumption": emissions,
            "carbon_emissions": emissions.get("emissions", {}),
            "duration_seconds": emissions.get("duration_seconds", 0),
            "cpu_usage": emissions.get("cpu_percent", 0),
            "memory_usage_gb": emissions.get("memory_used_gb", 0),
            "gpu_stats": emissions.get("gpu_stats", [])
        })
        
        with open(stage_dir / "emissions.json", "w") as f:
            json.dump(json.loads(tracker.final_emissions_data.toJSON()), f, indent=2)
        
        logger.info(f"Pipeline completed successfully. Results saved to {stage_dir}")
        
    except Exception as e:
        emissions = energy_monitor.stop()
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    # Set environment variable for project root
    os.environ["PROJECT_ROOT"] = project_root
    
    # Create output directory
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    cfg = OmegaConf.load(config_path)
    
    # Run pipeline
    run_pipeline(output_dir, cfg) 