import os
import json
from pathlib import Path
from datetime import datetime
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

# Set logging verbosity
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

from .data import load_codexglue_dataset, load_bigvul_dataset, prepare_dataset_for_model
from .energy import EnergyMonitor
from .models import get_model, get_optimizer, get_scheduler
from .training import fine_tune_model, evaluate_model
from .utils import get_system_info

def run_pipeline(output_dir: Path, cfg: DictConfig, task_type: str) -> None:
    """Run complete pipeline for a specific task.
    
    Args:
        output_dir: Directory to save results
        cfg: Configuration containing parameters for all stages
        task_type: Either "translation" or "classification"
    """
    logger.info(f"Starting pipeline for {task_type} task")
    
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
        tags=[task_type, "baseline"],
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
        logger.info(f"Loading dataset for {task_type}...")
        data_cfg = cfg.data.versions[config_version]
        
        # Load appropriate dataset
        if task_type == "translation":
            train_dataset, val_dataset, test_dataset = load_codexglue_dataset()
        else:  # classification
            train_dataset, val_dataset, test_dataset = load_bigvul_dataset()
        
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
        train_dataset = prepare_dataset_for_model(train_dataset, tokenizer, max_length=data_cfg.max_length, task_type=task_type)
        val_dataset = prepare_dataset_for_model(val_dataset, tokenizer, max_length=data_cfg.max_length, task_type=task_type)
        test_dataset = prepare_dataset_for_model(test_dataset, tokenizer, max_length=data_cfg.max_length, task_type=task_type)
        
        # Save dataset statistics
        with open(stage_dir / "dataset_stats.json", "w") as f:
            json.dump(dataset_stats, f, indent=2)
        
        # 2. Architecture Stage
        logger.info("Loading and preparing model for fine-tuning...")
        model, tokenizer = get_model(task_type=task_type, device=device)
        
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
        train_loader = DataLoader(train_dataset, batch_size=training_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_cfg.eval_batch_size)
        
        optimizer = get_optimizer(model, optimizer_name=training_cfg.optimizer, learning_rate=training_cfg.learning_rate)
        num_training_steps = len(train_loader) * training_cfg.num_epochs
        scheduler = get_scheduler(optimizer, scheduler_name=training_cfg.scheduler, num_training_steps=num_training_steps)
        
        # Watch model gradients
        wandb.watch(model, log="all")
        
        training_metrics = fine_tune_model(
            model, train_loader, val_loader, optimizer, scheduler,
            training_cfg.num_epochs, device, task_type=task_type
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
        inference_metrics = evaluate_model(model, test_loader, device, task_type=task_type)
        
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
        energy_stats = energy_monitor.stop()
        
        # Log energy stats to wandb
        wandb.log({
            "energy_consumption": energy_stats,
            "carbon_emissions": energy_stats.get("emissions", {}),
            "duration_seconds": energy_stats.get("duration_seconds", 0),
            "cpu_usage": energy_stats.get("cpu_percent", 0),
            "memory_usage_gb": energy_stats.get("memory_used_gb", 0),
            "gpu_stats": energy_stats.get("gpu_stats", [])
        })
        
        with open(stage_dir / "energy_stats.json", "w") as f:
            json.dump(energy_stats, f, indent=2)
        
        logger.info(f"Pipeline completed successfully. Results saved to {stage_dir}")
        
    except Exception as e:
        energy_stats = energy_monitor.stop()
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        wandb.finish() 