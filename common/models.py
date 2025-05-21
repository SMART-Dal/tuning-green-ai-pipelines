import os
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM
)
from transformers.utils import logging as transformers_logging

# Set logging verbosity
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

def get_cache_dir():
    """Get the path to the cache directory."""
    cache_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_model(task_type="translation", device="cuda"):
    """Get model and tokenizer for the specified task.
    
    Args:
        task_type: Either "translation" or "classification"
        device: Device to load the model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    if task_type == "translation":
        logger.info("Loading Qwen model for code translation")
        model_name = "Qwen/Qwen2.5-Coder-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:  # classification
        logger.info("Loading ModernBERT model for classification")
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Move model to specified device
    if device != "auto":
        model = model.to(device)
    
    return model, tokenizer

def get_optimizer(model, optimizer_name="adamw", learning_rate=5e-5):
    """Get optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of the optimizer to use
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Optimizer: PyTorch optimizer
    """
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name="linear", num_training_steps=1000):
    """Get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of the scheduler to use
        num_training_steps: Total number of training steps
        
    Returns:
        Scheduler: PyTorch learning rate scheduler
    """
    if scheduler_name.lower() == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def save_checkpoint(model, optimizer, epoch, path):
    """Save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        path (str): Path to save the checkpoint
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(model, optimizer, path):
    """Load a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path (str): Path to the checkpoint
        
    Returns:
        int: The epoch number from the checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    logger.info(f"Loaded checkpoint from {path}")
    return epoch
