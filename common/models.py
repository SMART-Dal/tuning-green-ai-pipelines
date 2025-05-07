import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import logging

logger = logging.getLogger(__name__)

def get_model(model_name, task_type="translation", device=None):
    """Get a model and its tokenizer for the specified task.
    
    Args:
        model_name (str): Name or path of the model
        task_type (str): Either "translation" or "classification"
        device (str, optional): Device to load the model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading {model_name} for {task_type} task on {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model based on task type
    if task_type == "translation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:  # classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification for vulnerability detection
        )
    
    # Move model to specified device
    model = model.to(device)
    
    return model, tokenizer

def get_optimizer(model, optimizer_name="adamw", learning_rate=5e-5):
    """Get an optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_name (str): Name of the optimizer
        learning_rate (float): Learning rate
        
    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name="linear", num_training_steps=None):
    """Get a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name (str): Name of the scheduler
        num_training_steps (int, optional): Number of training steps
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Scheduler instance
    """
    if scheduler_name.lower() == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer)
    elif scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

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
