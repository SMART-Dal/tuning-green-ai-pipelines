import torch
from transformers.utils import logging as transformers_logging
import time
import wandb
from torch.utils.data import DataLoader

# Set logging verbosity
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

def fine_tune_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, task_type="translation"):
    """Fine-tune a model and return training metrics.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        task_type: Either "translation" or "classification"
        
    Returns:
        dict: Training metrics
    """
    model.train()
    best_val_loss = float('inf')
    training_metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [] if task_type == "classification" else None,
        "val_accuracies": [] if task_type == "classification" else None,
        "best_epoch": 0,
        "learning_rates": []
    }
    
    total_batches = len(train_loader)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
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
                
                # Calculate accuracy for classification task
                if task_type == "classification":
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct_predictions += (predictions == batch["labels"]).sum().item()
                    total_samples += batch["labels"].size(0)
                
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
        
        if task_type == "classification":
            train_accuracy = correct_predictions / total_samples
            training_metrics["train_accuracies"].append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    
                    if task_type == "classification":
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        correct_predictions += (predictions == batch["labels"]).sum().item()
                        total_samples += batch["labels"].size(0)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(f"GPU OOM in validation batch. Skipping batch. Error: {str(e)}")
                        continue
                    raise e
        
        avg_val_loss = val_loss / len(val_loader)
        training_metrics["val_losses"].append(avg_val_loss)
        
        if task_type == "classification":
            val_accuracy = correct_predictions / total_samples
            training_metrics["val_accuracies"].append(val_accuracy)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            training_metrics["best_epoch"] = epoch + 1
            
        # Log epoch metrics to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/epoch_time_minutes": epoch_time/60,
            "train/best_val_loss": best_val_loss
        }
        
        if task_type == "classification":
            log_dict.update({
                "train/accuracy": train_accuracy,
                "val/accuracy": val_accuracy
            })
            
        wandb.log(log_dict)
            
        # Log progress
        log_msg = f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time/60:.1f} min - "
        log_msg += f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        if task_type == "classification":
            log_msg += f" - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}"
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

def evaluate_model(model, test_loader, device, task_type="translation"):
    """Evaluate a model and return inference metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        task_type: Either "translation" or "classification"
        
    Returns:
        dict: Inference metrics
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
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
                
                if task_type == "classification":
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct_predictions += (predictions == batch["labels"]).sum().item()
                
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
    
    if task_type == "classification":
        metrics["accuracy"] = correct_predictions / total_samples
    
    return metrics 