import os
from pathlib import Path
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging
import torch

# Set logging verbosity
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

def get_dataset_path():
    """Get the path to the datasets directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "datasets"

def load_codexglue_dataset(split_ratio=(0.8, 0.1, 0.1)):
    """Load and preprocess the CodexGLUE dataset for code-to-code translation.
    
    Args:
        split_ratio (tuple): Train/valid/test split ratios
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
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

def load_bigvul_dataset(split_ratio=(0.8, 0.1, 0.1)):
    """Load and preprocess the BigVul dataset for vulnerability classification.
    
    Args:
        split_ratio (tuple): Train/valid/test split ratios
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Loading BigVul dataset...")
    
    # Load from HuggingFace
    dataset = load_dataset("bstee615/bigvul")
    
    # Remove duplicates before splitting
    dataset = dataset.filter(lambda x: x["vul"] is not None)
    dataset = dataset.unique("func")
    
    # Split the dataset
    train_ratio, val_ratio, test_ratio = split_ratio
    total_size = len(dataset)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(val_dataset)} validation examples")
    logger.info(f"Loaded {len(test_dataset)} test examples")
    
    return train_dataset, val_dataset, test_dataset

def prepare_dataset_for_model(dataset, tokenizer, max_length=512, task_type="translation"):
    """Prepare dataset for model input.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        task_type (str): Either "translation" or "classification"
        
    Returns:
        Dataset: Tokenized and prepared dataset
    """
    def tokenize_function(examples):
        if task_type == "translation":
            # For code-to-code translation
            # The CodeXGLUE dataset uses 'java' and 'cs' as column names
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
        else:
            # For vulnerability classification
            inputs = tokenizer(
                examples["func"],
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": examples["vul"]
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
