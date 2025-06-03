#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers.utils import logging as transformers_logging

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from common.cache import setup_cache_env

# Set logging verbosity
transformers_logging.set_verbosity_info()
logger = transformers_logging.get_logger(__name__)

def download_model(model_name, task_type="translation", cache_dir=None):
    """Download and cache a model and its tokenizer.
    
    Args:
        model_name: Name of the model to download
        task_type: Either "translation" or "classification"
        cache_dir: Path to cache directory
    """
    logger.info(f"Downloading model {model_name} for {task_type} task...")
    
    # Download tokenizer
    logger.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir / "hub") if cache_dir else None,
        trust_remote_code=True  # Required for Qwen models
    )
    
    # Download model
    logger.info("Downloading model...")
    if model_name.startswith("Qwen"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir / "hub") if cache_dir else None,
            trust_remote_code=True,  # Required for Qwen models
            device_map="auto"  # Automatically handle device placement
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification for vulnerability detection
            cache_dir=str(cache_dir / "hub") if cache_dir else None
        )
    
    logger.info(f"Successfully cached model {model_name}")

def download_dataset(dataset_name, cache_dir=None):
    """Download and cache a dataset.
    
    Args:
        dataset_name: Name of the dataset to download
        cache_dir: Path to cache directory
    """
    logger.info(f"Downloading dataset {dataset_name}...")
    
    if dataset_name == "codexglue":
        dataset = load_dataset(
            "google/code_x_glue_cc_code_to_code_trans",
            cache_dir=str(cache_dir / "datasets") if cache_dir else None
        )
        logger.info("Successfully cached CodexGLUE dataset")
    elif dataset_name == "bigvul":
        dataset = load_dataset(
            "bstee615/bigvul",
            cache_dir=str(cache_dir / "datasets") if cache_dir else None
        )
        logger.info("Successfully cached BigVul dataset")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Download and cache models and datasets locally")
    parser.add_argument("--models", nargs="+", 
                      default=["Qwen/Qwen2.5-Coder-0.5B", "answerdotai/ModernBERT-base"],
                      help="List of model names to download")
    parser.add_argument("--datasets", nargs="+", default=["codexglue", "bigvul"],
                      help="List of datasets to download")
    
    args = parser.parse_args()
    
    # Set up cache environment
    cache_dir = setup_cache_env()
    logger.info(f"Using cache directory: {cache_dir}")
    logger.info(f"Models will be cached in: {cache_dir / 'hub'}")
    logger.info(f"Datasets will be cached in: {cache_dir / 'datasets'}")
    
    # Download models
    for model_name in args.models:
        try:
            # Try translation task first
            download_model(model_name, task_type="translation", cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to download {model_name} as translation model: {str(e)}")
            try:
                # Try classification task
                download_model(model_name, task_type="classification", cache_dir=cache_dir)
            except Exception as e:
                logger.error(f"Failed to download {model_name} as classification model: {str(e)}")
    
    # Download datasets
    for dataset_name in args.datasets:
        try:
            download_dataset(dataset_name, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {str(e)}")
    
    logger.info("Cache preparation completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nCache preparation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cache preparation failed: {str(e)}")
        sys.exit(1) 