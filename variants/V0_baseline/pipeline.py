import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import common utilities
from common.data import load_codexglue_dataset, load_bigvul_dataset
from common.models import get_model
from common.energy import EnergyMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselinePipeline:
    def __init__(self, output_dir=None):
        """Initialize the baseline pipeline.
        
        Args:
            output_dir (str, optional): Directory to save results. If None, uses project_root/results
        """
        if output_dir is None:
            output_dir = os.path.join(project_root, "results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.energy_monitor = EnergyMonitor()
        
        # Initialize results dictionary
        self.results = {
            "variant": "V0_baseline",
            "energy_kwh": {
                "data": 0.0,
                "architecture": 0.0,
                "training": 0.0,
                "system": 0.0,
                "inference": 0.0,
                "total": 0.0
            },
            "co2_kg": 0.0,
            "accuracy": 0.0,
            "latency_ms": 0.0
        }

    def run_task1(self):
        """Run code-to-code translation task using SmalLm"""
        logger.info("Starting Task 1: Code-to-Code Translation")
        
        # Data preparation
        with self.energy_monitor.measure("data"):
            train_dataset, val_dataset, test_dataset = load_codexglue_dataset()
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-2.5B")
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

        # Model setup
        with self.energy_monitor.measure("architecture"):
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-2.5B")
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Training
        with self.energy_monitor.measure("training"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            model.train()
            for epoch in range(3):  # Baseline: 3 epochs
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

        # Inference
        with self.energy_monitor.measure("inference"):
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(**batch)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.results["latency_ms"] = latency

    def run_task2(self):
        """Run vulnerability classification task using ModernBERT"""
        logger.info("Starting Task 2: Vulnerability Classification")
        
        # Data preparation
        with self.energy_monitor.measure("data"):
            train_dataset, val_dataset, test_dataset = load_bigvul_dataset()
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

        # Model setup
        with self.energy_monitor.measure("architecture"):
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Training
        with self.energy_monitor.measure("training"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            model.train()
            for epoch in range(3):  # Baseline: 3 epochs
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

        # Inference and accuracy calculation
        with self.energy_monitor.measure("inference"):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch["labels"]).sum().item()
                    total += len(batch["labels"])
            
            accuracy = correct / total
            self.results["accuracy"] = accuracy

    def run(self):
        """Run the complete pipeline"""
        try:
            # Run both tasks in sequence
            self.run_task1()
            self.run_task2()
            
            # Get final energy measurements
            energy_metrics = self.energy_monitor.get_metrics()
            self.results["energy_kwh"] = energy_metrics
            self.results["co2_kg"] = energy_metrics["total"] * 0.233  # Example CO2 conversion factor
            
            # Save results
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = self.output_dir / f"V0_baseline_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Pipeline completed. Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = BaselinePipeline()
    pipeline.run() 