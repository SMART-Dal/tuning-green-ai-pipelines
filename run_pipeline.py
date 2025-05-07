#!/usr/bin/env python3
"""Main entry point for running pipeline variants."""

import os
import sys
import importlib.util
from pathlib import Path

def run_variant(variant_name):
    """Run a specific pipeline variant.
    
    Args:
        variant_name (str): Name of the variant to run (e.g., 'V0_baseline')
    """
    # Get the variant directory
    variant_dir = Path(__file__).parent / "variants" / variant_name
    if not variant_dir.exists():
        raise ValueError(f"Variant directory not found: {variant_dir}")

    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Import the pipeline module
    pipeline_path = variant_dir / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)

    # Run the pipeline
    pipeline = pipeline_module.BaselinePipeline()
    pipeline.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <variant_name>")
        print("Example: python run_pipeline.py V0_baseline")
        sys.exit(1)
    
    variant_name = sys.argv[1]
    run_variant(variant_name) 