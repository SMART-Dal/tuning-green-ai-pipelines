#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run pipeline for specified task with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Get task to run from command line
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task not in ["code_translation", "vulnerability_detection", "all"]:
        logger.error(f"Invalid task: {task}. Must be one of: code_translation, vulnerability_detection, all")
        sys.exit(1)
    
    # Get stages to run
    stages = sys.argv[2:] if len(sys.argv) > 2 else ["all"]
    
    # Run specified task(s)
    if task in ["code_translation", "all"]:
        logger.info("Running code translation task...")
        code_translation_script = Path(__file__).parent / "tasks" / "code_translation" / "run.py"
        os.system(f"python {code_translation_script} {' '.join(stages)}")
    
    if task in ["vulnerability_detection", "all"]:
        logger.info("Running vulnerability detection task...")
        vulnerability_detection_script = Path(__file__).parent / "tasks" / "vulnerability_detection" / "run.py"
        os.system(f"python {vulnerability_detection_script} {' '.join(stages)}")

if __name__ == "__main__":
    main() 