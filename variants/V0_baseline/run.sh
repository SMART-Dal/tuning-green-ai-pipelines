#!/bin/bash
# run.sh for variant V0_baseline
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to Python path
export PROJECT_ROOT=$(pwd)

# Create necessary directories
mkdir -p results/V0_baseline/{logs,cache,code_translation,vulnerability_detection}

# Run the pipeline with specified task and stages
# Usage: ./run.sh [task] [stage1 stage2 ...]
# task can be: code_translation, vulnerability_detection, or all
# stages can be: data, architecture, training, inference, system, or all
python run.py "$@"
