#!/bin/bash

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a log directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Get the current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/training_${TIMESTAMP}.log"

# Function to run training for a variant
run_variant() {
    local variant=$1
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "Starting training for variant: $variant" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
    
    # Change to the variant directory and run train.py
    cd "$variant" || {
        echo "Error: Could not change to directory $variant" | tee -a "$LOG_FILE"
        return 1
    }
    
    # Run the training script
    python train.py 2>&1 | tee -a "$LOG_FILE"
    
    # Return to the original directory
    cd "$SCRIPT_DIR"
    
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "Completed training for variant: $variant" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Main execution
echo "Starting training for all variants at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE"

# Find all variant directories (V* or v*)
for variant in "$SCRIPT_DIR/variants/V"* "$SCRIPT_DIR/variants/v"*; do
    if [ -d "$variant" ] && [ -f "$variant/train.py" ]; then
        run_variant "$variant"
    fi
done

echo "All training runs completed at $(date)" | tee -a "$LOG_FILE" 