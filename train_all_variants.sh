#!/bin/bash

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create logs and status directories
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/status"

# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
STATUS_FILE="$SCRIPT_DIR/status/experiment_status.txt"

# Function to run training for a variant in tmux
run_variant() {
    local variant=$1
    local variant_name=$(basename "$variant")
    local log_file="$SCRIPT_DIR/logs/${variant_name}_${TIMESTAMP}.log"
    local session_name="exp_${variant_name}_${TIMESTAMP}"
    
    echo "===============================================" | tee -a "$STATUS_FILE"
    echo "Starting training for variant: $variant_name" | tee -a "$STATUS_FILE"
    echo "Time: $(date)" | tee -a "$STATUS_FILE"
    echo "Log file: $log_file" | tee -a "$STATUS_FILE"
    echo "TMUX session: $session_name" | tee -a "$STATUS_FILE"
    echo "===============================================" | tee -a "$STATUS_FILE"
    
    # Change to the variant directory and run in tmux
    cd "$variant" || {
        echo "Error: Could not change to directory $variant" | tee -a "$STATUS_FILE"
        echo "Status: Failed - Directory not found" | tee -a "$STATUS_FILE"
        return 1
    }
    
    # Create new tmux session and run training
    tmux new-session -d -s "$session_name" "python3 train.py 2>&1 | tee \"$log_file\""
    
    # Return to the original directory
    cd "$SCRIPT_DIR"
    
    # Add to status file
    echo "$variant_name|$session_name|Running|$(date)|$log_file" >> "$STATUS_FILE"
    
    # Wait for 5 minutes before starting next variant
    echo "Waiting 5 minutes before starting next variant..." | tee -a "$STATUS_FILE"
    sleep 300
}

# Initialize status file
echo "Variant|TMUX Session|Status|Start Time|Log File" > "$STATUS_FILE"
echo "----------------------------------------" >> "$STATUS_FILE"

# Main execution
echo "Starting training at $(date)" | tee -a "$STATUS_FILE"

VARIANTS_DIR="variants"

# If specific variants are provided as arguments, use those
if [ $# -gt 0 ]; then
    variants=("$@")
    echo "Running specified variants: ${variants[*]}" | tee -a "$STATUS_FILE"
else
    # Get all variant directories and sort them
    variants=($(ls -d ${VARIANTS_DIR}/V* ${VARIANTS_DIR}/v* | sort -V))
    echo "Running all variants: ${variants[*]}" | tee -a "$STATUS_FILE"
fi

echo "Found ${#variants[@]} variants"

# Run each variant
for variant in "${variants[@]}"; do
    run_variant "$variant"
done
echo "All training runs initiated at $(date)" | tee -a "$STATUS_FILE"
echo "Check $STATUS_FILE for status updates"
echo "Use 'tmux ls' to see active sessions"
echo "Attach to session: tmux attach -t <session_name>"
echo "Detach from session: Ctrl+B then D"
