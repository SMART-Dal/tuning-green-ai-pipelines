#!/bin/bash

# List of variants to run
variants=(
    "v11_torch_compile"
    "v26_pruning_plus_seq_lngth_torch_compile"
    "v28_pruning_plus_torch_compile_fp16"
    "v29_attention_plus_pin_memory_optimizer_gradient_accumulation"
    # Add more variants here
)

root_dir=~/greenai-pipeline-empirical-study/variants

# Function to check if a process is still running
check_process() {
    local pid=$1
    if ps -p $pid > /dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Function to cleanup any existing processes
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "python3 train.py" || true
    exit 1
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Run each variant
for variant in "${variants[@]}"; do
    echo "Starting variant: $variant"
    
    # Change to variant directory
    cd "$root_dir/$variant" || {
        echo "Error: Could not change to directory $variant"
        exit 1
    }
    
    # Clean up any existing processes for this variant
    pkill -f "python3 train.py" || true
    
    # Start the training process
    python3 train.py > run_log.txt 2>&1
    training_exit_code=$?
    
    if [ $training_exit_code -ne 0 ]; then
        echo "Error: Training failed for variant $variant with exit code $training_exit_code"
        cleanup
    fi
    
    echo "Variant $variant completed"
    
    # Wait 60 seconds before starting the next variant
    echo "Waiting 60 seconds before starting next variant..."
    sleep 60
    
    # Go back to the original directory
    cd "$root_dir"
done

echo "All variants have been processed"





# pick best performning variant across stages and combine and run the model