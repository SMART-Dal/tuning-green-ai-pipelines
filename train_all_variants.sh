#!/bin/bash

# List of variants to run
variants=(
    # Baseline and single optimizations
    "v0_baseline"
    "v1_gradient_checkpointing"
    "v2_lora_peft"
    "v3_quantization"
    "v4_tokenizer"
    "v5_power_limit_100W"
    "v6_optimizer"
    "v7_f16"
    "v8_sequence_length_trimming"
    "v9_inference_engine"
    "v10_dataloader_pin_memory"
    "v11_torch_compile"
    "V12_attention"

    # Layer pruning variants
    "v13_layer_pruning_4_top"
    "v14_layer_pruning_4_bottom"
    "v15_layer_pruning_8_top"
    "v16_layer_pruning_8_bottom"
    "v17_layer_pruning_12_top"
    "v18_layer_pruning_12_bottom"
    "v19_layer_pruning_16_top"
    "v20_layer_pruning_16_bottom"
    "v21_layer_pruning_20_top"
    "v22_layer_pruning_20_bottom"

    # Combined optimizations
    "v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation"
    "v24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16"
    "v25_gradient_accumulation_plus_fp16_plus_checkpointing"
    "v26_pruning_plus_seq_lngth_plus_torch_compile"
    "v27_torch_compile_plus_fp16"
    "v28_pruning_plus_torch_compile_plus_fp16"
    "v29_attention_plus_pin_memory_plus_optimizer"
    "v30_optimal"
)

root_dir=~/greenai-pipeline-empirical-study/variants
num_runs=2  # Number of times to run each variant

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

# Run each variant multiple times
for run in $(seq 1 $num_runs); do
    echo "Starting Run $run of $num_runs"
    
    for variant in "${variants[@]}"; do
        echo "Starting variant: $variant (Run $run)"
        
        # Change to variant directory
        cd "$root_dir/$variant" || {
            echo "Error: Could not change to directory $variant"
            exit 1
        }
        
        # Clean up any existing processes for this variant
        pkill -f "python3 train.py" || true
        
        # Create run-specific log file
        run_log="run_${run}_log.txt"
        
        # Start the training process
        python3 train.py > "$run_log" 2>&1
        training_exit_code=$?
        
        if [ $training_exit_code -ne 0 ]; then
            echo "Error: Training failed for variant $variant (Run $run) with exit code $training_exit_code"
            cleanup
        fi
        
        echo "Variant $variant (Run $run) completed"
        
        # Wait 60 seconds before starting the next variant
        echo "Waiting 60 seconds before starting next variant..."
        sleep 60
        
        # Go back to the original directory
        cd "$root_dir"
    done
    
    echo "Completed Run $run of $num_runs"
    
    # Wait 5 minutes between runs
    if [ $run -lt $num_runs ]; then
        echo "Waiting 5 minutes before starting next run..."
        sleep 300
    fi
done

echo "All variants have been processed for all runs"





# pick best performning variant across stages and combine and run the model