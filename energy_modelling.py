#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
from pathlib import Path

def calculate_savings(baseline_energy, variant_energy):
    """Calculate energy savings percentage"""
    if baseline_energy == 0:
        return 0.0
    return (baseline_energy - variant_energy) / baseline_energy

def run_modelling(results_dir, baseline, output_file):
    # Load metrics data
    metrics_path = os.path.join(results_dir, "combined_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Error: Could not find combined_metrics.json in {results_dir}")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Create variants dictionary from the list
    variants = {v["variant"].lower(): v for v in data}
    
    # Get baseline energy
    baseline = baseline.lower()
    baseline_energy = variants[baseline]["total_kwh"]
    
    # Define combined variants and their components
    combined_variants = {
        "v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation": {
            "name": "V23 (Attention+PinMem+Optim+GradAcc)",
            "components": ["v12_attention", "v10_dataloader_pin_memory", "v6_optimizer"]
        },
        "v24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16": {
            "name": "V24 (InfEngine+GradCkpt+LoRA+FP16)",
            "components": ["v9_inference_engine", "v1_gradient_checkpointing", "v2_lora_peft", "v7_f16"]
        },
        "v25_gradient_accumulation_plus_fp16_plus_checkpointing": {
            "name": "V25 (GradAcc+FP16+Ckpt)",
            "components": ["v1_gradient_checkpointing", "v7_f16"]
        },
        "v26_pruning_plus_seq_lngth_plus_torch_compile": {
            "name": "V26 (Prune+SeqLen+Compile)",
            "components": ["v18_layer_pruning_12_bottom", "v8_sequence_length_trimming", "v11_torch_compile"]
        },
        "v27_torch_compile_plus_fp16": {
            "name": "V27 (Compile+FP16)",
            "components": ["v11_torch_compile", "v7_f16"]
        },
        "v28_pruning_plus_torch_compile_plus_fp16": {
            "name": "V28 (Prune+Compile+FP16)",
            "components": ["v18_layer_pruning_12_bottom", "v11_torch_compile", "v7_f16"]
        },
        "v29_attention_plus_pin_memory_plus_optimizer": {
            "name": "V29",
            "components": ["v12_attention", "v10_dataloader_pin_memory", "v6_optimizer"]
        },
        "v30_optimal": {
            "name": "V30 (Optimal)",
            "components": ["v9_inference_engine", "v2_lora_peft", "v7_f16", "v11_torch_compile", "v21_layer_pruning_20_top"]
        }
    }
    
    # Prepare output data
    calculations = []
    
    # Process each combined variant
    for variant_id, variant_info in combined_variants.items():
        variant_id = variant_id.lower()
        if variant_id not in variants:
            continue
            
        # Get observed savings
        variant_energy = variants[variant_id]["total_kwh"]
        observed_savings = calculate_savings(baseline_energy, variant_energy)
        
        # Calculate expected savings
        component_savings = []
        valid_components = []
        for comp_id in variant_info["components"]:
            comp_id = comp_id.lower()
            if comp_id not in variants:
                print(f"Warning: Component {comp_id} not found in variants")
                continue
            comp_energy = variants[comp_id]["total_kwh"]
            s_i = calculate_savings(baseline_energy, comp_energy)
            component_savings.append(s_i)
            valid_components.append(comp_id)
        
        # Compute multiplicative cascade
        product = 1.0
        for s_i in component_savings:
            product *= (1 - s_i)
        expected_savings = 1 - product
        
        # Calculate delta
        delta = expected_savings - observed_savings
        
        # Format calculations
        calc_steps = []
        for i, (s_i, comp) in enumerate(zip(component_savings, valid_components)):
            calc_steps.append(f"  s{i+1} = {s_i:.3f} (from {comp})")
        
        # Use component_savings for product_str, annotate with component names (unique per variant)
        product_str = " * ".join([f"(1 - {s_i:.3f}) [{comp}]" for s_i, comp in zip(component_savings, valid_components)])
        
        calculations.append({
            "variant": variant_info["name"],
            "observed": observed_savings,
            "expected": expected_savings,
            "delta": delta,
            "components": variant_info["components"],
            "calc_steps": calc_steps,
            "product_str": product_str
        })
    
    # Write output to file
    with open(output_file, "w") as f:
        f.write("Modelling Calculations - Multiplicative Cascade Framework\n")
        f.write("========================================================\n\n")
        f.write(f"Baseline: {baseline}\n")
        f.write(f"Baseline Energy: {baseline_energy:.6f} kWh\n\n")
        
        for calc in calculations:
            f.write(f"Variant: {calc['variant']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Observed Savings: {calc['observed']:.3f} ({calc['observed']*100:.1f}%)\n")
            f.write(f"Expected Savings: {calc['expected']:.3f} ({calc['expected']*100:.1f}%)\n")
            f.write(f"Delta: {calc['delta']:.3f} ({calc['delta']*100:.1f}%)\n\n")
            
            f.write("Calculation Steps:\n")
            f.write(f"  S_combined = 1 - ∏(1 - s_i)\n")
            for step in calc['calc_steps']:
                f.write(step + "\n")
            
            f.write(f"  = 1 - [{calc['product_str']}]\n")
            f.write(f"  = {calc['expected']:.3f}\n\n")
            
            f.write(f"Deviation Analysis:\n")
            f.write(f"  Observed: {calc['observed']*100:.1f}%\n")
            f.write(f"  Expected: {calc['expected']*100:.1f}%\n")
            f.write(f"  Difference: {calc['delta']*100:.1f} percentage points\n")
            f.write("=" * 50 + "\n\n")

def main():
    import sys
    parser = argparse.ArgumentParser(description='Run multiplicative cascade modelling for combined variants')
    parser.add_argument('results_dir', type=str, nargs='?', default='analysis_results', help='Path to results directory containing combined_metrics.json')
    parser.add_argument('--baseline', type=str, default='v0_baseline', help='Baseline variant name')
    parser.add_argument('--output', type=str, default='modelling_calculations.txt', help='Output file path')
    args = parser.parse_args()
    
    run_modelling(args.results_dir, args.baseline, args.output)
    print(f"Modelling calculations saved to {args.output}")

if __name__ == "__main__":
    main()