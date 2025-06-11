#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, statistics, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from datetime import datetime
from adjustText import adjust_text

# ------------------------------------------------------------------
# ------------------------------ helpers ---------------------------
# ------------------------------------------------------------------

STAGE_ORDER = [
    "load_dataset",
    "tokenize_dataset",
    "load_model",
    "train_model",
    "save_model",
    "evaluate_model",
]

# Mapping of variant folder names to readable names
VARIANT_NAMES = {
    "V0_baseline": "V0",
    "V1_gradient_checkpointing": "V1",
    "V2_lora_peft": "V2",
    "V3_quantization": "V3",
    "V4_tokenizer": "V4",
    "V5_power_limit_100W": "V5",
    "V6_optimizer": "V6",
    "V7_f16": "V7",
    "V8_sequence_length_trimming": "V8",
    "V9_inference_engine": "V9",
    "V10_dataloader_pin_memory": "V10",
    "v11_torch_compile": "V11",
    "V12_attention": "V12",
    "v13_layer_pruning_4_top": "V13",
    "v14_layer_pruning_4_bottom": "V14",
    "v15_layer_pruning_8_top": "V15",
    "v16_layer_pruning_8_bottom": "V16",
    "v17_layer_pruning_12_top": "V17",
    "v18_layer_pruning_12_bottom": "V18",
    "v19_layer_pruning_16_top": "V19",
    "v20_layer_pruning_16_bottom": "V20",
    "v21_layer_pruning_20_top": "V21",
    "v22_layer_pruning_20_bottom": "V22",
    "v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation": "V23",
    "V24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16": "V24",
    "V25_gradient_accumulation_plus_fp16_plus_checkpointing": "V25",
    "v26_pruning_plus_seq_lngth_plus_torch_compile": "V26",
    "v27_torch_compile_plus_fp16": "V27",
    "v28_pruning_plus_torch_compile_plus_fp16": "V28"
}

# Add variant descriptions
VARIANT_DESCRIPTIONS = {
    "V0_baseline": "Baseline",
    "V1_gradient_checkpointing": "Gradient checkpointing",
    "V2_lora_peft": "LoRA PEFT",
    "V3_quantization": "Quantization",
    "V4_tokenizer": "Tokenizer optimization",
    "V5_power_limit_100W": "Power limit (100W)",
    "V6_optimizer": "Optimizer tuning",
    "V7_f16": "FP16 training",
    "V8_sequence_length_trimming": "Sequence length trimming",
    "V9_inference_engine": "Inference engine",
    "V10_dataloader_pin_memory": "Dataloader pin memory",
    "v11_torch_compile": "Torch compile",
    "V12_attention": "Attention optimization",
    "v13_layer_pruning_4_top": "Layer pruning (4 Top)",
    "v14_layer_pruning_4_bottom": "Layer pruning (4 Bottom)",
    "v15_layer_pruning_8_top": "Layer pruning (8 Top)",
    "v16_layer_pruning_8_bottom": "Layer pruning (8 Bottom)",
    "v17_layer_pruning_12_top": "Layer pruning (12 Top)",
    "v18_layer_pruning_12_bottom": "Layer pruning (12 Bottom)",
    "v19_layer_pruning_16_top": "Layer pruning (16 Top)",
    "v20_layer_pruning_16_bottom": "Layer pruning (16 Bottom)",
    "v21_layer_pruning_20_top": "Layer pruning (20 Top)",
    "v22_layer_pruning_20_bottom": "Layer pruning (20 Bottom)",
    "v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation": "Attention + pin mem + 8-bit + grad accum",
    "V24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16": "Inf engine + grad ckpt + LoRA + FP16",
    "V25_gradient_accumulation_plus_fp16_plus_checkpointing": "Grad accum + FP16 + ckpt",
    "v26_pruning_plus_seq_lngth_plus_torch_compile": "Pruning + seq len + compile",
    "v27_torch_compile_plus_fp16": "Torch compile + FP16",
    "v28_pruning_plus_torch_compile_plus_fp16": "Pruning + compile + FP16"
}

def get_variant_name(variant: str) -> str:
    """Get the display name for a variant."""
    # Convert to lowercase for consistent handling
    variant = variant.lower()
    # Handle v prefix
    if variant.startswith('v'):
        # Remove the prefix and convert to title case
        name = variant[1:].replace('_', ' ').title()
        # Special case for baseline
        if name.lower() == '0 baseline':
            return 'Baseline'
        return name
    return variant

def extract_variant_number(variant: str) -> int:
    """Extract the numeric part from a variant name for sorting."""
    try:
        # Convert to lowercase for consistent handling
        variant = variant.lower()
        # Handle v prefix
        if variant.startswith('v'):
            return int(variant[1:].split('_')[0])
        return 0
    except (ValueError, IndexError):
        return 0

INFERENCE_METRICS = [
    "inference_energy",
    "inference_time",
    "throughput_qps",
    "latency_ms"
]

def _safe_mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")

def _safe_stdev(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

def calculate_deltas(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Calculate deltas relative to baseline variant"""
    if df.empty:
        return df
        
    # Get baseline data - case insensitive comparison
    baseline_mask = df['variant'].str.lower() == baseline.lower()
    if not baseline_mask.any():
        print(f"Warning: Baseline variant {baseline} not found in data")
        return df
        
    baseline_data = df[baseline_mask].iloc[0]
    deltas = df.copy()
    
    # Calculate percentage changes for energy metrics
    for col in ['total_kwh', 'train_energy', 'inference_energy', 'eval_time_s']:
        if col in df.columns:
            # Calculate percentage change, handling zero baseline values
            baseline_value = baseline_data[col]
            if baseline_value == 0:
                # If baseline is 0, set delta to 0 to avoid division by zero
                deltas[f'Δ{col}'] = 0
            else:
                # Calculate percentage change: ((new - old) / old) * 100
                deltas[f'Δ{col}'] = ((deltas[col] - baseline_value) / baseline_value) * 100
    
    # Calculate absolute difference for runtime
    if 'runtime_s' in df.columns:
        deltas['Δruntime_s'] = deltas['runtime_s'] - baseline_data['runtime_s']
            
    # Calculate absolute changes for metrics
    for col in ['f1', 'accuracy']:
        if col in df.columns:
            deltas[f'Δ{col}'] = deltas[col] - baseline_data[col]
            
    # Calculate percentage differences for key metrics
    if 'total_kwh' in df.columns:
        baseline_energy = baseline_data['total_kwh']
        deltas['percent_diff_energy'] = ((deltas['total_kwh'] - baseline_energy) / baseline_energy * 100) if baseline_energy != 0 else 0
    
    if 'runtime_s' in df.columns:
        baseline_runtime = baseline_data['runtime_s']
        deltas['percent_diff_time'] = ((deltas['runtime_s'] - baseline_runtime) / baseline_runtime * 100) if baseline_runtime != 0 else 0
    
    if 'eval_time_s' in df.columns:
        baseline_eval_time = baseline_data['eval_time_s']
        deltas['percent_diff_eval_time'] = ((deltas['eval_time_s'] - baseline_eval_time) / baseline_eval_time * 100) if baseline_eval_time != 0 else 0
    
    if 'f1' in df.columns:
        baseline_f1 = baseline_data['f1']
        deltas['percent_diff_f1'] = ((deltas['f1'] - baseline_f1) / baseline_f1 * 100) if baseline_f1 != 0 else 0
            
    # Calculate efficiency metrics
    if 'f1' in df.columns and 'total_kwh' in df.columns:
        deltas['f1_per_kwh'] = deltas['f1'] / deltas['total_kwh']
        
    return deltas

def identify_pareto_frontier(df: pd.DataFrame, energy_col: str, perf_col: str) -> pd.DataFrame:
    """Identify Pareto optimal points in energy/performance trade-off"""
    points = df[[energy_col, perf_col]].values
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        if pareto_mask[i]:
            # Dominated if any point has lower energy AND higher performance
            mask = (points[:,0] <= point[0]) & (points[:,1] >= point[1])
            mask[i] = False  # Don't compare to self
            if np.any(mask):
                pareto_mask[i] = False
                
    return df[pareto_mask]

# ------------------------------------------------------------------
# ------------------------ data collection -------------------------
# ------------------------------------------------------------------

def walk_results(root: Path):
    """Yield (variant, run_dir). `variant` is the variant name from the directory structure."""
    print(f"Searching for variants in: {root}")
    for variant_dir in root.iterdir():
        if not variant_dir.is_dir() or not variant_dir.name.startswith(('V', 'v')):
            continue
        print(f"Found variant directory: {variant_dir.name}")
        # Look for results directory
        results_dir = variant_dir / "results"
        if not results_dir.exists():
            print(f"No results directory found in {variant_dir.name}")
            continue
        # Get all run directories
        run_dirs = sorted(results_dir.glob("default_*"), key=lambda x: x.name)
        if not run_dirs:
            print(f"No run directories found in {results_dir}")
            continue
        print(f"Found {len(run_dirs)} runs for {variant_dir.name}")
        # Convert variant name to lowercase for consistent comparison
        variant_name = variant_dir.name.lower()
        for run_dir in run_dirs:
            yield variant_name, run_dir


def load_energy(path: Path) -> Dict[str, Any]:
    """Load energy stats from JSON file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load test metrics from JSON file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def load_inference_metrics(path: Path) -> Dict[str, Any]:
    """Load inference metrics from JSON file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

# ------------------------------------------------------------------
# ---------------------- aggregation logic -------------------------
# ------------------------------------------------------------------

def aggregate(results_root: Path, baseline: str = "v0_baseline"):
    """Aggregate results from all variants and store in a single JSON file."""
    variant_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    # First pass: collect all run data
    for variant, run_dir in walk_results(results_root):
        print(f"\nProcessing variant: {variant} | Run: {run_dir.name}")
        
        # Training metrics
        train_energy_path = run_dir / "energy_stats_train.json"
        test_metrics_path = run_dir / "test_metrics.json"
        emissions_path = next(run_dir.glob("emissions_base_*.csv"), None)
        
        # Inference metrics
        inference_energy_path = run_dir / "energy_stats_inference.json"
        inference_metrics_path = run_dir / "inference_metrics.json"
        hardware_metrics_path = run_dir / "hardware_stats.json"
        
        data = {
            "train_energy": load_energy(train_energy_path),
            "test_metrics": load_metrics(test_metrics_path),
            "inference_energy": load_energy(inference_energy_path),
            "inference_metrics": load_inference_metrics(inference_metrics_path),
            "hardware_metrics": load_metrics(hardware_metrics_path) if hardware_metrics_path.exists() else {},
            "emissions_path": str(emissions_path) if emissions_path else None,
            "run_dir": str(run_dir),
            "variant": variant,
            "run_id": run_dir.name
        }
        variant_runs[variant].append(data)

    print("\nCollected data for variants:", list(variant_runs.keys()))
    
    # Prepare aggregated data structure
    aggregated_data = []
    
    for variant, runs in variant_runs.items():
        print(f"\nAggregating data for {variant} ({len(runs)} runs)")
        
        # Get variant metadata
        variant_name = get_variant_name(variant)
        variant_number = extract_variant_number(variant)
        
        # Training metrics
        train_energies = [r["train_energy"].get("energy_consumed", 0) for r in runs]
        train_times = [r["train_energy"].get("duration", 0) for r in runs]
        cpu_energies = [r["train_energy"].get("cpu_energy", 0) for r in runs]
        gpu_energies = [r["train_energy"].get("gpu_energy", 0) for r in runs]
        ram_energies = [r["train_energy"].get("ram_energy", 0) for r in runs]
        peak_mems = [r["train_energy"].get("max_gpu_mem", 0) for r in runs]
        
        # Calculate total runtime as sum of all stage durations
        total_runtimes = []
        for run in runs:
            if run["emissions_path"] and Path(run["emissions_path"]).exists():
                df_stages = pd.read_csv(run["emissions_path"])
                total_runtime = df_stages['duration'].sum()
                total_runtimes.append(total_runtime)
            else:
                total_runtimes.append(0)
        
        # Test metrics
        f1_scores = [r["test_metrics"].get("eval_f1", 0) for r in runs]
        accuracies = [r["test_metrics"].get("eval_accuracy", 0) for r in runs]
        
        # Inference metrics
        inf_energies = [r["inference_energy"].get("energy_consumed", 0) for r in runs]
        inf_times = [r["inference_energy"].get("duration", 0) for r in runs]
        throughputs = [r["inference_metrics"].get("throughput_qps", 0) for r in runs]
        latencies = [r["inference_metrics"].get("latency_ms", 0) for r in runs]
        
        # Hardware metrics
        gpu_utils = [r["hardware_metrics"].get("avg_gpu_util", 0) for r in runs]
        mem_utils = [r["hardware_metrics"].get("avg_gpu_mem_util", 0) for r in runs]
        
        # Calculate aggregated metrics
        variant_data = {
            "metadata": {
                "variant": variant,
                "variant_name": variant_name,
                "variant_number": variant_number,
                "num_runs": len(runs),
                "run_ids": [r["run_id"] for r in runs]
            },
            "training_metrics": {
                "total_energy": {
                    "mean": _safe_mean(train_energies),
                    "std": _safe_stdev(train_energies),
                    "raw_values": train_energies
                },
                "runtime": {
                    "mean": _safe_mean(total_runtimes),
                    "std": _safe_stdev(total_runtimes),
                    "raw_values": total_runtimes
                },
                "cpu_energy": {
                    "mean": _safe_mean(cpu_energies),
                    "std": _safe_stdev(cpu_energies),
                    "raw_values": cpu_energies
                },
                "gpu_energy": {
                    "mean": _safe_mean(gpu_energies),
                    "std": _safe_stdev(gpu_energies),
                    "raw_values": gpu_energies
                },
                "ram_energy": {
                    "mean": _safe_mean(ram_energies),
                    "std": _safe_stdev(ram_energies),
                    "raw_values": ram_energies
                },
                "peak_memory": {
                    "mean": _safe_mean(peak_mems),
                    "std": _safe_stdev(peak_mems),
                    "raw_values": peak_mems
                }
            },
            "test_metrics": {
                "f1_score": {
                    "mean": _safe_mean(f1_scores),
                    "std": _safe_stdev(f1_scores),
                    "raw_values": f1_scores
                },
                "accuracy": {
                    "mean": _safe_mean(accuracies),
                    "std": _safe_stdev(accuracies),
                    "raw_values": accuracies
                }
            },
            "inference_metrics": {
                "energy": {
                    "mean": _safe_mean(inf_energies),
                    "std": _safe_stdev(inf_energies),
                    "raw_values": inf_energies
                },
                "time": {
                    "mean": _safe_mean(inf_times),
                    "std": _safe_stdev(inf_times),
                    "raw_values": inf_times
                },
                "throughput": {
                    "mean": _safe_mean(throughputs),
                    "std": _safe_stdev(throughputs),
                    "raw_values": throughputs
                },
                "latency": {
                    "mean": _safe_mean(latencies),
                    "std": _safe_stdev(latencies),
                    "raw_values": latencies
                }
            },
            "hardware_metrics": {
                "gpu_utilization": {
                    "mean": _safe_mean(gpu_utils),
                    "std": _safe_stdev(gpu_utils),
                    "raw_values": gpu_utils
                },
                "memory_utilization": {
                    "mean": _safe_mean(mem_utils),
                    "std": _safe_stdev(mem_utils),
                    "raw_values": mem_utils
                }
            },
            "stage_data": []
        }
        
        # Add stage-wise data
        for run in runs:
            if run["emissions_path"] and Path(run["emissions_path"]).exists():
                df_stages = pd.read_csv(run["emissions_path"])
                for _, stage_row in df_stages.iterrows():
                    variant_data["stage_data"].append({
                        "run_id": run["run_id"],
                        "stage": stage_row['task_name'],
                        "energy": stage_row['energy_consumed'],
                        "duration": stage_row['duration']
                    })
        
        aggregated_data.append(variant_data)
        print(f"Added summary for {variant}")
    
    # Sort variants by variant number
    aggregated_data.sort(key=lambda x: x["metadata"]["variant_number"])
    
    # Save aggregated data to JSON
    output_path = results_root / "aggregated_metrics.json"
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "baseline": baseline,
                "timestamp": datetime.now().isoformat(),
                "total_variants": len(aggregated_data)
            },
            "variants": aggregated_data
        }, f, indent=2)
    
    print(f"\nAggregated metrics saved to {output_path}")
    
    # Convert to DataFrames for compatibility with existing analysis functions
    df_variant = pd.DataFrame([{
        "variant": v["metadata"]["variant"],
        "total_kwh": v["training_metrics"]["total_energy"]["mean"],
        "total_kwh_std": v["training_metrics"]["total_energy"]["std"],
        "runtime_s": v["training_metrics"]["runtime"]["mean"],
        "runtime_s_std": v["training_metrics"]["runtime"]["std"],
        "cpu_kwh": v["training_metrics"]["cpu_energy"]["mean"],
        "gpu_kwh": v["training_metrics"]["gpu_energy"]["mean"],
        "ram_kwh": v["training_metrics"]["ram_energy"]["mean"],
        "f1": v["test_metrics"]["f1_score"]["mean"],
        "f1_std": v["test_metrics"]["f1_score"]["std"],
        "accuracy": v["test_metrics"]["accuracy"]["mean"],
        "peak_mem_gb": v["training_metrics"]["peak_memory"]["mean"],
        "inference_energy": v["inference_metrics"]["energy"]["mean"],
        "inference_energy_std": v["inference_metrics"]["energy"]["std"],
        "inference_time": v["inference_metrics"]["time"]["mean"],
        "throughput_qps": v["inference_metrics"]["throughput"]["mean"],
        "latency_ms": v["inference_metrics"]["latency"]["mean"],
        "avg_gpu_util": v["hardware_metrics"]["gpu_utilization"]["mean"],
        "avg_gpu_mem_util": v["hardware_metrics"]["memory_utilization"]["mean"],
        "num_runs": v["metadata"]["num_runs"],
        "eval_time_s": next((stage["duration"] for stage in v["stage_data"] if stage["stage"] == "evaluate_model"), 0.0)  # Get duration of evaluate_model stage
    } for v in aggregated_data])
    
    # Convert variant names to lowercase in all DataFrames
    df_variant['variant'] = df_variant['variant'].astype(str).str.strip().str.lower()
    
    # Create stage DataFrame
    stage_rows = []
    for variant_data in aggregated_data:
        for stage in variant_data["stage_data"]:
            stage_rows.append({
                "variant": variant_data["metadata"]["variant"].lower(),  # Convert to lowercase
                "run_id": stage["run_id"],
                "stage": stage["stage"],
                "kwh": stage["energy"],
                "duration": stage["duration"]
            })
    df_stage = pd.DataFrame(stage_rows) if stage_rows else pd.DataFrame()
    
    # Create inference DataFrame
    inference_rows = []
    for variant_data in aggregated_data:
        for run_id in variant_data["metadata"]["run_ids"]:
            inference_rows.append({
                "variant": variant_data["metadata"]["variant"].lower(),  # Convert to lowercase
                "run_id": run_id,
                "throughput_qps": variant_data["inference_metrics"]["throughput"]["mean"],
                "latency_ms": variant_data["inference_metrics"]["latency"]["mean"]
            })
    df_inference = pd.DataFrame(inference_rows) if inference_rows else pd.DataFrame()
    
    # Calculate deltas relative to baseline
    df_variant = calculate_deltas(df_variant, baseline.lower())  # Convert baseline to lowercase
    
    # Identify Pareto frontier for energy/performance trade-off
    df_pareto = identify_pareto_frontier(
        df_variant, 
        'percent_diff_energy',
        'f1'
    )
    df_variant['on_pareto'] = df_variant.index.isin(df_pareto.index)
    
    # Add statistical significance markers
    baseline_data = df_variant[df_variant['variant'] == baseline.lower()].iloc[0]  # Convert baseline to lowercase
    for idx, row in df_variant.iterrows():
        if row['variant'] == baseline.lower():  # Convert baseline to lowercase
            continue
            
        # Compare F1 scores
        base_f1s = [r["test_metrics"].get("eval_f1", 0) 
                   for r in variant_runs[baseline]]
        variant_f1s = [r["test_metrics"].get("eval_f1", 0) 
                      for r in variant_runs[row['variant']]]
        
        # Perform Wilcoxon signed-rank test
        if len(base_f1s) > 1 and len(variant_f1s) > 1:
            _, p_value = stats.wilcoxon(base_f1s, variant_f1s)
            df_variant.at[idx, 'f1_p_value'] = p_value
            df_variant.at[idx, 'significant'] = p_value < 0.05
    
    return df_variant, df_stage, df_inference

# ------------------------------------------------------------------
# ---------------------------- plots -------------------------------
# ------------------------------------------------------------------

def plot_stacked(df_stage: pd.DataFrame, out: Path):
    """Plot energy consumption for each stage of the pipeline in vertically stacked subplots"""
    if df_stage.empty:
        return
        
    # Aggregate across runs
    df_agg = df_stage.groupby(['variant', 'stage'])['kwh'].mean().reset_index()
    
    # Pivot and reorder columns
    pivot = df_agg.pivot(index="variant", columns="stage", values="kwh").fillna(0)
    pivot = pivot[[c for c in STAGE_ORDER if c in pivot.columns]]
    
    # Convert variant names to V1, V2, etc. and sort numerically
    pivot.index = [f'V{extract_variant_number(v)}' for v in pivot.index]
    pivot = pivot.sort_index(key=lambda x: [extract_variant_number(v) for v in x])
    
    # Create figure with subplots for each stage
    n_stages = len(pivot.columns)
    fig, axes = plt.subplots(n_stages, 1, figsize=(8, 2.5*n_stages), sharex=True)
    
    # Plot each stage in its own subplot
    for i, stage in enumerate(pivot.columns):
        ax = axes[i]
        pivot[[stage]].div(1000).plot(
            kind="bar",
            ax=ax,
            color='#2ecc71' if stage == 'train' else '#3498db'
        )
        ax.set_ylabel("kWh")
        # Add title inside the plot
        ax.text(0.02, 0.95, stage.title(), 
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                verticalalignment='top')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Remove legend for all subplots
        ax.get_legend().remove()
    
    # Add x-label to the bottom subplot
    axes[-1].set_xlabel("Variant", fontsize=9)
    
    # Adjust layout with minimal spacing
    plt.tight_layout(h_pad=0)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_energy_tradeoff(df: pd.DataFrame, baseline: str, out: Path):
    """Plot energy tradeoff with F1 score and Pareto frontier"""
    if df.empty:
        return
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Case-insensitive baseline mask (do not modify df in-place)
    baseline_mask = df['variant'].str.lower() == baseline.lower()
    
    print("Available variants:", df['variant'].tolist())
    print("Looking for baseline:", baseline)
    if not baseline_mask.any():
        print(f"Warning: Baseline {baseline} not found in data. Skipping plot.")
        return
    
    baseline_row = df[baseline_mask].iloc[0]
    plt.scatter(
        baseline_row['total_kwh'], 
        baseline_row['f1'],
        s=150, c='red', marker='*', label='Baseline',
        edgecolor='black', linewidth=1.5
    )
    
    # Pareto frontier
    pareto_df = df[df['on_pareto'] & ~baseline_mask]
    if not pareto_df.empty:
        plt.scatter(
            pareto_df['total_kwh'], 
            pareto_df['f1'],
            s=80, c='green', marker='D', label='Pareto Frontier',
            edgecolor='black', linewidth=1
        )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('total_kwh')
        plt.plot(
            pareto_sorted['total_kwh'], 
            pareto_sorted['f1'],
            'g--', alpha=0.5, linewidth=1.5
        )
    
    # Other variants
    other_df = df[~df['on_pareto'] & ~baseline_mask]
    if not other_df.empty:
        plt.scatter(
            other_df['total_kwh'], 
            other_df['f1'],
            s=60, c='blue', alpha=0.7, label='Other Variants',
            edgecolor='black', linewidth=1
        )
    
    # Add labels for all points using adjustText
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(
        baseline_row['total_kwh'],
        baseline_row['f1'],
        'V0',
        ha='center',
        va='center',
        fontsize=12,
        color='black',
        fontweight='bold',
        # bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
    ))
    
    # Add labels for all other points
    for _, row in df[~baseline_mask].iterrows():
        variant_num = extract_variant_number(row['variant'])
        texts.append(plt.text(
            row['total_kwh'],
            row['f1'],
            f'V{variant_num}',
            fontsize=10,
            fontweight='bold',
            # bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
        ))
    
    # Adjust text positions to avoid overlap with increased arrow length
    adjust_text(texts, 
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle='arc3,rad=0.5'),
               expand_text=(8.0, 8.0),
               expand_points=(8.0, 8.0),
               force_text=(6.0, 6.0),
               force_points=(6.0, 6.0),
               only_move={'points':'xy', 'text':'xy'},
               avoid_text=True,
               avoid_points=True,
               avoid_self=True,
               avoid_axes=True)
    
    # Formatting
    plt.xlabel("Total Energy (kWh)")
    plt.ylabel("F1 Score")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot Pareto frontier for energy vs evaluation time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Convert baseline to lowercase for consistent comparison
    baseline = baseline.lower()
    
    # Convert variant column to lowercase for consistent comparison
    df = df.copy()
    df['variant'] = df['variant'].str.lower()
    
    # Case-insensitive baseline mask
    baseline_mask = df['variant'] == baseline
    if not baseline_mask.any():
        print(f"Warning: Baseline {baseline} not found in data. Skipping plot.")
        return
    
    # Baseline point
    baseline_row = df[baseline_mask].iloc[0]
    plt.scatter(
        baseline_row['total_kwh'], 
        baseline_row['eval_time_s'],
        s=200, c='red', marker='*', label=get_variant_name(baseline)
    )
    
    # Identify Pareto frontier
    points = df[['total_kwh', 'eval_time_s']].values
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        if pareto_mask[i]:
            # Dominated if any point has lower energy AND lower time
            mask = (points[:,0] <= point[0]) & (points[:,1] <= point[1])
            mask[i] = False  # Don't compare to self
            if np.any(mask):
                pareto_mask[i] = False
    
    # Plot Pareto frontier
    pareto_df = df[pareto_mask & ~baseline_mask]
    if not pareto_df.empty:
        plt.scatter(
            pareto_df['total_kwh'], 
            pareto_df['eval_time_s'],
            s=100, c='green', marker='D', label='Pareto Frontier'
        )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('total_kwh')
        plt.plot(
            pareto_sorted['total_kwh'], 
            pareto_sorted['eval_time_s'],
            'g--', alpha=0.5
        )
    
    # Other variants
    other_df = df[~pareto_mask & ~baseline_mask]
    if not other_df.empty:
        plt.scatter(
            other_df['total_kwh'], 
            other_df['eval_time_s'],
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Annotate points
    for _, row in df.iterrows():
        if baseline_mask.loc[row.name]:
            continue
        plt.annotate(
            get_variant_name(row['variant']), 
            (row['total_kwh'], row['eval_time_s']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=9
        )
    
    # Formatting
    plt.xlabel("Total Energy Consumption (kWh)")
    plt.ylabel("Evaluation Time (seconds)")
    plt.title("Energy-Evaluation Time Trade-off")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_delta_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot delta energy vs delta evaluation time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Convert baseline to lowercase for consistent comparison
    baseline = baseline.lower()
    
    # Convert variant column to lowercase for consistent comparison
    df = df.copy()
    df['variant'] = df['variant'].str.lower()
    
    # Case-insensitive baseline mask
    baseline_mask = df['variant'] == baseline
    if not baseline_mask.any():
        print(f"Warning: Baseline {baseline} not found in data. Skipping plot.")
        return
    
    # Set equal aspect ratio and limits
    max_delta = max(
        abs(df['Δtotal_kwh'].min()),
        abs(df['Δtotal_kwh'].max()),
        abs(df['percent_diff_time'].min()),
        abs(df['percent_diff_time'].max())
    )
    limit = max_delta * 1.1  # Add 10% padding
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Plot origin point (baseline)
    plt.scatter(0, 0, s=200, c='red', marker='*', label='Baseline')
    
    # Identify Pareto frontier
    points = df[['Δtotal_kwh', 'percent_diff_time']].values
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        if pareto_mask[i]:
            # Dominated if any point has lower energy delta AND lower time delta
            mask = (points[:,0] <= point[0]) & (points[:,1] <= point[1])
            mask[i] = False  # Don't compare to self
            if np.any(mask):
                pareto_mask[i] = False
    
    # Plot Pareto frontier
    pareto_df = df[pareto_mask & ~baseline_mask]
    if not pareto_df.empty:
        plt.scatter(
            pareto_df['Δtotal_kwh'], 
            pareto_df['percent_diff_time'],
            s=100, c='green', marker='D', label='Pareto Frontier'
        )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('Δtotal_kwh')
        plt.plot(
            pareto_sorted['Δtotal_kwh'], 
            pareto_sorted['percent_diff_time'],
            'g--', alpha=0.5
        )
    
    # Other variants
    other_df = df[~pareto_mask & ~baseline_mask]
    if not other_df.empty:
        plt.scatter(
            other_df['Δtotal_kwh'], 
            other_df['percent_diff_time'],
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Annotate points
    for _, row in df.iterrows():
        if baseline_mask.loc[row.name]:
            continue
        plt.annotate(
            get_variant_name(row['variant']), 
            (row['Δtotal_kwh'], row['percent_diff_time']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=9
        )
    
    # Add quadrant lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add quadrant labels
    plt.text(limit*0.8, limit*0.8, 'Worse Energy\nWorse Time', ha='center', va='center')
    plt.text(-limit*0.8, limit*0.8, 'Better Energy\nWorse Time', ha='center', va='center')
    plt.text(limit*0.8, -limit*0.8, 'Worse Energy\nBetter Time', ha='center', va='center')
    plt.text(-limit*0.8, -limit*0.8, 'Better Energy\nBetter Time', ha='center', va='center')
    
    # Formatting
    plt.xlabel("Δ Energy Consumption (%)")
    plt.ylabel("Δ Evaluation Time (%)")
    plt.title("Energy-Time Trade-off Relative to Baseline")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_delta_training_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot delta training energy vs delta training time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Set equal aspect ratio and limits
    max_delta = max(
        abs(df['Δtotal_kwh'].min()),
        abs(df['Δtotal_kwh'].max()),
        abs(df['percent_diff_time'].min()),
        abs(df['percent_diff_time'].max())
    )
    limit = max_delta * 1.1  # Add 10% padding
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Add quadrant background colors
    ax.axhspan(0, limit, xmin=0.5, xmax=1, color='red', alpha=0.1)  # Top right
    ax.axhspan(-limit, 0, xmin=0, xmax=0.5, color='green', alpha=0.1)  # Bottom left
    
    # Add quadrant lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Baseline point (should be at origin since it's the reference)
    plt.scatter(0, 0, s=200, c='red', marker='*', zorder=5, edgecolor='black', linewidth=1.5)
    
    # Plot all variants except baseline
    other_df = df[df['variant'] != baseline]
    if not other_df.empty:
        # Group variants by quadrant
        better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)]
        worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0)]
        mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)) & 
                        ~((other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0))]
        
        # Plot each group with different colors
        for _, row in better_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_time'],
                s=100, c='green', marker='o', zorder=4, edgecolor='black', linewidth=1.5
            )
            
        for _, row in worse_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_time'],
                s=100, c='red', marker='o', zorder=4, edgecolor='black', linewidth=1.5
            )
            
        for _, row in mixed.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_time'],
                s=100, c='gray', marker='o', zorder=4, edgecolor='black', linewidth=1.5
            )
    
    # Add quadrant annotations with background
    annotations = [
        (0.95, 0.95, 'Worse Energy\nWorse Runtime', 'right', 'top'),
        (0.05, 0.95, 'Better Energy\nWorse Runtime', 'left', 'top'),
        (0.95, 0.05, 'Worse Energy\nBetter Runtime', 'right', 'bottom'),
        (0.05, 0.05, 'Better Energy\nBetter Runtime', 'left', 'bottom')
    ]
    
    for x, y, text, ha, va in annotations:
        plt.text(
            x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.8,
                pad=5
            ),
            fontsize=14,
            fontweight='bold'
        )
    
    # Add labels for all points
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(0, 0, 'V0', ha='center', va='center', fontsize=14, color='black', fontweight='bold'))
    
    # Add labels for all other points
    for _, row in df[df['variant'] != baseline].iterrows():
        variant_num = extract_variant_number(row['variant'])
        texts.append(plt.text(
            row['Δtotal_kwh'],
            row['percent_diff_time'],
            f'V{variant_num}',
            fontsize=12,
            fontweight='bold'
        ))
    
    # Adjust text positions to avoid overlaps
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.3'),
                expand_points=(4.0, 4.0),
                force_points=(0.8, 0.8),
                force_text=(1.5, 1.5),
                only_move={'points':'xy', 'text':'xy'},
                avoid_text=True,
                avoid_points=True,
                avoid_self=True)
    
    # Formatting
    plt.xlabel("Δ Training Energy (%)", fontsize=16)
    plt.ylabel("Δ Training Time (%)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_delta_eval_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot delta evaluation energy vs delta evaluation time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Set equal aspect ratio and limits
    max_delta = max(
        abs(df['Δtotal_kwh'].min()),
        abs(df['Δtotal_kwh'].max()),
        abs(df['percent_diff_eval_time'].min()),
        abs(df['percent_diff_eval_time'].max())
    )
    limit = max_delta * 1.1  # Add 10% padding
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Add quadrant background colors
    ax.axhspan(0, limit, xmin=0.5, xmax=1, color='red', alpha=0.1)  # Top right
    ax.axhspan(-limit, 0, xmin=0, xmax=0.5, color='green', alpha=0.1)  # Bottom left
    
    # Add quadrant lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Baseline point (should be at origin since it's the reference)
    plt.scatter(0, 0, s=200, c='red', marker='*', zorder=5, edgecolor='black', linewidth=1.5)
    
    # Plot all variants except baseline
    other_df = df[df['variant'] != baseline]
    if not other_df.empty:
        # Group variants by quadrant
        better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_eval_time'] < 0)]
        worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_eval_time'] > 0)]
        mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_eval_time'] < 0)) & 
                        ~((other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_eval_time'] > 0))]
        
        # Plot each group with different colors
        for _, row in better_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_eval_time'],
                s=100, c='green', marker='o', zorder=4, edgecolor='black', linewidth=1.5
            )
            
        for _, row in worse_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_eval_time'],
                s=100, c='red', marker='o', zorder=4, edgecolor='black', linewidth=1.5
            )
            
        for _, row in mixed.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_eval_time'],
                s=100, c='gray', marker='o', zorder=4, edgecolor='black', linewidth=1.5
            )
    
    # Add quadrant annotations with background
    annotations = [
        (0.95, 0.95, 'Worse Energy\nWorse Runtime', 'right', 'top'),
        (0.05, 0.95, 'Better Energy\nWorse Runtime', 'left', 'top'),
        (0.95, 0.05, 'Worse Energy\nBetter Runtime', 'right', 'bottom'),
        (0.05, 0.05, 'Better Energy\nBetter Runtime', 'left', 'bottom')
    ]
    
    for x, y, text, ha, va in annotations:
        plt.text(
            x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.8,
                pad=5
            ),
            fontsize=14,
            fontweight='bold'
        )
    
    # Add labels for all points
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(0, 0, 'V0', ha='center', va='center', fontsize=14, color='black', fontweight='bold'))
    
    # Add labels for all other points
    for _, row in df[df['variant'] != baseline].iterrows():
        variant_num = extract_variant_number(row['variant'])
        texts.append(plt.text(
            row['Δtotal_kwh'],
            row['percent_diff_eval_time'],
            f'V{variant_num}',
            fontsize=12,
            fontweight='bold'
        ))
    
    # Adjust text positions to avoid overlaps
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.3'),
                expand_points=(4.0, 4.0),
                force_points=(0.8, 0.8),
                force_text=(1.5, 1.5),
                only_move={'points':'xy', 'text':'xy'},
                avoid_text=True,
                avoid_points=True,
                avoid_self=True)
    
    # Formatting
    plt.xlabel("Δ Evaluation Energy (%)", fontsize=16)
    plt.ylabel("Δ Evaluation Time (%)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_total_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot total energy vs total time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Baseline point
    baseline_row = df[df['variant'] == baseline].iloc[0]
    plt.scatter(
        baseline_row['total_kwh'], 
        baseline_row['runtime_s'],
        s=200, c='red', marker='*', label=get_variant_name(baseline)
    )
    
    # Identify Pareto frontier
    points = df[['total_kwh', 'runtime_s']].values
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        if pareto_mask[i]:
            # Dominated if any point has lower energy AND lower time
            mask = (points[:,0] <= point[0]) & (points[:,1] <= point[1])
            mask[i] = False  # Don't compare to self
            if np.any(mask):
                pareto_mask[i] = False
    
    # Plot Pareto frontier
    pareto_df = df[pareto_mask & (df['variant'] != baseline)]
    if not pareto_df.empty:
        plt.scatter(
            pareto_df['total_kwh'], 
            pareto_df['runtime_s'],
            s=100, c='green', marker='D', label='Pareto Frontier'
        )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('total_kwh')
        plt.plot(
            pareto_sorted['total_kwh'], 
            pareto_sorted['runtime_s'],
            'g--', alpha=0.5
        )
    
    # Other variants
    other_df = df[~pareto_mask & (df['variant'] != baseline)]
    if not other_df.empty:
        plt.scatter(
            other_df['total_kwh'], 
            other_df['runtime_s'],
            s=100, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Add labels for all points
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(
        baseline_row['total_kwh'],
        baseline_row['runtime_s'],
        'V0',
        ha='center',
        va='center',
        fontsize=14,
        color='black',
        fontweight='bold'
    ))
    
    # Add labels for all other points
    for _, row in df[df['variant'] != baseline].iterrows():
        variant_num = extract_variant_number(row['variant'])
        texts.append(plt.text(
            row['total_kwh'],
            row['runtime_s'],
            f'V{variant_num}',
            fontsize=12,
            fontweight='bold'
        ))
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, 
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               expand_text=(1.2, 1.2),
               expand_points=(1.2, 1.2),
               force_text=(0.5, 0.5),
               force_points=(0.5, 0.5))
    
    # Formatting
    plt.xlabel("Total Energy (kWh)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_faceted_energy_tradeoff(df: pd.DataFrame, baseline: str, out: Path):
    """Create faceted energy trade-off plots separated by model size"""
    if df.empty:
        return
        
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Get unique model sizes
    model_sizes = sorted(df['model_size'].unique())
    n_sizes = len(model_sizes)
    
    # Calculate grid dimensions
    n_cols = min(3, n_sizes)  # Max 3 columns
    n_rows = (n_sizes + n_cols - 1) // n_cols
    
    # Create subplots
    for idx, size in enumerate(model_sizes, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        
        # Filter data for this model size
        size_df = df[df['model_size'] == size]
        
        # Set equal aspect ratio and limits
        max_delta = max(
            abs(size_df['Δtotal_kwh'].min()),
            abs(size_df['Δtotal_kwh'].max()),
            abs(size_df['percent_diff_time'].min()),
            abs(size_df['percent_diff_time'].max())
        )
        limit = max_delta * 1.1  # Add 10% padding
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Add quadrant background colors
        ax.axhspan(0, limit, xmin=0.5, xmax=1, color='red', alpha=0.1)
        ax.axhspan(-limit, 0, xmin=0, xmax=0.5, color='green', alpha=0.1)
        
        # Add quadrant lines
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Baseline point
        baseline_row = size_df[size_df['variant'] == baseline]
        if not baseline_row.empty:
            ax.scatter(0, 0, s=200, c='red', marker='*', zorder=5, edgecolor='black', linewidth=1.5)
        
        # Plot all variants except baseline
        other_df = size_df[size_df['variant'] != baseline]
        if not other_df.empty:
            # Group variants by quadrant
            better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)]
            worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0)]
            mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)) & 
                           ~((other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0))]
            
            # Plot each group
            for _, row in better_both.iterrows():
                ax.scatter(
                    row['Δtotal_kwh'], 
                    row['percent_diff_time'],
                    s=100, c='green', marker='o', zorder=4, edgecolor='black', linewidth=1.5
                )
            
            for _, row in worse_both.iterrows():
                ax.scatter(
                    row['Δtotal_kwh'], 
                    row['percent_diff_time'],
                    s=100, c='red', marker='o', zorder=4, edgecolor='black', linewidth=1.5
                )
            
            for _, row in mixed.iterrows():
                ax.scatter(
                    row['Δtotal_kwh'], 
                    row['percent_diff_time'],
                    s=100, c='gray', marker='o', zorder=4, edgecolor='black', linewidth=1.5
                )
        
        # Add quadrant annotations
        annotations = [
            (0.95, 0.95, 'Worse Energy\nWorse Runtime', 'right', 'top'),
            (0.05, 0.95, 'Better Energy\nWorse Runtime', 'left', 'top'),
            (0.95, 0.05, 'Worse Energy\nBetter Runtime', 'right', 'bottom'),
            (0.05, 0.05, 'Better Energy\nBetter Runtime', 'left', 'bottom')
        ]
        
        for x, y, text, ha, va in annotations:
            ax.text(
                x, y, text,
                transform=ax.transAxes,
                ha=ha, va=va,
                bbox=dict(
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.8,
                    pad=5
                ),
                fontsize=12,
                fontweight='bold'
            )
        
        # Add labels for all points
        from adjustText import adjust_text
        texts = []
        
        # Add baseline label
        texts.append(ax.text(0, 0, 'V0', ha='center', va='center', fontsize=12, color='black', fontweight='bold'))
        
        # Add labels for all other points
        for _, row in size_df[size_df['variant'] != baseline].iterrows():
            variant_num = extract_variant_number(row['variant'])
            texts.append(ax.text(
                row['Δtotal_kwh'],
                row['percent_diff_time'],
                f'V{variant_num}',
                fontsize=10,
                fontweight='bold'
            ))
        
        # Adjust text positions to avoid overlap
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle='arc3,rad=0.2'),
                   expand_text=(1.5, 1.5),
                   expand_points=(1.5, 1.5),
                   force_text=(1.0, 1.0),
                   force_points=(1.0, 1.0))
        
        # Add title and labels
        ax.set_title(f"Model Size: {size}", fontsize=14, pad=20)
        ax.set_xlabel("Δ Energy (kWh)", fontsize=12)
        ax.set_ylabel("Δ Runtime (%)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_energy_tradeoff_by_type(df: pd.DataFrame, baseline: str, out: Path):
    """Plot energy trade-off with variants grouped by optimization type"""
    if df.empty:
        return
        
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
    
    # Define optimization types and their markers
    opt_types = {
        'pruning': ('Layer Pruning', 's'),  # square
        'quantization': ('Quantization', '^'),  # triangle up
        'compilation': ('Torch Compile', 'D'),  # diamond
        'fp16': ('FP16', 'o'),  # circle
        'gradient': ('Gradient Opts', 'v'),  # triangle down
        'sequence': ('Sequence Length', '>'),  # triangle right
        'other': ('Other', 'x')  # x
    }
    
    # Set equal aspect ratio and limits
    max_delta = max(
        abs(df['Δtotal_kwh'].min()),
        abs(df['Δtotal_kwh'].max()),
        abs(df['percent_diff_time'].min()),
        abs(df['percent_diff_time'].max())
    )
    limit = max_delta * 1.1  # Add 10% padding
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Add quadrant background colors
    ax.axhspan(0, limit, xmin=0.5, xmax=1, color='red', alpha=0.1)
    ax.axhspan(-limit, 0, xmin=0, xmax=0.5, color='green', alpha=0.1)
    
    # Add quadrant lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Baseline point
    plt.scatter(0, 0, s=200, c='red', marker='*', zorder=5, edgecolor='black', linewidth=1.5, label='Baseline (V0)')
    
    # Plot all variants except baseline
    other_df = df[df['variant'] != baseline]
    if not other_df.empty:
        # Group variants by quadrant
        better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)]
        worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0)]
        mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)) & 
                        ~((other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0))]
        
        # Function to determine optimization type
        def get_opt_type(variant):
            variant = variant.lower()
            if 'pruning' in variant:
                return 'pruning'
            elif 'quantization' in variant or 'quant' in variant:
                return 'quantization'
            elif 'compile' in variant:
                return 'compilation'
            elif 'fp16' in variant:
                return 'fp16'
            elif 'gradient' in variant or 'checkpoint' in variant:
                return 'gradient'
            elif 'sequence' in variant or 'length' in variant:
                return 'sequence'
            else:
                return 'other'
        
        # Plot each group with different markers based on optimization type
        for _, row in better_both.iterrows():
            opt_type = get_opt_type(row['variant'])
            marker = opt_types[opt_type][1]
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_time'],
                s=80, c='green', marker=marker, zorder=4, edgecolor='black', linewidth=1.5,
                label=opt_types[opt_type][0] if opt_type not in [l.get_label() for l in plt.gca().lines] else ""
            )
            
        for _, row in worse_both.iterrows():
            opt_type = get_opt_type(row['variant'])
            marker = opt_types[opt_type][1]
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_time'],
                s=80, c='red', marker=marker, zorder=4, edgecolor='black', linewidth=1.5,
                label=opt_types[opt_type][0] if opt_type not in [l.get_label() for l in plt.gca().lines] else ""
            )
            
        for _, row in mixed.iterrows():
            opt_type = get_opt_type(row['variant'])
            marker = opt_types[opt_type][1]
            plt.scatter(
                row['Δtotal_kwh'], 
                row['percent_diff_time'],
                s=80, c='gray', marker=marker, zorder=4, edgecolor='black', linewidth=1.5,
                label=opt_types[opt_type][0] if opt_type not in [l.get_label() for l in plt.gca().lines] else ""
            )
    
    # Add quadrant annotations
    annotations = [
        (0.95, 0.95, 'Worse Energy\nWorse Runtime', 'right', 'top'),
        (0.05, 0.95, 'Better Energy\nWorse Runtime', 'left', 'top'),
        (0.95, 0.05, 'Worse Energy\nBetter Runtime', 'right', 'bottom'),
        (0.05, 0.05, 'Better Energy\nBetter Runtime', 'left', 'bottom')
    ]
    
    for x, y, text, ha, va in annotations:
        plt.text(
            x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.8,
                pad=5
            ),
            fontsize=12,
            fontweight='bold'
        )
    
    # Add labels for all points
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(0, 0, 'V0', ha='center', va='center', fontsize=12, color='black', fontweight='bold'))
    
    # Add labels for all other points
    for _, row in df[df['variant'] != baseline].iterrows():
        variant_num = extract_variant_number(row['variant'])
        texts.append(plt.text(
            row['Δtotal_kwh'],
            row['percent_diff_time'],
            f'V{variant_num}',
            fontsize=10,
            fontweight='bold'
        ))
    
    # Adjust text positions
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.3'),
                expand_points=(4.0, 4.0),
                force_points=(0.8, 0.8),
                force_text=(1.5, 1.5),
                only_move={'points':'xy', 'text':'xy'},
                avoid_text=True,
                avoid_points=True,
                avoid_self=True)
    
    # Formatting
    plt.xlabel("Δ Energy (%)", fontsize=16)
    plt.ylabel("Δ Time (%)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add legend with optimization types
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

# ------------------------------------------------------------------
# ------------------------ reporting -------------------------------
# ------------------------------------------------------------------

def generate_delta_table(df: pd.DataFrame, baseline: str, out: Path):
    """Generate Table 1: Individual optimization impacts"""
    if df.empty:
        print("Warning: No data available for delta table")
        return None
        
    # Filter to individual variants (not combined)
    # This assumes combined variants start with V and have multiple numbers
    df = df[df['variant'].apply(lambda x: len(x) < 4)]  
    
    if df.empty:
        print("Warning: No individual variants found after filtering")
        return None
    
    # Select and rename columns
    required_cols = ['variant', 'Δtotal_kwh', 'Δruntime_s', 'Δf1', 'significant']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return None
        
    table_df = df[required_cols].copy()
    
    # Replace variant names with readable names
    table_df['variant'] = table_df['variant'].apply(get_variant_name)
    
    # Format values
    table_df['Δtotal_kwh'] = table_df['Δtotal_kwh'].apply(lambda x: f"{x:.1f}%")
    table_df['Δruntime_s'] = table_df['Δruntime_s'].apply(lambda x: f"{x:.1f}%")
    table_df['Δf1'] = table_df['Δf1'].apply(lambda x: f"{x:.3f}")
    
    # Add significance markers
    table_df['Δf1'] = table_df.apply(
        lambda row: f"{row['Δf1']}*" if row['significant'] else row['Δf1'],
        axis=1
    )
    
    # Rename columns
    table_df.columns = ['Variant', 'Δ Energy', 'Δ Runtime', 'Δ F1', 'Significant']
    
    # Save to CSV
    table_df.to_csv(out / "individual_impacts.csv", index=False)
    
    # Print markdown table
    print("\n## Individual Optimization Impacts")
    print(table_df.to_markdown(index=False))
    
    # Find best performers
    try:
        best_energy = get_variant_name(df.loc[df['Δtotal_kwh'].idxmin(), 'variant'])
        best_time = get_variant_name(df.loc[df['Δruntime_s'].idxmin(), 'variant'])
        best_f1 = get_variant_name(df.loc[df['Δf1'].idxmax(), 'variant'])
        
        # Practitioner takeaways
        print("\n**Practitioner Takeaways:**")
        print(f"- For maximum energy savings: **{best_energy}**")
        print(f"- For fastest runtime: **{best_time}**")
        print(f"- For best model performance: **{best_f1}**")
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not determine best performers: {e}")
    
    return table_df

def generate_pareto_analysis(df: pd.DataFrame, out: Path):
    """Analyze and report on Pareto frontier"""
    if df.empty:
        return
        
    pareto_df = df[df['on_pareto']]
    
    # Save to CSV
    pareto_df.to_csv(out / "pareto_frontier.csv", index=False)
    
    # Print analysis
    print("\n## Pareto Frontier Analysis")
    print("Optimal trade-off points (energy vs performance):")
    print(pareto_df[['variant', 'total_kwh', 'f1']].to_markdown(index=False))
    
    # Recommendations
    print("\n**Recommendations based on use case:**")
    for _, row in pareto_df.sort_values('total_kwh').iterrows():
        if row['total_kwh'] < pareto_df['total_kwh'].median():
            perf = "energy-efficient"
        else:
            perf = "high-performance"
            
        print(f"- **{row['variant']}**: Best for {perf} scenarios "
              f"({row['total_kwh']:.1f} kWh, F1: {row['f1']:.3f})")

def generate_comprehensive_report(df_variant: pd.DataFrame, df_stage: pd.DataFrame, df_inference: pd.DataFrame, baseline: str, out: Path):
    """Generate a comprehensive report with detailed statistics and analysis"""
    with open(out / "report.txt", "w") as f:
        f.write("=== GREEN AI PIPELINE ANALYSIS REPORT ===\n\n")
        
        # 1. Overall Statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("===================\n")
        f.write(f"Total number of variants analyzed: {len(df_variant)}\n")
        f.write(f"Baseline variant: {baseline}\n\n")
        
        # 2. Individual Variant Analysis
        f.write("2. INDIVIDUAL VARIANT ANALYSIS\n")
        f.write("===========================\n")
        for _, row in df_variant.iterrows():
            f.write(f"\nVariant: {row['variant']}\n")
            f.write("-" * (len(row['variant']) + 9) + "\n")
            f.write(f"Total Energy: {row['total_kwh']:.6f} kWh (±{row['total_kwh_std']:.6f})\n")
            f.write(f"Runtime: {row['runtime_s']:.6f} s (±{row['runtime_s_std']:.6f})\n")
            f.write(f"F1 Score: {row['f1']:.6f} (±{row['f1_std']:.6f})\n")
            f.write(f"Accuracy: {row['accuracy']:.6f}\n")
            f.write(f"GPU Utilization: {row['avg_gpu_util']:.6f}%\n")
            f.write(f"GPU Memory Utilization: {row['avg_gpu_mem_util']:.6f}%\n")
            f.write(f"Peak Memory: {row['peak_mem_gb']:.6f} GB\n")
            
            # Percentage differences from baseline
            f.write("\nDifferences from baseline:\n")
            f.write(f"Energy: {row['percent_diff_energy']:.6f}%\n")
            f.write(f"Time: {row['percent_diff_time']:.6f}%\n")
            f.write(f"F1 Score: {row['percent_diff_f1']:.6f}%\n")
            
            # Statistical significance
            if 'significant' in row:
                f.write(f"Statistically significant: {'Yes' if row['significant'] else 'No'}\n")
            f.write("\n")
        
        # 3. Combined Variants Analysis
        f.write("3. COMBINED VARIANTS ANALYSIS\n")
        f.write("===========================\n")
        
        # Define the mapping of combined variants to their individual components
        combined_components = {
            "V13_gradient_accumulation_plus_fp16_plus_checkpointing": {
                "components": ["V1_gradient_checkpointing", "V7_f16", ],
                "description": "Combines gradient accumulation, FP16 precision, and gradient checkpointing"
            },
            "v26_pruning_plus_seq_lngth_torch_compile": {
                "components": ["v21_layer_pruning_12_bottom", "V8_sequence_length_trimming", "v11_torch_compile"],
                "description": "Combines bottom 12 layer pruning, sequence length trimming, and torch compile"
            },
            "v27_torch_compile_plus_fp16": {
                "components": ["v11_torch_compile", "V7_f16"],
                "description": "Combines torch compile and FP16 precision"
            },
            "v28_pruning_plus_torch_compile_fp16": {
                "components": ["v21_layer_pruning_12_bottom", "v11_torch_compile", "V7_f16"],
                "description": "Combines bottom 12 layer pruning, torch compile, and FP16 precision"
            },
            "v29_attention_plus_pin_memory_optimizer_gradient_accumulation": {
                "components": ["V12_attention", "V10_dataloader_pin_memory", "V6_optimizer"],
                "description": "Combines attention optimization, pin memory, and optimizer improvements"
            }
        }

        for combined_variant, info in combined_components.items():
            if combined_variant not in df_variant['variant'].values:
                continue

            f.write(f"\nCombined Variant: {combined_variant}\n")
            f.write("-" * (len(combined_variant) + 17) + "\n")
            f.write(f"Description: {info['description']}\n\n")
            
            # Get combined variant data
            combined_data = df_variant[df_variant['variant'] == combined_variant].iloc[0]
            
            f.write("Combined Variant Metrics:\n")
            f.write(f"Total Energy: {combined_data['total_kwh']:.6f} kWh\n")
            f.write(f"Runtime: {combined_data['runtime_s']:.6f} s\n")
            f.write(f"F1 Score: {combined_data['f1']:.6f}\n")
            f.write(f"Energy Savings vs Baseline: {combined_data['percent_diff_energy']:.6f}%\n")
            f.write(f"Time Savings vs Baseline: {combined_data['percent_diff_time']:.6f}%\n")
            f.write(f"F1 Impact vs Baseline: {combined_data['percent_diff_f1']:.6f}%\n\n")
            
            f.write("Comparison with Individual Components:\n")
            f.write("---------------------------------\n")
            
            # Compare with each individual component
            for component in info['components']:
                if component not in df_variant['variant'].values:
                    f.write(f"Component {component} not found in data\n")
                    continue
                    
                component_data = df_variant[df_variant['variant'] == component].iloc[0]
                
                # Calculate relative differences
                energy_diff = ((combined_data['total_kwh'] - component_data['total_kwh']) / component_data['total_kwh'] * 100) if component_data['total_kwh'] != 0 else 0
                time_diff = ((combined_data['runtime_s'] - component_data['runtime_s']) / component_data['runtime_s'] * 100) if component_data['runtime_s'] != 0 else 0
                f1_diff = combined_data['f1'] - component_data['f1']
                
                f.write(f"\nComponent: {component}\n")
                f.write(f"Energy Impact: {energy_diff:.6f}%\n")
                f.write(f"Time Impact: {time_diff:.6f}%\n")
                f.write(f"F1 Score Impact: {f1_diff:.6f}\n")
                
                # Add interpretation
                f.write("Interpretation:\n")
                if energy_diff < 0:
                    f.write(f"- Energy efficient compared to {component}\n")
                if time_diff < 0:
                    f.write(f"- Faster than {component}\n")
                if f1_diff > 0:
                    f.write(f"- Better F1 score than {component}\n")
                elif f1_diff < 0:
                    f.write(f"- Lower F1 score than {component}\n")
                else:
                    f.write(f"- Similar F1 score to {component}\n")
            
            # Add overall analysis
            f.write("\nOverall Analysis:\n")
            f.write("----------------\n")
            
            # Calculate average impact across components
            energy_impacts = []
            time_impacts = []
            f1_impacts = []
            
            for component in info['components']:
                if component in df_variant['variant'].values:
                    component_data = df_variant[df_variant['variant'] == component].iloc[0]
                    energy_impacts.append(((combined_data['total_kwh'] - component_data['total_kwh']) / component_data['total_kwh'] * 100) if component_data['total_kwh'] != 0 else 0)
                    time_impacts.append(((combined_data['runtime_s'] - component_data['runtime_s']) / component_data['runtime_s'] * 100) if component_data['runtime_s'] != 0 else 0)
                    f1_impacts.append(combined_data['f1'] - component_data['f1'])
            
            if energy_impacts:
                avg_energy_impact = sum(energy_impacts) / len(energy_impacts)
                avg_time_impact = sum(time_impacts) / len(time_impacts)
                avg_f1_impact = sum(f1_impacts) / len(f1_impacts)
                
                f.write(f"Average Energy Impact: {avg_energy_impact:.6f}%\n")
                f.write(f"Average Time Impact: {avg_time_impact:.6f}%\n")
                f.write(f"Average F1 Impact: {avg_f1_impact:.6f}\n")
                
                # Add recommendation
                f.write("\nRecommendation:\n")
                if avg_energy_impact < 0 and avg_time_impact < 0 and avg_f1_impact >= 0:
                    f.write("This combination is recommended as it improves both efficiency and performance\n")
                elif avg_energy_impact < 0 and avg_time_impact < 0:
                    f.write("This combination improves efficiency but may impact performance\n")
                elif avg_f1_impact > 0:
                    f.write("This combination improves performance but may impact efficiency\n")
                else:
                    f.write("Consider using individual components based on specific requirements\n")
            
            f.write("\n" + "="*50 + "\n")
        
        # 4. Stage-wise Analysis
        if not df_stage.empty:
            f.write("4. STAGE-WISE ANALYSIS\n")
            f.write("=====================\n")
            stage_stats = df_stage.groupby('stage').agg({
                'kwh': ['mean', 'std', 'min', 'max'],
                'duration': ['mean', 'std', 'min', 'max']
            }).round(6)
            
            for stage in STAGE_ORDER:
                if stage in stage_stats.index:
                    stats = stage_stats.loc[stage]
                    f.write(f"\nStage: {stage}\n")
                    f.write("-" * (len(stage) + 7) + "\n")
                    f.write(f"Average Energy: {stats[('kwh', 'mean')]:.6f} kWh\n")
                    f.write(f"Std Dev: {stats[('kwh', 'std')]:.6f} kWh\n")
                    f.write(f"Min Energy: {stats[('kwh', 'min')]:.6f} kWh\n")
                    f.write(f"Max Energy: {stats[('kwh', 'max')]:.6f} kWh\n")
                    f.write(f"Average Duration: {stats[('duration', 'mean')]:.6f} s\n")
                    f.write(f"Duration Std Dev: {stats[('duration', 'std')]:.6f} s\n")
                    f.write(f"Min Duration: {stats[('duration', 'min')]:.6f} s\n")
                    f.write(f"Max Duration: {stats[('duration', 'max')]:.6f} s\n")
            
            # Stage-wise percentage of total energy
            total_energy = df_stage['kwh'].sum()
            f.write("\nStage-wise Energy Distribution:\n")
            for stage in STAGE_ORDER:
                if stage in stage_stats.index:
                    stage_energy = stage_stats.loc[stage, ('kwh', 'mean')]
                    percentage = (stage_energy / total_energy) * 100
                    f.write(f"{stage}: {percentage:.6f}%\n")
            
            # Stage-wise percentage of total time
            total_time = df_stage['duration'].sum()
            f.write("\nStage-wise Time Distribution:\n")
            for stage in STAGE_ORDER:
                if stage in stage_stats.index:
                    stage_time = stage_stats.loc[stage, ('duration', 'mean')]
                    percentage = (stage_time / total_time) * 100
                    f.write(f"{stage}: {percentage:.6f}%\n")
        
        # 5. Statistical Analysis
        f.write("\n5. STATISTICAL ANALYSIS\n")
        f.write("=====================\n")
        
        # Energy statistics
        f.write("\nEnergy Consumption Statistics:\n")
        f.write(f"Mean: {df_variant['total_kwh'].mean():.6f} kWh\n")
        f.write(f"Median: {df_variant['total_kwh'].median():.6f} kWh\n")
        f.write(f"Std Dev: {df_variant['total_kwh'].std():.6f} kWh\n")
        f.write(f"Min: {df_variant['total_kwh'].min():.6f} kWh\n")
        f.write(f"Max: {df_variant['total_kwh'].max():.6f} kWh\n")
        
        # Runtime statistics
        f.write("\nRuntime Statistics:\n")
        f.write(f"Mean: {df_variant['runtime_s'].mean():.6f} s\n")
        f.write(f"Median: {df_variant['runtime_s'].median():.6f} s\n")
        f.write(f"Std Dev: {df_variant['runtime_s'].std():.6f} s\n")
        f.write(f"Min: {df_variant['runtime_s'].min():.6f} s\n")
        f.write(f"Max: {df_variant['runtime_s'].max():.6f} s\n")
        
        # Evaluation time statistics
        if 'eval_time_s' in df_variant.columns:
            f.write("\nEvaluation Time Statistics:\n")
            f.write(f"Mean: {df_variant['eval_time_s'].mean():.6f} s\n")
            f.write(f"Median: {df_variant['eval_time_s'].median():.6f} s\n")
            f.write(f"Std Dev: {df_variant['eval_time_s'].std():.6f} s\n")
            f.write(f"Min: {df_variant['eval_time_s'].min():.6f} s\n")
            f.write(f"Max: {df_variant['eval_time_s'].max():.6f} s\n")
        
        # F1 score statistics
        f.write("\nF1 Score Statistics:\n")
        f.write(f"Mean: {df_variant['f1'].mean():.6f}\n")
        f.write(f"Median: {df_variant['f1'].median():.6f}\n")
        f.write(f"Std Dev: {df_variant['f1'].std():.6f}\n")
        f.write(f"Min: {df_variant['f1'].min():.6f}\n")
        f.write(f"Max: {df_variant['f1'].max():.6f}\n")
        
        # 6. Best Performers
        f.write("\n6. BEST PERFORMERS\n")
        f.write("================\n")
        
        # Best energy efficiency
        best_energy = df_variant.loc[df_variant['percent_diff_energy'].idxmin()]
        f.write(f"\nMost Energy Efficient: {best_energy['variant']}\n")
        f.write(f"Energy Savings: {best_energy['percent_diff_energy']:.6f}%\n")
        f.write(f"F1 Impact: {best_energy['percent_diff_f1']:.6f}%\n")
        
        # Best runtime
        best_time = df_variant.loc[df_variant['percent_diff_time'].idxmin()]
        f.write(f"\nFastest Runtime: {best_time['variant']}\n")
        f.write(f"Time Savings: {best_time['percent_diff_time']:.6f}%\n")
        f.write(f"F1 Impact: {best_time['percent_diff_f1']:.6f}%\n")
        
        # Best evaluation time
        if 'eval_time_s' in df_variant.columns:
            best_eval_time = df_variant.loc[df_variant['percent_diff_eval_time'].idxmin()]
            f.write(f"\nFastest Evaluation: {best_eval_time['variant']}\n")
            f.write(f"Evaluation Time Savings: {best_eval_time['percent_diff_eval_time']:.6f}%\n")
            f.write(f"F1 Impact: {best_eval_time['percent_diff_f1']:.6f}%\n")
        
        # Best F1 score
        best_f1 = df_variant.loc[df_variant['f1'].idxmax()]
        f.write(f"\nBest F1 Score: {best_f1['variant']}\n")
        f.write(f"F1 Score: {best_f1['f1']:.6f}\n")
        f.write(f"Energy Impact: {best_f1['percent_diff_energy']:.6f}%\n")
        f.write(f"Time Impact: {best_f1['percent_diff_time']:.6f}%\n")
        
        # 7. Recommendations
        f.write("\n7. RECOMMENDATIONS\n")
        f.write("================\n")
        
        # Energy-focused recommendations
        f.write("\nFor Energy Efficiency:\n")
        energy_efficient = df_variant[df_variant['percent_diff_energy'] < 0].sort_values('percent_diff_energy')
        for _, row in energy_efficient.head(3).iterrows():
            f.write(f"- {row['variant']}: {row['percent_diff_energy']:.6f}% energy savings, F1 impact: {row['percent_diff_f1']:.6f}%\n")
        
        # Performance-focused recommendations
        f.write("\nFor Performance:\n")
        high_performance = df_variant[df_variant['f1'] > df_variant['f1'].median()].sort_values('f1', ascending=False)
        for _, row in high_performance.head(3).iterrows():
            f.write(f"- {row['variant']}: F1 score {row['f1']:.6f}, Energy impact: {row['percent_diff_energy']:.6f}%\n")
        
        # Balanced recommendations
        f.write("\nFor Balanced Approach:\n")
        balanced = df_variant[df_variant['on_pareto']].sort_values('total_kwh')
        for _, row in balanced.iterrows():
            f.write(f"- {row['variant']}: Energy {row['total_kwh']:.6f} kWh, F1: {row['f1']:.6f}\n")

def generate_experiment_setup(df_variant: pd.DataFrame, df_stage: pd.DataFrame, df_inference: pd.DataFrame, baseline: str, out: Path):
    """Generate experiment setup information including hardware and software configuration"""
    with open(out / "experiment_setup.txt", "w") as f:
        f.write("=== EXPERIMENT SETUP AND CONFIGURATION ===\n\n")
        
        # 1. Hardware Configuration
        f.write("1. HARDWARE CONFIGURATION\n")
        f.write("=======================\n")
        
        # GPU Information
        f.write("\nGPU Information:\n")
        f.write("----------------\n")
        # Get GPU info from the first run that has it
        gpu_info = next((r for r in df_variant.itertuples() if hasattr(r, 'gpu_info')), None)
        if gpu_info:
            f.write(f"GPU Model: {gpu_info.gpu_info.get('model', 'N/A')}\n")
            f.write(f"GPU Memory: {gpu_info.gpu_info.get('memory', 'N/A')} GB\n")
            f.write(f"GPU Driver: {gpu_info.gpu_info.get('driver', 'N/A')}\n")
            f.write(f"CUDA Version: {gpu_info.gpu_info.get('cuda', 'N/A')}\n")
        
        # GPU Utilization Statistics
        f.write("\nGPU Utilization Statistics:\n")
        f.write(f"Average GPU Utilization: {df_variant['avg_gpu_util'].mean():.6f}%\n")
        f.write(f"Average GPU Memory Utilization: {df_variant['avg_gpu_mem_util'].mean():.6f}%\n")
        f.write(f"Peak GPU Memory Usage: {df_variant['peak_mem_gb'].max():.6f} GB\n")
        
        # CPU Information
        f.write("\nCPU Information:\n")
        f.write("----------------\n")
        # Get CPU info from the first run that has it
        cpu_info = next((r for r in df_variant.itertuples() if hasattr(r, 'cpu_info')), None)
        if cpu_info:
            f.write(f"CPU Model: {cpu_info.cpu_info.get('model', 'N/A')}\n")
            f.write(f"CPU Cores: {cpu_info.cpu_info.get('cores', 'N/A')}\n")
            f.write(f"CPU Memory: {cpu_info.cpu_info.get('memory', 'N/A')} GB\n")
        
        # 2. Software Configuration
        f.write("\n2. SOFTWARE CONFIGURATION\n")
        f.write("=======================\n")
        
        # Python and Framework Versions
        f.write("\nPython and Framework Versions:\n")
        f.write("----------------------------\n")
        # Get version info from the first run that has it
        version_info = next((r for r in df_variant.itertuples() if hasattr(r, 'version_info')), None)
        if version_info:
            f.write(f"Python Version: {version_info.version_info.get('python', 'N/A')}\n")
            f.write(f"PyTorch Version: {version_info.version_info.get('pytorch', 'N/A')}\n")
            f.write(f"CUDA Version: {version_info.version_info.get('cuda', 'N/A')}\n")
            f.write(f"Other Dependencies: {version_info.version_info.get('dependencies', 'N/A')}\n")
        
        # 3. Model Configuration
        f.write("\n3. MODEL CONFIGURATION\n")
        f.write("====================\n")
        f.write(f"Model Architecture: ModernBERT-base\n")
        f.write(f"Model Type: Transformer-based\n")
        f.write(f"Task: Vulnerability Detection\n")
        f.write(f"Precision: Mixed (FP16/FP32)\n")
        
        # 4. Dataset Configuration
        f.write("\n4. DATASET CONFIGURATION\n")
        f.write("======================\n")
        f.write(f"Dataset Name: BigVul\n")
        f.write(f"Task Type: Vulnerability Detection\n")
        f.write(f"Dataset Type: Code Analysis\n")
        f.write(f"Language: Source Code\n")
        
        # 5. Training Configuration
        f.write("\n5. TRAINING CONFIGURATION\n")
        f.write("======================\n")
        # Get training info from the first run that has it
        training_info = next((r for r in df_variant.itertuples() if hasattr(r, 'training_info')), None)
        if training_info:
            f.write(f"Batch Size: {training_info.training_info.get('batch_size', 'N/A')}\n")
            f.write(f"Learning Rate: {training_info.training_info.get('learning_rate', 'N/A')}\n")
            f.write(f"Optimizer: {training_info.training_info.get('optimizer', 'N/A')}\n")
            f.write(f"Number of Epochs: {training_info.training_info.get('epochs', 'N/A')}\n")
            f.write(f"Training Time: {training_info.training_info.get('training_time', 'N/A')}\n")
        
        # 6. Measurement Configuration
        f.write("\n6. MEASUREMENT CONFIGURATION\n")
        f.write("=========================\n")
        
        # Energy Measurement
        f.write("\nEnergy Measurement:\n")
        f.write("------------------\n")
        f.write(f"Energy Measurement Tool: CodeCarbon\n")
        f.write(f"Measurement Frequency: Per Training Run\n")
        
        # Performance Metrics
        f.write("\nPerformance Metrics:\n")
        f.write("------------------\n")
        f.write(f"Evaluation Metrics: F1 Score, Accuracy\n")
        f.write(f"Hardware Metrics: GPU Utilization, Memory Usage\n")
        f.write(f"Energy Metrics: Total Energy, CPU Energy, GPU Energy\n")
        
        # 7. Variant Information
        f.write("\n7. VARIANT INFORMATION\n")
        f.write("=====================\n")
        f.write(f"Total Number of Variants: {len(df_variant)}\n")
        f.write(f"Baseline Variant: {baseline}\n")
        
        # Split variants into individual and combined
        individual_variants = df_variant[~df_variant['variant'].str.contains('plus')]['variant'].tolist()
        combined_variants = df_variant[df_variant['variant'].str.contains('plus')]['variant'].tolist()
        
        f.write("\nVariant Types:\n")
        f.write("-------------\n")
        f.write("Individual Optimizations:\n")
        for variant in sorted(individual_variants):
            f.write(f"- {variant}\n")
        
        f.write("\nCombined Optimizations:\n")
        for variant in sorted(combined_variants):
            f.write(f"- {variant}\n")

def combine_variant_metrics(results_root: Path, baseline: str, out_dir: Path):
    """Combine metrics for each variant across stages using raw results data."""
    # Use the same aggregation logic as the main analysis
    df_variant, df_stage, df_inference = aggregate(results_root, baseline)
    
    # Set variant as index for easier lookup (case-insensitive)
    df_variant = df_variant.set_index('variant')
    df_variant.index = df_variant.index.str.lower()
    
    # Convert baseline to lowercase for consistent comparison
    baseline = baseline.lower()
    
    # Get all variant directories and sort them
    variant_dirs = [d for d in results_root.iterdir() if d.is_dir() and not d.name.startswith('__')]
    all_variants = [d.name.lower() for d in variant_dirs]  # Convert to lowercase
    
    # Prepare combined metrics
    combined = []
    for variant in all_variants:
        # Get variant name and number
        variant_name = get_variant_name(variant)
        variant_number = extract_variant_number(variant)
        
        # Initialize default metrics
        default_metrics = {
            'variant': variant,
            'variant_name': variant_name,
            'variant_number': variant_number,
            'description': '',
            'total_kwh': 0.0,
            'runtime_s': 0.0,
            'eval_time_s': 0.0,
            'f1': 0.0,
            'Δf1': 0.0,
            'percent_diff_energy': 0.0,
            'stages': []
        }
        
        # If variant has results, update metrics
        if variant in df_variant.index:
            variant_data = df_variant.loc[variant].to_dict()
            default_metrics.update({
                'description': variant_data.get('description', ''),
                'total_kwh': float(variant_data.get('total_kwh', 0.0)),
                'runtime_s': float(variant_data.get('runtime_s', 0.0)),
                'eval_time_s': float(variant_data.get('eval_time_s', 0.0)),
                'f1': float(variant_data.get('f1', 0.0)),
                'Δf1': float(variant_data.get('Δf1', 0.0)),
                'percent_diff_energy': float(variant_data.get('percent_diff_energy', 0.0))
            })
            
            # Get stage data for this variant (case-insensitive)
            stages = df_stage[df_stage['variant'].str.lower() == variant]
            stage_metrics = []
            total_duration = 0.0
            
            # Process each stage
            for stage in ['load_dataset', 'tokenize_dataset', 'load_model', 'train_model', 'save_model', 'evaluate_model']:
                stage_data = stages[stages['stage'] == stage]
                if not stage_data.empty:
                    duration = float(stage_data['duration'].iloc[0])
                    total_duration += duration
                    stage_metrics.append({
                        'stage': stage,
                        'kwh': float(stage_data['kwh'].iloc[0]),
                        'duration': duration
                    })
                else:
                    stage_metrics.append({
                        'stage': stage,
                        'kwh': 0.0,
                        'duration': 0.0
                    })
            
            default_metrics['stages'] = stage_metrics
            default_metrics['runtime_s'] = total_duration
            
            # Add inference metrics if available (case-insensitive)
            inference_data = df_inference[df_inference['variant'].str.lower() == variant]
            if not inference_data.empty:
                inference_dict = inference_data.to_dict(orient='records')[0]
                # Convert numeric values to float
                for k, v in inference_dict.items():
                    if isinstance(v, (int, float)):
                        inference_dict[k] = float(v)
                default_metrics['inference'] = inference_dict
        else:
            # Add empty stages for variants without results
            default_metrics['stages'] = [
                {'stage': stage, 'kwh': 0.0, 'duration': 0.0}
                for stage in ['load_dataset', 'tokenize_dataset', 'load_model', 'train_model', 'save_model', 'evaluate_model']
            ]
        
        combined.append(default_metrics)
    
    # Sort by variant number
    combined.sort(key=lambda x: x['variant_number'])
    
    # Write to JSON
    with open(out_dir / 'combined_metrics.json', 'w') as f_json:
        json.dump(combined, f_json, indent=2)
    
    # Create flattened CSV
    flat_rows = []
    for entry in combined:
        flat = {k: v for k, v in entry.items() if k not in ['stages', 'inference', 'variant_number']}
        for stage in entry['stages']:
            flat[f"{stage['stage']}_kwh"] = stage['kwh']
            flat[f"{stage['stage']}_duration"] = stage['duration']
        if 'inference' in entry:
            for k, v in entry['inference'].items():
                flat[f"inference_{k}"] = v
        flat_rows.append(flat)
    
    df_flat = pd.DataFrame(flat_rows)
    df_flat.to_csv(out_dir / 'combined_metrics.csv', index=False)
    print("Combined metrics written to combined_metrics.json and combined_metrics.csv")
    
    # Return DataFrames with lowercase variant names for consistent comparison
    df_variant.index = df_variant.index.str.lower()
    df_stage['variant'] = df_stage['variant'].str.lower()
    df_inference['variant'] = df_inference['variant'].str.lower()
    
    return df_variant, df_stage, df_inference

def get_stage_indicators(variant):
    """Return appropriate cell commands for each development phase"""
    # Convert variant to lowercase for consistent comparison
    variant = variant.lower()
    
    data_cell = "\\emptycell"
    model_cell = "\\emptycell"
    train_cell = "\\emptycell"
    system_cell = "\\emptycell"
    infer_cell = "\\emptycell"
    
    # Data stage
    if variant in ['v4_tokenizer', 'v8_sequence_length_trimming', 'v10_dataloader_pin_memory', 
                   'v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation',
                   'v26_pruning_plus_seq_lngth_plus_torch_compile']:
        data_cell = "\\datacell"
    
    # Model stage
    if variant in ['v2_lora_peft', 'v3_quantization', 'v7_f16', 'v12_attention',
                   'v13_layer_pruning_4_top', 'v14_layer_pruning_4_bottom',
                   'v15_layer_pruning_8_top', 'v16_layer_pruning_8_bottom',
                   'v17_layer_pruning_12_top', 'v18_layer_pruning_12_bottom',
                   'v19_layer_pruning_16_top', 'v20_layer_pruning_16_bottom',
                   'v21_layer_pruning_20_top', 'v22_layer_pruning_20_bottom',
                   'v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation',
                   'v24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16',
                   'v25_gradient_accumulation_plus_fp16_plus_checkpointing',
                   'v26_pruning_plus_seq_lngth_plus_torch_compile',
                   'v27_torch_compile_plus_fp16',
                   'v28_pruning_plus_torch_compile_plus_fp16']:
        model_cell = "\\modelcell"
    
    # Training stage
    if variant in ['v1_gradient_checkpointing', 'v2_lora_peft', 'v3_quantization',
                   'v6_optimizer', 'v7_f16', 'v8_sequence_length_trimming',
                   'v10_dataloader_pin_memory', 'v11_torch_compile', 'v12_attention',
                   'v13_layer_pruning_4_top', 'v14_layer_pruning_4_bottom',
                   'v15_layer_pruning_8_top', 'v16_layer_pruning_8_bottom',
                   'v17_layer_pruning_12_top', 'v18_layer_pruning_12_bottom',
                   'v19_layer_pruning_16_top', 'v20_layer_pruning_16_bottom',
                   'v21_layer_pruning_20_top', 'v22_layer_pruning_20_bottom',
                   'v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation',
                   'v24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16',
                   'v25_gradient_accumulation_plus_fp16_plus_checkpointing',
                   'v26_pruning_plus_seq_lngth_plus_torch_compile',
                   'v27_torch_compile_plus_fp16',
                   'v28_pruning_plus_torch_compile_plus_fp16']:
        train_cell = "\\traincell"
    
    # System stage
    if variant in ['v3_quantization', 'v4_tokenizer', 'v5_power_limit_100w',
                   'v7_f16', 'v8_sequence_length_trimming', 'v9_inference_engine',
                   'v11_torch_compile', 'v13_layer_pruning_4_top',
                   'v14_layer_pruning_4_bottom', 'v15_layer_pruning_8_top',
                   'v16_layer_pruning_8_bottom', 'v17_layer_pruning_12_top',
                   'v18_layer_pruning_12_bottom', 'v19_layer_pruning_16_top',
                   'v20_layer_pruning_16_bottom', 'v21_layer_pruning_20_top',
                   'v22_layer_pruning_20_bottom',
                   'v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation',
                   'v24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16',
                   'v25_gradient_accumulation_plus_fp16_plus_checkpointing',
                   'v26_pruning_plus_seq_lngth_plus_torch_compile',
                   'v27_torch_compile_plus_fp16',
                   'v28_pruning_plus_torch_compile_plus_fp16']:
        system_cell = "\\systemcell"
    
    # Inference stage
    if variant in ['v2_lora_peft', 'v3_quantization', 'v4_tokenizer',
                   'v9_inference_engine', 'v11_torch_compile', 'v12_attention',
                   'v13_layer_pruning_4_top', 'v14_layer_pruning_4_bottom',
                   'v15_layer_pruning_8_top', 'v16_layer_pruning_8_bottom',
                   'v17_layer_pruning_12_top', 'v18_layer_pruning_12_bottom',
                   'v19_layer_pruning_16_top', 'v20_layer_pruning_16_bottom',
                   'v21_layer_pruning_20_top', 'v22_layer_pruning_20_bottom',
                   'v23_attention_plus_pin_memory_plus_optimizer_plus_gradient_accumulation',
                   'v24_inference_engine_plus_grad_cpting_plus_lora_plus_fp16',
                   'v25_gradient_accumulation_plus_fp16_plus_checkpointing',
                   'v26_pruning_plus_seq_lngth_plus_torch_compile',
                   'v27_torch_compile_plus_fp16',
                   'v28_pruning_plus_torch_compile_plus_fp16']:
        infer_cell = "\\infercell"
    
    return data_cell, model_cell, train_cell, system_cell, infer_cell

def plot_3d_metrics(df: pd.DataFrame, baseline: str, out: Path):
    """Create a 3D plot showing the relationship between time, energy, and F1 score"""
    if df.empty:
        return
        
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define optimization types and their markers
    opt_types = {
        'pruning': ('Layer Pruning', 's'),  # square
        'quantization': ('Quantization', '^'),  # triangle up
        'compilation': ('Torch Compile', 'D'),  # diamond
        'fp16': ('FP16', 'o'),  # circle
        'gradient': ('Gradient Opts', 'v'),  # triangle down
        'sequence': ('Sequence Length', '>'),  # triangle right
        'other': ('Other', 'x')  # x
    }
    
    # Function to determine optimization type
    def get_opt_type(variant):
        variant = variant.lower()
        if 'pruning' in variant:
            return 'pruning'
        elif 'quantization' in variant or 'quant' in variant:
            return 'quantization'
        elif 'compile' in variant:
            return 'compilation'
        elif 'fp16' in variant:
            return 'fp16'
        elif 'gradient' in variant or 'checkpoint' in variant:
            return 'gradient'
        elif 'sequence' in variant or 'length' in variant:
            return 'sequence'
        else:
            return 'other'
    
    # Plot baseline
    baseline_row = df[df['variant'] == baseline].iloc[0]
    ax.scatter(
        baseline_row['runtime_s'],
        baseline_row['total_kwh'],
        baseline_row['f1'],
        s=200, c='red', marker='*', label='Baseline (V0)',
        edgecolor='black', linewidth=1.5
    )
    
    # Plot all variants except baseline
    other_df = df[df['variant'] != baseline]
    if not other_df.empty:
        # Group variants by quadrant (based on energy and time)
        better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)]
        worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0)]
        mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['percent_diff_time'] < 0)) & 
                        ~((other_df['Δtotal_kwh'] > 0) & (other_df['percent_diff_time'] > 0))]
        
        # Plot each group with different markers based on optimization type
        for _, row in better_both.iterrows():
            opt_type = get_opt_type(row['variant'])
            marker = opt_types[opt_type][1]
            ax.scatter(
                row['runtime_s'],
                row['total_kwh'],
                row['f1'],
                s=100, c='green', marker=marker, label=opt_types[opt_type][0] if opt_type not in [l.get_label() for l in ax.lines] else "",
                edgecolor='black', linewidth=1.5
            )
            
        for _, row in worse_both.iterrows():
            opt_type = get_opt_type(row['variant'])
            marker = opt_types[opt_type][1]
            ax.scatter(
                row['runtime_s'],
                row['total_kwh'],
                row['f1'],
                s=100, c='red', marker=marker, label=opt_types[opt_type][0] if opt_type not in [l.get_label() for l in ax.lines] else "",
                edgecolor='black', linewidth=1.5
            )
            
        for _, row in mixed.iterrows():
            opt_type = get_opt_type(row['variant'])
            marker = opt_types[opt_type][1]
            ax.scatter(
                row['runtime_s'],
                row['total_kwh'],
                row['f1'],                s=100, c='gray', marker=marker, label=opt_types[opt_type][0] if opt_type not in [l.get_label() for l in ax.lines] else "",
                edgecolor='black', linewidth=1.5
            )
    
    # Add labels for all points
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(ax.text(
        baseline_row['runtime_s'],
        baseline_row['total_kwh'],
        baseline_row['f1'],
        'V0',
        ha='center',
        va='center',
        fontsize=12,
        color='black',
        fontweight='bold'
    ))
    
    # Add labels for all other points
    for _, row in df[df['variant'] != baseline].iterrows():
        variant_num = extract_variant_number(row['variant'])
        texts.append(ax.text(
            row['runtime_s'],
            row['total_kwh'],
            row['f1'],
            f'V{variant_num}',
            fontsize=10,
            fontweight='bold'
        ))
    
    # Formatting
    ax.set_xlabel("Runtime (seconds)", fontsize=14, labelpad=10)
    ax.set_ylabel("Energy (kWh)", fontsize=14, labelpad=10)
    ax.set_zlabel("F1 Score", fontsize=14, labelpad=10)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='center left', bbox_to_anchor=(1.1, 0.5),
             fontsize=12, framealpha=0.9)
    
    # Adjust the viewing angle for better perspective
    ax.view_init(elev=25, azim=125)  # Changed from (20, 45) to (30, 135)
    
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_energy_f1(df_variant: pd.DataFrame, df_stage: pd.DataFrame, baseline: str, out: Path):
    """Plot training energy vs F1 score"""
    if df_variant.empty or df_stage.empty:
        return
        
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Get training energy from stage data and convert index to lowercase
    train_energy = df_stage[df_stage['stage'] == 'train'].groupby('variant')['kwh'].mean()
    train_energy.index = train_energy.index.str.lower()
    
    # Case-insensitive baseline mask
    baseline_mask = df_variant['variant'].str.lower() == baseline.lower()
    if not baseline_mask.any() or baseline.lower() not in train_energy.index:
        print(f"Warning: Baseline {baseline} not found in data. Skipping plot.")
        return
    
    # Baseline point
    baseline_row = df_variant[baseline_mask].iloc[0]
    plt.scatter(
        train_energy[baseline.lower()], 
        baseline_row['f1'],
        s=200, c='red', marker='*', label='Baseline'
    )
    
    # Other variants
    other_df = df_variant[~baseline_mask]
    other_variants = [v.lower() for v in other_df['variant'].unique() if v.lower() in train_energy.index]
    
    if other_variants:
        plt.scatter(
            train_energy[other_variants], 
            other_df[other_df['variant'].str.lower().isin(other_variants)].groupby('variant')['f1'].mean(),
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Add labels for all points using adjustText
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(
        train_energy[baseline.lower()],
        baseline_row['f1'],
        'V0',
        ha='center',
        va='center',
        fontsize=12,
        color='black',
        fontweight='bold'
    ))
    
    # Add labels for all other points
    for variant in other_variants:
        variant_num = extract_variant_number(variant)
        texts.append(plt.text(
            train_energy[variant],
            other_df[other_df['variant'].str.lower() == variant]['f1'].mean(),
            f'V{variant_num}',
            fontsize=10,
            fontweight='bold'
        ))
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, 
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               expand_text=(1.2, 1.2),
               expand_points=(1.2, 1.2),
               force_text=(0.5, 0.5),
               force_points=(0.5, 0.5))
    
    # Formatting
    plt.xlabel("Training Energy (kWh)")
    plt.ylabel("F1 Score")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()

def plot_eval_energy_f1(df_variant: pd.DataFrame, df_stage: pd.DataFrame, baseline: str, out: Path):
    """Plot evaluation energy vs F1 score"""
    if df_variant.empty or df_stage.empty:
        return
        
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Get evaluation energy from stage data and convert index to lowercase
    eval_energy = df_stage[df_stage['stage'] == 'eval'].groupby('variant')['kwh'].mean()
    eval_energy.index = eval_energy.index.str.lower()
    
    # Case-insensitive baseline mask
    baseline_mask = df_variant['variant'].str.lower() == baseline.lower()
    if not baseline_mask.any() or baseline.lower() not in eval_energy.index:
        print(f"Warning: Baseline {baseline} not found in data. Skipping plot.")
        return
    
    # Baseline point
    baseline_row = df_variant[baseline_mask].iloc[0]
    plt.scatter(
        eval_energy[baseline.lower()], 
        baseline_row['f1'],
        s=200, c='red', marker='*', label='Baseline'
    )
    
    # Other variants
    other_df = df_variant[~baseline_mask]
    other_variants = [v.lower() for v in other_df['variant'].unique() if v.lower() in eval_energy.index]
    
    if other_variants:
        plt.scatter(
            eval_energy[other_variants], 
            other_df[other_df['variant'].str.lower().isin(other_variants)].groupby('variant')['f1'].mean(),
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Add labels for all points using adjustText
    from adjustText import adjust_text
    texts = []
    
    # Add baseline label
    texts.append(plt.text(
        eval_energy[baseline.lower()],
        baseline_row['f1'],
        'V0',
        ha='center',
        va='center',
        fontsize=12,
        color='black',
        fontweight='bold'
    ))
    
    # Add labels for all other points
    for variant in other_variants:
        variant_num = extract_variant_number(variant)
        texts.append(plt.text(
            eval_energy[variant],
            other_df[other_df['variant'].str.lower() == variant]['f1'].mean(),
            f'V{variant_num}',
            fontsize=10,
            fontweight='bold'
        ))
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, 
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               expand_text=(1.2, 1.2),
               expand_points=(1.2, 1.2),
               force_text=(0.5, 0.5),
               force_points=(0.5, 0.5))
    
    # Formatting
    plt.xlabel("Evaluation Energy (kWh)")
    plt.ylabel("F1 Score")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()


# ---------------------------------------------------------------
#  Normalized stage-wise energy plot
# ---------------------------------------------------------------
def plot_stage_energy_normalized(
    df_stage: pd.DataFrame,
    out: Path,
    *,
    cmap: str = "viridis"          # perceptually-uniform colormap
):
    """
    Creates a stacked bar chart showing energy consumption across stages for each variant.
    Each bar represents a variant's total energy consumption, divided into colored segments
    representing the proportion of energy from each pipeline stage.
    """
    if df_stage.empty:
        return

    # ---------- 1. stage-energy pivot (mean across runs) ----------
    stage_mean = (
        df_stage
        .groupby(["variant", "stage"])["kwh"]
        .mean()
        .unstack("stage")
        .reindex(columns=[s for s in STAGE_ORDER if s in df_stage["stage"].unique()])
        .fillna(0.0)
    )

    # match naming convention: variant index ➝ V0 … V28
    stage_mean.index = [f"V{extract_variant_number(v)}" for v in stage_mean.index]
    stage_mean = stage_mean.sort_index(key=lambda x: [int(v[1:]) for v in x])

    # ---------- 2. draw ----------
    # Create figure with appropriate size
    n_variants = len(stage_mean.index)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Create stacked bars
    bottom = np.zeros(n_variants)
    for stage in stage_mean.columns:
        values = stage_mean[stage].values
        ax.bar(stage_mean.index, values, bottom=bottom, 
               label=stage.replace("_", " ").title(),
               alpha=0.8)
        bottom += values

    # Customize the plot
    ax.set_xlabel("Variant", fontsize=11)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=11)
    ax.set_title("Stage-wise Energy Consumption by Variant", fontsize=12, pad=8)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels on top of each bar
    for i, variant in enumerate(stage_mean.index):
        total = stage_mean.loc[variant].sum()
        ax.text(i, total, f'{total:.1f}', 
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------
#  Stage-wise energy line plots
# ---------------------------------------------------------------
def plot_stage_energy_lines(
    df_stage: pd.DataFrame,
    df_variant: pd.DataFrame,
    out: Path,
    *,
    cmap: str = "viridis"          # perceptually-uniform colormap
):
    """
    Creates line plots showing energy consumption for each stage across variants,
    with F1 scores on a secondary y-axis.
    This helps visualize trends in stages with smaller energy consumption
    and their relationship with model performance.
    """
    if df_stage.empty or df_variant.empty:
        return

    # ---------- 1. stage-energy pivot (mean across runs) ----------
    stage_mean = (
        df_stage
        .groupby(["variant", "stage"])["kwh"]
        .mean()
        .unstack("stage")
        .reindex(columns=[s for s in STAGE_ORDER if s in df_stage["stage"].unique()])
        .fillna(0.0)
    )

    # match naming convention: variant index ➝ V0 … V28
    stage_mean.index = [f"V{extract_variant_number(v)}" for v in stage_mean.index]
    stage_mean = stage_mean.sort_index(key=lambda x: [int(v[1:]) for v in x])

    # ---------- 2. Get F1 scores ----------
    f1_scores = (
        df_variant
        .set_index("variant")["f1"]
        .rename(lambda v: f"V{extract_variant_number(v.lower())}")
        .reindex(stage_mean.index)    # ensure same ordering
    )

    # ---------- 3. draw ----------
    # Create figure with appropriate size
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # Create line plots for each stage on primary y-axis
    for stage in stage_mean.columns:
        values = stage_mean[stage].values
        ax1.plot(stage_mean.index, values, 
                marker='o', 
                label=stage.replace("_", " ").title(),
                alpha=0.8,
                linewidth=2)

    # Customize primary y-axis
    ax1.set_xlabel("Variant", fontsize=16)
    ax1.set_ylabel("Energy Consumption (kWh) - Log Scale", fontsize=16)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.3)
    # Increase tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=10)

    # Create secondary y-axis for F1 scores
    ax2 = ax1.twinx()
    ax2.plot(f1_scores.index, f1_scores.values,
            marker='s',  # square markers to distinguish from energy lines
            color='black',
            label='F1 Score',
            linewidth=2,
            linestyle='--')  # dashed line to distinguish from energy lines
    
    # Customize secondary y-axis
    ax2.set_ylabel("F1 Score", fontsize=16)
    # Increase tick label sizes for secondary axis
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='minor', labelsize=10)
    # Set y-axis limits to show variation better
    min_f1 = f1_scores.min()
    max_f1 = f1_scores.max()
    margin = (max_f1 - min_f1) * 0.1  # 10% margin
    ax2.set_ylim(min_f1 - margin, max_f1 + margin)  # Dynamic range based on actual scores

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              bbox_to_anchor=(1.1, 1), loc='upper left',
              borderaxespad=0., fontsize=11)

    # Rotate x-axis labels for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    
    # Set title
    # plt.title("Stage-wise Energy Consumption and F1 Score Trends", fontsize=12, pad=8)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
# ------------------------------ main ------------------------------
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results', type=str, default='variants', help='Path to results directory')
    parser.add_argument('--baseline', type=str, default='v0_baseline', help='Baseline variant name')
    parser.add_argument('--out-dir', type=str, default='analysis_results', help='Output directory for analysis')
    args = parser.parse_args()
    
    results_root = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert baseline to lowercase for consistent comparison
    baseline = args.baseline.lower()
    print(f"Starting analysis with baseline: {baseline}")
    
    # Aggregate results
    df_variant, df_stage, df_inference = aggregate(results_root, baseline)
    
    # Generate reports
    generate_comprehensive_report(df_variant, df_stage, df_inference, baseline, out_dir)
    generate_experiment_setup(df_variant, df_stage, df_inference, baseline, out_dir)
    
    # Save CSVs
    df_variant.to_csv(out_dir / "by_variant.csv", index=False)
    if not df_stage.empty:
        df_stage.to_csv(out_dir / "by_variant_stage.csv", index=False)
    if not df_inference.empty:
        df_inference.to_csv(out_dir / "inference_metrics.csv", index=False)
    
    # Combine metrics using raw results data
    combine_variant_metrics(Path(args.results), baseline, out_dir)
    
    # Create plots
    if not df_stage.empty:
        plot_stacked(df_stage, out_dir / "stage_energy.png")
        plot_stage_energy_normalized(df_stage, out_dir / "stage_energy_normalized.png")
        plot_stage_energy_lines(df_stage, df_variant, out_dir / "stage_energy_lines.png")
    if not df_variant.empty:
        plot_energy_tradeoff(df_variant, baseline, out_dir / "energy_tradeoff.png")
        plot_training_energy_f1(df_variant, df_stage, baseline, out_dir / "training_energy_f1.png")
        plot_eval_energy_f1(df_variant, df_stage, baseline, out_dir / "eval_energy_f1.png")
        plot_energy_time_pareto(df_variant, baseline, out_dir / "energy_time_pareto.png")
        plot_delta_energy_time_pareto(df_variant, baseline, out_dir / "delta_energy_time_pareto.png")
        plot_delta_training_energy_time_pareto(df_variant, baseline, out_dir / "delta_training_energy_time_pareto.png")
        plot_delta_eval_energy_time_pareto(df_variant, baseline, out_dir / "delta_eval_energy_time_pareto.png")
        plot_total_energy_time_pareto(df_variant, baseline, out_dir / "total_energy_time_pareto.png")
    
    # Replace the faceted plot with the new grouped plot
    plot_energy_tradeoff_by_type(
        df=df_variant,
        baseline=baseline,
        out=out_dir / "energy_tradeoff_by_type.png"
    )
    
    # Add the 3D plot
    plot_3d_metrics(
        df=df_variant,
        baseline=baseline,
        out=out_dir / "3d_metrics.png"
    )

    print("\nAnalysis complete. Results saved to:", out_dir)


if __name__ == "__main__":
    main()
    