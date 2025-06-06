#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D

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
    "V0_baseline": "Baseline",
    "V1_gradient_checkpointing": "Grad Check",
    "V2_lora_peft": "LoRA",
    "V3_quantization": "Quant",
    "V4_tokenizer": "Tokenizer",
    "V5_power_limit_100W": "100W Limit",
    "V6_optimizer": "Optimizer",
    "V7_f16": "FP16",
    "V8_sequence_length_trimming": "Seq Trim",
    "V9_inference_engine": "Inf Engine",
    "V10_dataloader_pin_memory": "Pin Memory",
    "v11_torch_compile": "Compile",
    "V12_attention": "Attention",
    "V13_gradient_accumulation_plus_fp16_plus_checkpointing": "Grad+FP16+Check",
    "v14_layer_pruning": "Layer Prune",
    "V15_inference_engine_plus_grad_cpting_plus_lora_plus_fp16": "Inf+Grad+LoRA+FP16",
    "v16_layer_pruning_4_top": "4 Top",
    "v17_layer_pruning_4_bottom": "4 Bottom",
    "v18_layer_pruning_8_top": "8 Top",
    "v19_layer_pruning_8_bottom": "8 Bottom",
    "v20_layer_pruning_12_top": "12 Top",
    "v21_layer_pruning_12_bottom": "12 Bottom",
    "v22_layer_pruning_16_top": "16 Top",
    "v23_layer_pruning_16_bottom": "16 Bottom",
    "v24_layer_pruning_20_top": "20 Top",
    "v25_layer_pruning_20_bottom": "20 Bottom",
}

def get_variant_name(variant: str) -> str:
    """Get readable name for a variant, falling back to original name if not found"""
    return VARIANT_NAMES.get(variant, variant)

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
        
    # Get baseline data
    baseline_mask = df['variant'] == baseline
    if not baseline_mask.any():
        print(f"Warning: Baseline variant {baseline} not found in data")
        return df
        
    baseline_data = df[baseline_mask].iloc[0]
    deltas = df.copy()
    
    # Calculate percentage changes
    for col in ['total_kwh', 'runtime_s', 'train_energy', 'inference_energy', 'eval_time_s']:
        if col in df.columns:
            deltas[f'Δ{col}'] = ((deltas[col] - baseline_data[col]) / baseline_data[col] * 100) if baseline_data[col] != 0 else 0
            
    # Calculate absolute changes for metrics
    for col in ['f1', 'accuracy']:
        if col in df.columns:
            deltas[f'Δ{col}'] = deltas[col] - baseline_data[col]
            
    # Calculate percentage differences for key metrics
    if 'total_kwh' in df.columns:
        deltas['percent_diff_energy'] = ((deltas['total_kwh'] - baseline_data['total_kwh']) / baseline_data['total_kwh'] * 100) if baseline_data['total_kwh'] != 0 else 0
    
    if 'runtime_s' in df.columns:
        deltas['percent_diff_time'] = ((deltas['runtime_s'] - baseline_data['runtime_s']) / baseline_data['runtime_s'] * 100) if baseline_data['runtime_s'] != 0 else 0
    
    if 'eval_time_s' in df.columns:
        deltas['percent_diff_eval_time'] = ((deltas['eval_time_s'] - baseline_data['eval_time_s']) / baseline_data['eval_time_s'] * 100) if baseline_data['eval_time_s'] != 0 else 0
    
    if 'f1' in df.columns:
        deltas['percent_diff_f1'] = ((deltas['f1'] - baseline_data['f1']) / baseline_data['f1'] * 100) if baseline_data['f1'] != 0 else 0
            
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
        for run_dir in run_dirs:
            yield variant_dir.name, run_dir


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

def aggregate(results_root: Path, baseline: str = "V0_baseline"):
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
            "emissions_path": emissions_path,
            "run_dir": run_dir,
            "variant": variant
        }
        variant_runs[variant].append(data)

    print("\nCollected data for variants:", list(variant_runs.keys()))
    
    # --- per‑variant summaries ------------------------------------
    by_variant_rows = []
    by_stage_rows = []
    by_inference_rows = []

    for variant, runs in variant_runs.items():
        print(f"\nAggregating data for {variant} ({len(runs)} runs)")
        
        # Training metrics
        train_energies = [r["train_energy"].get("energy_consumed", 0) for r in runs]
        train_times = [r["train_energy"].get("duration", 0) for r in runs]
        cpu_energies = [r["train_energy"].get("cpu_energy", 0) for r in runs]
        gpu_energies = [r["train_energy"].get("gpu_energy", 0) for r in runs]
        ram_energies = [r["train_energy"].get("ram_energy", 0) for r in runs]
        peak_mems = [r["train_energy"].get("max_gpu_mem", 0) for r in runs]
        
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
        
        # Calculate means and stdevs
        row = {
            "variant": variant,
            "total_kwh": _safe_mean(train_energies),
            "total_kwh_std": _safe_stdev(train_energies),
            "runtime_s": _safe_mean(train_times),
            "runtime_s_std": _safe_stdev(train_times),
            "cpu_kwh": _safe_mean(cpu_energies),
            "gpu_kwh": _safe_mean(gpu_energies),
            "ram_kwh": _safe_mean(ram_energies),
            "f1": _safe_mean(f1_scores),
            "f1_std": _safe_stdev(f1_scores),
            "accuracy": _safe_mean(accuracies),
            "peak_mem_gb": _safe_mean(peak_mems),
            "inference_energy": _safe_mean(inf_energies),
            "inference_energy_std": _safe_stdev(inf_energies),
            "inference_time": _safe_mean(inf_times),
            "throughput_qps": _safe_mean(throughputs),
            "latency_ms": _safe_mean(latencies),
            "avg_gpu_util": _safe_mean(gpu_utils),
            "avg_gpu_mem_util": _safe_mean(mem_utils),
            "num_runs": len(runs)
        }
        
        # Calculate energy per 1k inferences
        if row["throughput_qps"] and row["inference_energy"]:
            row["energy_per_1k_inf"] = (row["inference_energy"] / row["throughput_qps"]) * 1000
        
        # Add statistical significance markers later
        row["significant"] = False
        
        by_variant_rows.append(row)
        print(f"Added summary for {variant}")

        # Stage-wise energy data
        for run in runs:
            if run["emissions_path"] and run["emissions_path"].exists():
                df_stages = pd.read_csv(run["emissions_path"])
                for _, stage_row in df_stages.iterrows():
                    by_stage_rows.append({
                        "variant": variant,
                        "run_id": run["train_energy"].get("run_id", "unknown"),
                        "stage": stage_row['task_name'],
                        "kwh": stage_row['energy_consumed'],
                        "duration": stage_row['duration']
                    })

        # Inference metrics per run
        for run in runs:
            if "inference_metrics" in run:
                by_inference_rows.append({
                    "variant": variant,
                    "run_id": run["train_energy"].get("run_id", "unknown"),
                    **run["inference_metrics"]
                })

    if not by_variant_rows:
        print("WARNING: No variant data collected!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print("\nCreating DataFrames...")
    df_variant = pd.DataFrame(by_variant_rows)
    df_variant = df_variant.sort_values("variant")
    
    # Add evaluation time from stage data
    if not by_stage_rows:
        print("WARNING: No stage data collected!")
    else:
        df_stage = pd.DataFrame(by_stage_rows)
        eval_times = df_stage[df_stage['stage'] == 'evaluate_model'].groupby('variant')['duration'].agg(['mean', 'std']).reset_index()
        eval_times.columns = ['variant', 'eval_time_s', 'eval_time_std']
        df_variant = df_variant.merge(eval_times, on='variant', how='left')
    
    # Calculate deltas relative to baseline
    df_variant = calculate_deltas(df_variant, baseline)
    
    # Identify Pareto frontier for energy/performance trade-off
    df_pareto = identify_pareto_frontier(
        df_variant, 
        'Δtotal_kwh', 
        'f1'
    )
    df_variant['on_pareto'] = df_variant.index.isin(df_pareto.index)
    
    # Add statistical significance markers
    baseline_data = df_variant[df_variant['variant'] == baseline].iloc[0]
    for idx, row in df_variant.iterrows():
        if row['variant'] == baseline:
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

    df_inference = pd.DataFrame(by_inference_rows) if by_inference_rows else pd.DataFrame()
    
    return df_variant, df_stage, df_inference

# ------------------------------------------------------------------
# ---------------------------- plots -------------------------------
# ------------------------------------------------------------------

def plot_stacked(df_stage: pd.DataFrame, out: Path):
    if df_stage.empty:
        return
    # Aggregate across runs
    df_agg = df_stage.groupby(['variant', 'stage'])['kwh'].mean().reset_index()
    
    # Pivot and reorder columns
    pivot = df_agg.pivot(index="variant", columns="stage", values="kwh").fillna(0)
    pivot = pivot[[c for c in STAGE_ORDER if c in pivot.columns]]
    
    # Plot
    pivot.div(1000).plot(kind="bar", stacked=True, figsize=(11,6))
    plt.ylabel("Energy (kWh)")
    plt.title("Stage-wise Energy Consumption per Variant")
    plt.legend(title='Pipeline Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_energy_tradeoff(df_variant: pd.DataFrame, baseline: str, out: Path):
    """Plot F1 vs Energy with Pareto frontier"""
    if df_variant.empty:
        return
        
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Baseline point
    baseline_row = df_variant[df_variant['variant'] == baseline].iloc[0]
    plt.scatter(
        baseline_row['total_kwh'], 
        baseline_row['f1'],
        s=200, c='red', marker='*', label='Baseline'
    )
    
    # Pareto frontier
    pareto_df = df_variant[df_variant['on_pareto'] & (df_variant['variant'] != baseline)]
    if not pareto_df.empty:
        plt.scatter(
            pareto_df['total_kwh'], 
            pareto_df['f1'],
            s=100, c='green', marker='D', label='Pareto Frontier'
        )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('total_kwh')
        plt.plot(
            pareto_sorted['total_kwh'], 
            pareto_sorted['f1'],
            'g--', alpha=0.5
        )
    
    # Other variants
    other_df = df_variant[~df_variant['on_pareto'] & (df_variant['variant'] != baseline)]
    if not other_df.empty:
        plt.scatter(
            other_df['total_kwh'], 
            other_df['f1'],
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Annotate points
    for _, row in df_variant.iterrows():
        if row['variant'] == baseline:
            continue
        plt.annotate(
            row['variant'], 
            (row['total_kwh'], row['f1']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=9
        )
    
    # Formatting
    plt.xlabel("Total Energy Consumption (kWh)")
    plt.ylabel("F1 Score")
    plt.title("Energy-Performance Trade-off")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot Pareto frontier for energy vs evaluation time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Baseline point
    baseline_row = df[df['variant'] == baseline].iloc[0]
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
    pareto_df = df[pareto_mask & (df['variant'] != baseline)]
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
    other_df = df[~pareto_mask & (df['variant'] != baseline)]
    if not other_df.empty:
        plt.scatter(
            other_df['total_kwh'], 
            other_df['eval_time_s'],
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Annotate points
    for _, row in df.iterrows():
        if row['variant'] == baseline:
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
    """Plot Pareto frontier for delta energy vs delta evaluation time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Set equal aspect ratio and limits
    max_delta = max(
        abs(df['Δtotal_kwh'].min()),
        abs(df['Δtotal_kwh'].max()),
        abs(df['Δeval_time_s'].min()),
        abs(df['Δeval_time_s'].max())
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
    plt.scatter(0, 0, s=200, c='red', marker='*', label=get_variant_name(baseline), zorder=5)
    
    # Identify Pareto frontier
    points = df[['Δtotal_kwh', 'Δeval_time_s']].values
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        if pareto_mask[i]:
            # Dominated if any point has lower delta energy AND lower delta time
            # (more negative deltas are better)
            mask = (points[:,0] <= point[0]) & (points[:,1] <= point[1])
            mask[i] = False  # Don't compare to self
            if np.any(mask):
                pareto_mask[i] = False
    
    # Plot Pareto frontier
    pareto_df = df[pareto_mask & (df['variant'] != baseline)]
    if not pareto_df.empty:
        for _, row in pareto_df.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δeval_time_s'],
                s=100, c='green', marker='D', label=get_variant_name(row['variant']), zorder=4
            )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('Δtotal_kwh')
        plt.plot(
            pareto_sorted['Δtotal_kwh'], 
            pareto_sorted['Δeval_time_s'],
            'g--', alpha=0.5, zorder=3
        )
    
    # Other variants with color coding based on quadrant
    other_df = df[~pareto_mask & (df['variant'] != baseline)]
    if not other_df.empty:
        # Group variants by quadrant
        better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['Δeval_time_s'] < 0)]
        worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['Δeval_time_s'] > 0)]
        mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['Δeval_time_s'] < 0)) & 
                        ~((other_df['Δtotal_kwh'] > 0) & (other_df['Δeval_time_s'] > 0))]
        
        # Plot each group with different markers
        for _, row in better_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δeval_time_s'],
                s=80, c='green', marker='o', label=get_variant_name(row['variant']), zorder=4
            )
            
        for _, row in worse_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δeval_time_s'],
                s=80, c='red', marker='s', label=get_variant_name(row['variant']), zorder=4
            )
            
        for _, row in mixed.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δeval_time_s'],
                s=80, c='gray', marker='^', label=get_variant_name(row['variant']), zorder=4
            )
    
    # Add quadrant annotations with background
    annotations = [
        (0.95, 0.95, 'Worse Energy\nWorse Eval Time', 'right', 'top'),
        (0.05, 0.95, 'Better Energy\nWorse Eval Time', 'left', 'top'),
        (0.95, 0.05, 'Worse Energy\nBetter Eval Time', 'right', 'bottom'),
        (0.05, 0.05, 'Better Energy\nBetter Eval Time', 'left', 'bottom')
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
            fontsize=10
        )
    
    # Add legend in top left quadrant
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(0.05, 0.95),
        framealpha=0.9,
        edgecolor='gray',
        fontsize=9
    )
    
    # Formatting
    plt.xlabel("Δ Energy Consumption (%)", fontsize=12)
    plt.ylabel("Δ Evaluation Time (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
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
    table_df.columns = ['Variant', 'Δ Energy', 'Δ Time', 'Δ F1', 'Significant']
    
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
        print(f"- For fastest training: **{best_time}**")
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

# ------------------------------------------------------------------
# ------------------------------ main ------------------------------
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Green AI Pipeline Analyzer")
    parser.add_argument("--results_dir", type=Path, default=Path("variants"))
    parser.add_argument("--out_dir",     type=Path, default=Path("analysis_results"))
    parser.add_argument("--baseline",    type=str,  default="V0_baseline")
    parser.add_argument("--cascade_variants", nargs='+', default=["V0_baseline", "V4_tokenizer", "V2_lora_peft", "V15_inference_engine_plus_grad_cpting_plus_lora_plus_fp16"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting analysis with baseline: {args.baseline}")

    df_variant, df_stage, df_inference = aggregate(args.results_dir, args.baseline)
    
    # Save CSVs
    df_variant.to_csv(args.out_dir / "by_variant.csv", index=False)
    if not df_stage.empty:
        df_stage.to_csv(args.out_dir / "by_variant_stage.csv", index=False)
    if not df_inference.empty:
        df_inference.to_csv(args.out_dir / "inference_metrics.csv", index=False)
    
    # Generate reports
    generate_delta_table(df_variant, args.baseline, args.out_dir)
    generate_pareto_analysis(df_variant, args.out_dir)
    generate_comprehensive_report(df_variant, df_stage, df_inference, args.baseline, args.out_dir)
    generate_experiment_setup(df_variant, df_stage, df_inference, args.baseline, args.out_dir)
    
    # Create plots
    if not df_stage.empty:
        plot_stacked(df_stage, args.out_dir / "stage_energy.png")
    if not df_variant.empty:
        plot_energy_tradeoff(df_variant, args.baseline, args.out_dir / "energy_tradeoff.png")
        plot_energy_time_pareto(df_variant, args.baseline, args.out_dir / "energy_time_pareto.png")
        plot_delta_energy_time_pareto(df_variant, args.baseline, args.out_dir / "delta_energy_time_pareto.png")

    print("\nAnalysis complete. Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()