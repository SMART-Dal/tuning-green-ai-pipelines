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
    "V1_gradient_checkpointing": "Gradient Checkpointing",
    "V2_lora_peft": "LoRA + PEFT",
    "V3_quantization": "Quantization",
    "V4_tokenizer": "Tokenizer Optimization",
    "V5_power_limit_100W": "Power Limit (100W)",
    "V6_optimizer": "Optimizer Tuning",
    "V7_f16": "FP16 Precision",
    "V8_sequence_length_trimming": "Sequence Length Trimming",
    "V9_inference_engine": "Inference Engine",
    "V10_dataloader_pin_memory": "Dataloader Pin Memory",
    "v11_torch_compile": "Torch Compile",
    "V12_attention": "Attention Optimization",
    "V13_gradient_accumulation_plus_fp16_plus_checkpointing": "Grad Accum + FP16 + Checkpointing",
    "v14_layer_pruning": "Layer Pruning",
    "V15_inference_engine_plus_grad_cpting_plus_lora_plus_fp16": "Inference Engine + Grad Checkpointing + LoRA + FP16",
    "v16_layer_pruning_4_top": "Layer Pruning (4 Top)",
    "v17_layer_pruning_4_bottom": "Layer Pruning (4 Bottom)",
    "v18_layer_pruning_8_top": "Layer Pruning (8 Top)",
    "v19_layer_pruning_8_bottom": "Layer Pruning (8 Bottom)",
    "v20_layer_pruning_12_top": "Layer Pruning (12 Top)",
    "v21_layer_pruning_12_bottom": "Layer Pruning (12 Bottom)",
    "v22_layer_pruning_16_top": "Layer Pruning (16 Top)",
    "v23_layer_pruning_16_bottom": "Layer Pruning (16 Bottom)",
    "v24_layer_pruning_20_top": "Layer Pruning (20 Top)",
    "v25_layer_pruning_20_bottom": "Layer Pruning (20 Bottom)",
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
    for col in ['total_kwh', 'runtime_s', 'train_energy', 'inference_energy']:
        if col in df.columns:
            deltas[f'Δ{col}'] = ((deltas[col] - baseline_data[col]) / baseline_data[col] * 100) if baseline_data[col] != 0 else 0
            
    # Calculate absolute changes for metrics
    for col in ['f1', 'accuracy']:
        if col in df.columns:
            deltas[f'Δ{col}'] = deltas[col] - baseline_data[col]
            
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

    df_stage = pd.DataFrame(by_stage_rows) if by_stage_rows else pd.DataFrame()
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

def plot_cascade_effect(df_variant: pd.DataFrame, variants: List[str], out: Path):
    """Plot cumulative effect of optimizations"""
    if not variants or df_variant.empty:
        return
        
    # Select and order variants
    plot_df = df_variant[df_variant['variant'].isin(variants)]
    plot_df = plot_df.sort_values('total_kwh', ascending=False)
    
    # Calculate cumulative savings
    baseline = plot_df.iloc[0]['total_kwh']
    plot_df['cumulative_savings'] = baseline - plot_df['total_kwh']
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar plot for energy consumption
    ax1.bar(
        plot_df['variant'], 
        plot_df['total_kwh'],
        color='skyblue', 
        label='Energy Consumption'
    )
    ax1.set_ylabel('Total Energy (kWh)')
    ax1.set_xlabel('Variant')
    ax1.tick_params(axis='y')
    
    # Line plot for cumulative savings
    ax2 = ax1.twinx()
    ax2.plot(
        plot_df['variant'], 
        plot_df['cumulative_savings'],
        'r-o', 
        linewidth=2,
        label='Cumulative Savings'
    )
    ax2.set_ylabel('Cumulative Energy Savings (kWh)')
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.label.set_color('red')
    
    # Formatting
    plt.title('Cascade Effect of Optimizations')
    fig.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_inference_tradeoffs(df: pd.DataFrame, out: Path):
    """Plot 3D tradeoff between Energy, Latency, and Accuracy"""
    if df.empty or len(df) < 3:
        return
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize values for better visualization
    df['energy_norm'] = (df['inference_energy'] - df['inference_energy'].min()) / \
                        (df['inference_energy'].max() - df['inference_energy'].min())
    df['latency_norm'] = (df['latency_ms'] - df['latency_ms'].min()) / \
                        (df['latency_ms'].max() - df['latency_ms'].min())
    df['f1_norm'] = (df['f1'] - df['f1'].min()) / \
                   (df['f1'].max() - df['f1'].min())
    
    # Create scatter plot
    sc = ax.scatter(
        df['energy_norm'], 
        df['latency_norm'], 
        df['f1_norm'],
        c=df['f1'], 
        cmap='viridis',
        s=100,
        alpha=0.8
    )
    
    # Annotate points
    for _, row in df.iterrows():
        ax.text(
            row['energy_norm'], 
            row['latency_norm'], 
            row['f1_norm'],
            row['variant'],
            fontsize=9
        )
    
    # Labels
    ax.set_xlabel('Energy (Normalized)')
    ax.set_ylabel('Latency (Normalized)')
    ax.set_zlabel('F1 Score (Normalized)')
    plt.title('3D Trade-off: Energy vs Latency vs Accuracy')
    
    # Colorbar
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label('F1 Score')
    
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot Pareto frontier for energy vs time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(10, 7))
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
            s=80, c='blue', alpha=0.7, label='Other Variants'
        )
    
    # Annotate points
    for _, row in df.iterrows():
        if row['variant'] == baseline:
            continue
        plt.annotate(
            get_variant_name(row['variant']), 
            (row['total_kwh'], row['runtime_s']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=9
        )
    
    # Formatting
    plt.xlabel("Total Energy Consumption (kWh)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Energy-Time Trade-off")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_delta_energy_time_pareto(df: pd.DataFrame, baseline: str, out: Path):
    """Plot Pareto frontier for delta energy vs delta time trade-off"""
    if df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Set equal aspect ratio and limits
    max_delta = max(
        abs(df['Δtotal_kwh'].min()),
        abs(df['Δtotal_kwh'].max()),
        abs(df['Δruntime_s'].min()),
        abs(df['Δruntime_s'].max())
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
    points = df[['Δtotal_kwh', 'Δruntime_s']].values
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
                row['Δruntime_s'],
                s=100, c='green', marker='D', label=get_variant_name(row['variant']), zorder=4
            )
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values('Δtotal_kwh')
        plt.plot(
            pareto_sorted['Δtotal_kwh'], 
            pareto_sorted['Δruntime_s'],
            'g--', alpha=0.5, zorder=3
        )
    
    # Other variants with color coding based on quadrant
    other_df = df[~pareto_mask & (df['variant'] != baseline)]
    if not other_df.empty:
        # Group variants by quadrant
        better_both = other_df[(other_df['Δtotal_kwh'] < 0) & (other_df['Δruntime_s'] < 0)]
        worse_both = other_df[(other_df['Δtotal_kwh'] > 0) & (other_df['Δruntime_s'] > 0)]
        mixed = other_df[~((other_df['Δtotal_kwh'] < 0) & (other_df['Δruntime_s'] < 0)) & 
                        ~((other_df['Δtotal_kwh'] > 0) & (other_df['Δruntime_s'] > 0))]
        
        # Plot each group with different markers
        for _, row in better_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δruntime_s'],
                s=80, c='green', marker='o', label=get_variant_name(row['variant']), zorder=4
            )
            
        for _, row in worse_both.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δruntime_s'],
                s=80, c='red', marker='s', label=get_variant_name(row['variant']), zorder=4
            )
            
        for _, row in mixed.iterrows():
            plt.scatter(
                row['Δtotal_kwh'], 
                row['Δruntime_s'],
                s=80, c='gray', marker='^', label=get_variant_name(row['variant']), zorder=4
            )
    
    # Add quadrant annotations with background
    annotations = [
        (0.95, 0.95, 'Worse Energy\nWorse Time', 'right', 'top'),
        (0.05, 0.95, 'Better Energy\nWorse Time', 'left', 'top'),
        (0.95, 0.05, 'Worse Energy\nBetter Time', 'right', 'bottom'),
        (0.05, 0.05, 'Better Energy\nBetter Time', 'left', 'bottom')
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
    plt.ylabel("Δ Runtime (%)", fontsize=12)
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

# ------------------------------------------------------------------
# ------------------------------ main ------------------------------
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Green AI Pipeline Analyzer")
    parser.add_argument("--results_dir", type=Path, default=Path("variants"))
    parser.add_argument("--out_dir",     type=Path, default=Path("analysis"))
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
    
    # Create plots
    if not df_stage.empty:
        plot_stacked(df_stage, args.out_dir / "stage_energy.png")
    if not df_variant.empty:
        plot_energy_tradeoff(df_variant, args.baseline, args.out_dir / "energy_tradeoff.png")
        plot_cascade_effect(df_variant, args.cascade_variants, args.out_dir / "cascade_effect.png")
        plot_energy_time_pareto(df_variant, args.baseline, args.out_dir / "energy_time_pareto.png")
        plot_delta_energy_time_pareto(df_variant, args.baseline, args.out_dir / "delta_energy_time_pareto.png")
    if not df_inference.empty and len(df_inference) >= 3:
        plot_inference_tradeoffs(df_variant, args.out_dir / "3d_tradeoff.png")

    print("\nAnalysis complete. Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()