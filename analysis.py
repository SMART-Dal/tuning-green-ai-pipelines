#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt

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


def _safe_mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


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
        # Get the most recent run directory
        run_dirs = sorted(results_dir.glob("default_*"), key=lambda x: x.name, reverse=True)
        if not run_dirs:
            print(f"No run directories found in {results_dir}")
            continue
        print(f"Using run directory: {run_dirs[0]}")
        yield variant_dir.name, run_dirs[0]


def load_energy(path: Path) -> Dict[str, Any]:
    """Load energy stats from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load test metrics from JSON file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# ---------------------- aggregation logic -------------------------
# ------------------------------------------------------------------

def aggregate(results_root: Path):
    variant_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for variant, run_dir in walk_results(results_root):
        print(f"\nProcessing variant: {variant}")
        print(f"Run directory: {run_dir}")
        
        energy_path = run_dir / "energy_stats_train.json"
        metrics_path = run_dir / "test_metrics.json"
        
        print(f"Energy path exists: {energy_path.exists()}")
        print(f"Metrics path exists: {metrics_path.exists()}")
        
        if not energy_path.exists():
            print(f"Skipping {variant} - no energy stats found")
            continue
            
        data = {
            "energy": load_energy(energy_path),
            "metrics": load_metrics(metrics_path),
        }
        variant_runs[variant].append(data)
        print(f"Added data for {variant}")

    print("\nCollected data for variants:", list(variant_runs.keys()))
    
    # --- per‑variant summaries ------------------------------------
    by_variant_rows = []
    by_stage_rows = []

    for variant, runs in variant_runs.items():
        print(f"\nAggregating data for {variant}")
        # aggregate totals
        total_kwh = _safe_mean([r["energy"]["energy_consumed"] for r in runs])
        cpu_kwh = _safe_mean([r["energy"].get("cpu_energy", 0) for r in runs])
        gpu_kwh = _safe_mean([r["energy"].get("gpu_energy", 0) for r in runs])
        ram_kwh = _safe_mean([r["energy"].get("ram_energy", 0) for r in runs])
        runtime_s = _safe_mean([r["energy"]["duration"] for r in runs])
        f1 = _safe_mean([r["metrics"].get("eval_f1", float("nan")) for r in runs])
        peak_mem = _safe_mean([r["energy"].get("max_gpu_mem", float("nan")) for r in runs])

        effectiveness = f1 / total_kwh if total_kwh else float("nan")

        row = dict(
            variant=variant,
            total_kwh=total_kwh,
            cpu_kwh=cpu_kwh,
            gpu_kwh=gpu_kwh,
            ram_kwh=ram_kwh,
            runtime_s=runtime_s,
            f1=f1,
            peak_mem_gb=peak_mem,
            f1_per_kwh=effectiveness,
        )
        print(f"Adding row: {row}")
        by_variant_rows.append(row)

        # stage‑wise kWh from emissions_base CSV
        run_dir = next(d for v, d in walk_results(results_root) if v == variant)
        emissions_base = run_dir / f"emissions_base_{runs[0]['energy']['run_id']}.csv"
        if emissions_base.exists():
            print(f"Processing stage data from {emissions_base}")
            df_stages = pd.read_csv(emissions_base)
            for _, row in df_stages.iterrows():
                by_stage_rows.append(dict(
                    variant=variant,
                    stage=row['task_name'],
                    kwh=row['energy_consumed'],
                ))

    if not by_variant_rows:
        print("WARNING: No variant data collected!")
        return pd.DataFrame(), pd.DataFrame()

    print("\nCreating DataFrames...")
    df_variant = pd.DataFrame(by_variant_rows)
    print("Variant DataFrame columns:", df_variant.columns.tolist())
    df_variant = df_variant.sort_values("variant")
    df_stage = pd.DataFrame(by_stage_rows)

    return df_variant, df_stage

# ------------------------------------------------------------------
# ---------------------------- plots -------------------------------
# ------------------------------------------------------------------

def plot_stacked(df_stage: pd.DataFrame, out: Path):
    if df_stage.empty:
        return
    # ensure same stage order across variants
    pivot = df_stage.pivot(index="variant", columns="stage", values="kwh").fillna(0)
    # reorder
    pivot = pivot[[c for c in STAGE_ORDER if c in pivot.columns]]
    pivot.div(1000).plot(kind="bar", stacked=True, figsize=(11,6))
    plt.ylabel("Energy (kWh)")
    plt.title("Stage‑wise energy per variant")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_total(df_variant: pd.DataFrame, out: Path):
    df_variant.set_index("variant")["total_kwh"].div(1000).plot(kind="bar", figsize=(9,4))
    plt.ylabel("Total energy (kWh)")
    plt.title("Total GPU+CPU energy per variant")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_f1_vs_energy(df_variant: pd.DataFrame, out: Path):
    plt.figure(figsize=(6,6))
    x = df_variant["total_kwh"].values/1000
    y = df_variant["f1"].values
    plt.scatter(x,y)
    for xi, yi, label in zip(x,y, df_variant["variant"]):
        plt.text(xi, yi, label, fontsize=8, va="bottom", ha="left")
    plt.xlabel("Total kWh")
    plt.ylabel("F1 score")
    plt.title("Model performance vs energy cost")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_runtime_vs_mem(df_variant: pd.DataFrame, out: Path):
    if df_variant["peak_mem_gb"].isna().all():
        return  # no mem data
    plt.figure(figsize=(6,6))
    plt.scatter(df_variant["peak_mem_gb"], df_variant["runtime_s"], c="tab:orange")
    for xi, yi, label in zip(df_variant["peak_mem_gb"], df_variant["runtime_s"], df_variant["variant"]):
        plt.text(xi, yi, label, fontsize=8, va="bottom", ha="left")
    plt.xlabel("Peak GPU memory (GB)")
    plt.ylabel("Train runtime (s)")
    plt.title("Speed vs memory")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# ------------------------------------------------------------------
# ------------------------------ main ------------------------------
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("variants"))
    parser.add_argument("--out_dir",     type=Path, default=Path("analysis"))
    parser.add_argument("--plots",       default="all",
                        help="comma‑separated list: stacked,total,f1,rt_mem or 'all'")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_variant, df_stage = aggregate(args.results_dir)

    # save csvs
    df_variant.to_csv(args.out_dir / "by_variant.csv", index=False)
    df_stage.to_csv(args.out_dir / "by_variant_stage.csv", index=False)

    wanted = set([p.strip() for p in args.plots.split(',')])
    if "all" in wanted:
        wanted = {"stacked","total","f1","rt_mem"}

    if "stacked" in wanted and not df_stage.empty:
        plot_stacked(df_stage, args.out_dir / "stage_energy.png")
    if "total" in wanted and not df_variant.empty:
        plot_total(df_variant, args.out_dir / "total_energy.png")
    if "f1" in wanted and not df_variant.empty:
        plot_f1_vs_energy(df_variant, args.out_dir / "f1_vs_energy.png")
    if "rt_mem" in wanted and not df_variant.empty:
        plot_runtime_vs_mem(df_variant, args.out_dir / "runtime_vs_mem.png")

    print("Analysis complete. CSVs and plots saved to", args.out_dir)


if __name__ == "__main__":
    main()
