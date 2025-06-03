#!/usr/bin/env python3
"""
Visualise a *local* wandb run by spinning up the wandb local server,
copying the latest run directory, syncing it, and saving key metrics
as JSON that can be plotted with any tool.

Tested with wandb==0.16.4
"""

import argparse, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from socket import create_connection

# --------------------------------------------------------------------------- #
# 1. helpers
# --------------------------------------------------------------------------- #
def wait_on_port(host: str, port: int, timeout: int = 30):
    """Block until TCP port is open."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"wandb server on {host}:{port} did not open in {timeout}s")

def start_wandb_server(port: int = 8080) -> str | None:
    """Run `wandb server start` in a detached process and wait until it is live."""
    env = os.environ.copy()
    env["WANDB_BASE_URL"] = f"http://localhost:{port}"
    env["WANDB_MODE"] = "offline"          # ensure no accidental cloud push

    print(f"[wandb] launching local server on port {port} ...")
    try:
        subprocess.Popen(
            ["wandb", "server", "start", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
        )
        wait_on_port("localhost", port)
        return env["WANDB_BASE_URL"]
    except (TimeoutError, subprocess.SubprocessError) as e:
        print(f"[!] Failed to start wandb server: {str(e)}")
        print("[!] Continuing with data extraction only...")
        return None

def find_latest_run(results_dir: Path) -> Path | None:
    """Return newest timestamped sub-dir inside `results_dir`."""
    if not results_dir.exists():
        return None
    runs = [d for d in results_dir.iterdir() if d.is_dir() and "_" in d.name]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)

def extract_jsonl(jsonl_path: Path, out_json: Path, metrics_out: Path):
    """Convert wandb-events.jsonl → pretty JSON; pull out a few scalar keys."""
    events, summary = [], {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                evt = json.loads(line)
                events.append(evt)
                summary.update({k: v for k, v in evt.items()
                                if k in {"test_metrics",
                                         "energy_consumption",
                                         "duration_seconds"}})
            except json.JSONDecodeError:
                continue
    out_json.write_text(json.dumps(events, indent=2))
    metrics_out.write_text(json.dumps(summary, indent=2))
    print(f"[✓] wrote {out_json.name} and {metrics_out.name}")

# --------------------------------------------------------------------------- #
# 2. main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="e.g. V0_baseline")
    p.add_argument("--task",    required=True, help="e.g. vulnerability_detection")
    p.add_argument("--port",    type=int, default=8080)
    p.add_argument("--output_dir", default="./plots", type=Path)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "variants" / args.variant / args.task / "results"
    if not results_dir.exists():
        sys.exit(f"[x] {results_dir} not found")

    run_dir = find_latest_run(results_dir)
    if run_dir is None:
        sys.exit(f"[x] no timestamped run directories in {results_dir}")
    print(f"[•] using run directory  {run_dir}")

    wandb_src = run_dir / "wandb"
    if not wandb_src.exists():
        sys.exit(f"[x] expected wandb/ inside {run_dir}")

    # ----------------------------------------------------------------------- #
    # 2.  spin up local server
    # ----------------------------------------------------------------------- #
    base_url = start_wandb_server(args.port)
    if base_url:
        print(f"[✓] wandb local ready at {base_url}")

    # ----------------------------------------------------------------------- #
    # 3.  copy + sync logs
    # ----------------------------------------------------------------------- #
    out_run = args.output_dir.resolve() / run_dir.name
    if out_run.exists():
        shutil.rmtree(out_run)
    shutil.copytree(wandb_src, out_run)
    print(f"[•] copied wandb dir to  {out_run}")

    # Only try to sync if server is running
    if base_url:
        try:
            subprocess.run(["wandb", "sync", "--sync-all", str(out_run)], check=False)
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to sync with wandb: {str(e)}")
            print("[!] Continuing with local data only...")

    # ----------------------------------------------------------------------- #
    # 4.  JSONL → JSON (+ skinny metrics)
    # ----------------------------------------------------------------------- #
    jsonl_file = out_run / "files" / "wandb-events.jsonl"
    if jsonl_file.exists():
        plots_dir = out_run / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        extract_jsonl(jsonl_file,
                      plots_dir / "wandb_events.json",
                      plots_dir / "metrics_summary.json")
    else:
        print("[!] wandb-events.jsonl not found – nothing to parse")

    if base_url:
        print("\nOpen your browser:", base_url)
        print("Press Ctrl+C when done.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[✓] shutdown requested – exiting")
    else:
        print("\n[✓] Data has been extracted to:", out_run)
        print("You can find:")
        print("1. Raw wandb data in:", out_run)
        print("2. Processed metrics in:", out_run / "plots")
        print("   - wandb_events.json (complete data)")
        print("   - metrics_summary.json (key metrics)")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
