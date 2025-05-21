#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import argparse
import webbrowser
from datetime import datetime

def start_wandb_server(port=8080):
    """Start the wandb local server."""
    print(f"Starting wandb local server on port {port}...")
    subprocess.Popen(["wandb", "server", "--port", str(port)])
    # Wait a moment for the server to start
    import time
    time.sleep(2)
    return f"http://localhost:{port}"

def sync_wandb_data(wandb_dir):
    """Sync wandb data from the specified directory."""
    print(f"Syncing wandb data from {wandb_dir}...")
    subprocess.run(["wandb", "sync", str(wandb_dir)])

def main():
    parser = argparse.ArgumentParser(description="Visualize wandb data using local server")
    parser.add_argument("--results_dir", type=str, default="./results",
                      help="Directory containing the results (default: ./results)")
    parser.add_argument("--port", type=int, default=8080,
                      help="Port for wandb local server (default: 8080)")
    parser.add_argument("--open_browser", action="store_true",
                      help="Automatically open browser to wandb local server")
    
    args = parser.parse_args()
    
    # Convert to Path object
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        sys.exit(1)
    
    # Find all wandb directories
    wandb_dirs = list(results_dir.rglob("wandb"))
    if not wandb_dirs:
        print(f"No wandb directories found in {results_dir}")
        sys.exit(1)
    
    # Start wandb server
    server_url = start_wandb_server(args.port)
    
    # Sync all wandb directories
    for wandb_dir in wandb_dirs:
        print(f"\nProcessing {wandb_dir}...")
        sync_wandb_data(wandb_dir)
    
    # Open browser if requested
    if args.open_browser:
        print(f"\nOpening browser to {server_url}")
        webbrowser.open(server_url)
    
    print("\nWandb local server is running. You can access it at:", server_url)
    print("Press Ctrl+C to stop the server when done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping wandb local server...")
        sys.exit(0) 