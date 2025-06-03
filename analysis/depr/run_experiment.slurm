#!/bin/bash

#SBATCH --account=def-tusharma
#SBATCH --mail-user=saurabh@dal.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=greenai-exp
#SBATCH --output=greenai-%j.out
#SBATCH --error=greenai-%j.err
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=150G

# Clean environment and load required modules
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.1
module load arrow/18.1.0
module load git

# Store original location and project name
ORIGINAL_PWD=$(pwd)
PROJECT_NAME=$(basename "$ORIGINAL_PWD")
PROJECT_DIR_IN_TMPDIR="$SLURM_TMPDIR/$PROJECT_NAME"

# Parameters
TASK=${1:-"code_translation"}  # Default to code_translation if no task specified
VENV_NAME="greenai_venv" # Name for the virtual environment

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Task: $TASK"
echo "Original PWD: $ORIGINAL_PWD"
echo "Project Name: $PROJECT_NAME"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "Project dir in SLURM_TMPDIR: $PROJECT_DIR_IN_TMPDIR"
echo "SLURM_MAIL_TYPE: $SLURM_MAIL_TYPE"

# Copy project to SLURM_TMPDIR and cd into it
echo "Copying project from $ORIGINAL_PWD to $PROJECT_DIR_IN_TMPDIR..."
mkdir -p "$SLURM_TMPDIR" # Ensure SLURM_TMPDIR base exists, though Slurm should create it
cp -R "$ORIGINAL_PWD" "$SLURM_TMPDIR/" # Copies the project folder into SLURM_TMPDIR
cd "$PROJECT_DIR_IN_TMPDIR"
echo "Current working directory: $(pwd)"

# Create and activate virtual environment (now inside $PROJECT_DIR_IN_TMPDIR)
VENV_PATH="./$VENV_NAME" # Relative to current dir ($PROJECT_DIR_IN_TMPDIR)
echo "Creating virtual environment at $VENV_PATH..."
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated."

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install --no-index -r requirements.txt # requirements.txt is now in $PROJECT_DIR_IN_TMPDIR
echo "Dependencies installed."

# Create results directory (now inside $PROJECT_DIR_IN_TMPDIR)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR_RELATIVE="results/${TASK}_${TIMESTAMP}" # Relative path for use within $PROJECT_DIR_IN_TMPDIR
mkdir -p "$RESULTS_DIR_RELATIVE"
echo "Results will be stored in $PROJECT_DIR_IN_TMPDIR/$RESULTS_DIR_RELATIVE"

# Run the experiment (python scripts run from $PROJECT_DIR_IN_TMPDIR)
echo "Running experiment for task: $TASK..."
case $TASK in
    "code_translation")
        srun python variants/V0_baseline/tasks/code_translation/baseline.py
        ;;
    "vulnerability_detection")
        srun python variants/V0_baseline/tasks/vulnerability_detection/baseline.py
        ;;
    *)
        echo "Unknown task: $TASK"
        cd "$ORIGINAL_PWD" # Go back before exiting on error if needed
        rm -rf "$PROJECT_DIR_IN_TMPDIR"
        exit 1
        ;;
esac
echo "Experiment execution finished."

# Consolidate logs and results into the results directory within $SLURM_TMPDIR
echo "Consolidating logs and results in $PROJECT_DIR_IN_TMPDIR/$RESULTS_DIR_RELATIVE..."
# SLURM .out/.err files are in $ORIGINAL_PWD initially
if [ -f "$ORIGINAL_PWD/greenai-$SLURM_JOB_ID.out" ]; then
    cp "$ORIGINAL_PWD/greenai-$SLURM_JOB_ID.out" "$RESULTS_DIR_RELATIVE/"
else
    echo "Warning: SLURM output file $ORIGINAL_PWD/greenai-$SLURM_JOB_ID.out not found."
fi
if [ -f "$ORIGINAL_PWD/greenai-$SLURM_JOB_ID.err" ]; then
    cp "$ORIGINAL_PWD/greenai-$SLURM_JOB_ID.err" "$RESULTS_DIR_RELATIVE/"
else
    echo "Warning: SLURM error file $ORIGINAL_PWD/greenai-$SLURM_JOB_ID.err not found."
fi

# If energy_logs directory exists at the root of $PROJECT_DIR_IN_TMPDIR, move it into the results dir
if [ -d "./energy_logs" ]; then
    echo "Moving ./energy_logs into $RESULTS_DIR_RELATIVE"
    mv ./energy_logs "$RESULTS_DIR_RELATIVE/"
fi
echo "Log consolidation complete."

# Copy the consolidated results back to the original project location on network storage
echo "Copying results from $PROJECT_DIR_IN_TMPDIR/$RESULTS_DIR_RELATIVE back to $ORIGINAL_PWD/results..."
mkdir -p "$ORIGINAL_PWD/results" # Ensure target directory exists in original location
cp -R "$RESULTS_DIR_RELATIVE" "$ORIGINAL_PWD/results/"
echo "Results copied back."

# Change back to original directory for Git operations
cd "$ORIGINAL_PWD"
echo "Changed back to original directory: $(pwd)"

# Set up Git configuration for pushing results
git config --global user.name "Saurabhsingh Rajput"
git config --global user.email "saurabh@dal.ca"

# Add and commit results
echo "Committing results to Git..."
GIT_ADD_PATH="results/${TASK}_${TIMESTAMP}" # Path relative to current dir ($ORIGINAL_PWD)
git add "$GIT_ADD_PATH"
git commit -m "Add results for $TASK experiment (Job ID: $SLURM_JOB_ID, from $SLURM_TMPDIR)"
git push origin HEAD
echo "Results pushed to Git."

# Cleanup
echo "Cleaning up..."
deactivate # Deactivate venv
# Remove the copied project from $SLURM_TMPDIR. Venv is inside it.
rm -rf "$PROJECT_DIR_IN_TMPDIR"
echo "Cleanup of $PROJECT_DIR_IN_TMPDIR complete."

echo "Experiment completed successfully!" 