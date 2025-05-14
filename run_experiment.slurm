#!/bin/bash
#SBATCH --job-name=greenai-exp
#SBATCH --output=greenai-%j.out
#SBATCH --error=greenai-%j.err
#SBATCH --time=${SLURM_JOB_TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}
#SBATCH --gres=${SLURM_GPU}
#SBATCH --mem=${SLURM_MEM}

# Source configuration
source slurm_config.sh

# Parameters
TASK=${1:-"code_translation"}  # Default to code_translation if no task specified
REPO_URL="https://github.com/yourusername/greenai-pipeline-empirical-study.git"
BRANCH="main"
VENV_NAME="greenai_venv"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Task: $TASK"

# Load required modules (adjust based on your HPC setup)
module load ${PYTHON_VERSION}
module load ${CUDA_VERSION}
module load git

# Create and activate virtual environment
python -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# Set up Git configuration
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# Clone repository
git clone $REPO_URL
cd greenai-pipeline-empirical-study
git checkout $BRANCH

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TASK}_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Run the experiment
case $TASK in
    "code_translation")
        python variants/V0_baseline/tasks/code_translation/baseline.py
        ;;
    "vulnerability_detection")
        python variants/V0_baseline/tasks/vulnerability_detection/baseline.py
        ;;
    *)
        echo "Unknown task: $TASK"
        exit 1
        ;;
esac

# Copy results and logs
cp -r energy_logs $RESULTS_DIR/
cp greenai-$SLURM_JOB_ID.out $RESULTS_DIR/
cp greenai-$SLURM_JOB_ID.err $RESULTS_DIR/

# Commit and push results
git add $RESULTS_DIR
git commit -m "Add results for $TASK experiment (Job ID: $SLURM_JOB_ID)"
git push origin $BRANCH

# Cleanup
deactivate
cd ..
rm -rf greenai-pipeline-empirical-study

echo "Experiment completed successfully!" 