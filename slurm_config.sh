#!/bin/bash

# Repository configuration
export REPO_URL="https://github.com/yourusername/greenai-pipeline-empirical-study.git"
export BRANCH="main"

# Git configuration
export GIT_NAME="Your Name"
export GIT_EMAIL="your.email@example.com"

# SLURM configuration
export SLURM_JOB_TIME="24:00:00"
export SLURM_CPUS_PER_TASK=8
export SLURM_MEM="0"
export SLURM_GPU="gpu:1"

# Python/CUDA versions
export PYTHON_VERSION="python/3.11"
export CUDA_VERSION="cuda/12.1"

# Virtual environment name
export VENV_NAME="greenai_venv" 