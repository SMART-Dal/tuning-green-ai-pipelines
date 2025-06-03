#!/bin/bash

# Repository configuration
export REPO_URL="git@github.com:SMART-Dal/greenai-pipeline-empirical-study.git"
export BRANCH="main"

# Git configuration
export GIT_NAME="Saurabhsingh Rajput"
export GIT_EMAIL="saurabh@dal.ca"

# SLURM configuration
export SLURM_ACCOUNT="def-tusharma"
export SLURM_JOB_TIME="8:00:00"
export SLURM_MAIL_USER="saurabh@dal.ca"
export SLURM_MAIL_TYPE="ALL"
export SLURM_CPUS_PER_TASK=4
export SLURM_MEM="0"
export SLURM_GPU="gpu:a100:1"

# Python/CUDA versions
export PYTHON_VERSION="python/3.11"
export CUDA_VERSION="cuda/12.1"
export STD_ENV_MODULE="StdEnv/2023"
export ARROW_MODULE="arrow/18.1.0"

# Virtual environment name
export VENV_NAME="greenai_venv" 