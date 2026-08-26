#!/bin/bash
#SBATCH --job-name=deepseekmath_7b_balanced
#SBATCH --output=./logs/deepseekmath_7b_balanced_%j.out
#SBATCH --error=./logs/deepseekmath_7b_balanced_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /local/scratch/yhu383/Iterative_Reasoning
source ./venv/bin/activate

# Pass instruct or rl as the first sbatch arg, e.g.:
#   sbatch deepseekmath_7b_test.sbatch rl
VARIANT=${1:-instruct}
python ./run_deepseekmath_7b.py --variant "$VARIANT"

echo "Job finished at $(date)"
