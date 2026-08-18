#!/bin/bash
#SBATCH --job-name=qwen3_8b_balanced
#SBATCH --output=./logs/qwen3_8b_balanced_%j.out
#SBATCH --error=./logs/qwen3_8b_balanced_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /local/scratch/yhu383/Iterative_Reasoning
source ./venv/bin/activate

# Add --thinking to enable Qwen3 thinking mode (slower, uses a different
# sampling config). Non-thinking mode runs by default.
python ./run_qwen3_8b_test.py

echo "Job finished at $(date)"
