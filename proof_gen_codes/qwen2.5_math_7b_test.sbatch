#!/bin/bash
#SBATCH --job-name=qwen2.5_math_7b_balanced
#SBATCH --output=./logs/qwen2.5_math_7b_balanced_%j.out
#SBATCH --error=./logs/qwen2.5_math_7b_balanced_%j.err
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

python ./proof_gen_codes/run_qwen2.5math_test.py

echo "Job finished at $(date)"
