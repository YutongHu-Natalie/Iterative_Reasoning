#!/bin/bash
#SBATCH --job-name=iterative_reasoning
#SBATCH --output=../logs/models_try_scale_%j.out
#SBATCH --error=../logs/models_try_scale_%j.err
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
source ./.venv/bin/activate
python models_try_scale.py
echo "Job finished at $(date)"