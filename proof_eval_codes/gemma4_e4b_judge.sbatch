#!/bin/bash
#SBATCH --job-name=gemma4_e4b_judge
#SBATCH --output=./logs/gemma4_e4b_judge_%j.out
#SBATCH --error=./logs/gemma4_e4b_judge_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /local/scratch/yhu383/Iterative_Reasoning
source ./venv/bin/activate

# Add --thinking to enable Gemma4 thinking mode before it emits the judge
# JSON (slower). Non-thinking mode runs by default.
python ./proof_eval_codes/run_gemma4_e4b_judge.py

echo "Job finished at $(date)"
