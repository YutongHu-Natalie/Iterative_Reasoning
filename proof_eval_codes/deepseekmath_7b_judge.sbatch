#!/bin/bash
#SBATCH --job-name=deepseekmath_7b_judge
#SBATCH --output=./logs/deepseekmath_7b_judge_%j.out
#SBATCH --error=./logs/deepseekmath_7b_judge_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /local/scratch/yhu383/Iterative_Reasoning

# Uses the dedicated venv_deepseekmath, not ./venv -- see
# proof_gen_codes/deepseekmath_7b_test.sbatch for why.
source ./venv_deepseekmath/bin/activate

# Pass instruct or rl as the first sbatch arg, e.g.:
#   sbatch deepseekmath_7b_judge.sbatch rl
VARIANT=${1:-instruct}
python ./proof_eval_codes/run_deepseekmath_7b_judge.py --variant "$VARIANT"

echo "Job finished at $(date)"
