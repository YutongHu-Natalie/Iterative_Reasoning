#!/bin/bash
#SBATCH --job-name=ai_detect_premiere
#SBATCH --output=./logs/ai_detect_premiere_%j.out
#SBATCH --error=./logs/ai_detect_premiere_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=34:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /local/scratch/yhu383/Iterative_Reasoning
source ./venv/bin/activate

# MAGE's official preprocessing needs clean-text; roberta (coai) is gated and
# needs HF_TOKEN from an account that has been granted access.
# pip install clean-text
# export HF_TOKEN=...

# Pick any subset: qwen3_8b_think qwen3_8b_nothink gemma4_e4b_think
# gemma4_e4b_nothink mage modernbert roberta  (or: all)
# Add --limit 2 for a quick smoke test, --n-runs 3 for majority voting.
# Full dataset (Data/full_data.json, results in TestResult/full/; reruns resume):
#   classifiers on everything:
#     python ./AI_Detect/Premiere/run_detection.py --data full --models mage modernbert roberta
#   slow LLM judges on a balanced sample (200 LLM + 200 human):
#     python ./AI_Detect/Premiere/run_detection.py --data full --sample 200 --models qwen3_8b_think
#   only problems with both an LLM and a human proof (controls for source):
#     python ./AI_Detect/Premiere/run_detection.py --data full --paired-only --out-dir ./AI_Detect/Premiere/TestResult/full_paired --models mage
python ./AI_Detect/Premiere/run_detection.py --data full --models gemma4_e4b_think 
#gemma4_e4b_think gemma4_e4b_nothink 

echo "Job finished at $(date)"
