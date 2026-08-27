#!/bin/bash
# One-time setup for a venv dedicated to running DeepSeekMath-7B.
#
# Why a separate venv: DeepSeekMath's tokenizer.json (released Feb 2024) is
# parsed incorrectly by the transformers==5.14.1 / tokenizers==0.22.2 pins in
# the main ./venv (required there for vLLM and the newer Gemma/Qwen scripts).
# Newer `tokenizers` silently drops whitespace while *encoding* text with
# this tokenizer's pretokenizer rules, producing garbled/run-together output.
# transformers==4.39.3 + tokenizers==0.15.2 (contemporary with the model's
# release) round-trip it correctly -- confirmed by direct testing.
#
# Run this once on a login node (needs internet access for pip):
#   bash setup_deepseekmath_venv.sh

set -euo pipefail
cd "$(dirname "$0")"

python3.12 -m venv venv_deepseekmath
source ./venv_deepseekmath/bin/activate

pip install --upgrade pip
pip install torch==2.11.0
pip install "transformers==4.39.3" "tokenizers==0.15.2" accelerate datasets tqdm

echo "venv_deepseekmath ready."
