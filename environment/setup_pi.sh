#!/bin/bash
# One-time setup on the Raspberry Pi 5 (run from home directory)
set -e
sudo apt update && sudo apt install -y git build-essential cmake python3-pip
# Build llama.cpp
if [ ! -d llama.cpp ]; then git clone https://github.com/ggerganov/llama.cpp; fi
cd llama.cpp && cmake -B build && cmake --build build --config Release -j4 && cd ..
# Model: Llama-3.2-1B-Instruct Q4_K_M (~0.8 GB). Download on the Pi:
mkdir -p models
echo "Download the model with:"
echo "  wget -O models/llama-3.2-1b-instruct-q4_k_m.gguf \\"
echo "    'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'"
echo "Then start the server:"
echo "  ./llama.cpp/build/bin/llama-server -m models/llama-3.2-1b-instruct-q4_k_m.gguf -c 2048 --port 8080"
# Sanity-check the PMIC power readout:
echo "PMIC check:"; vcgencmd pmic_read_adc | head -5
