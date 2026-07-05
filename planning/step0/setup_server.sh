#!/usr/bin/env bash
# Step 0: vLLM CPU dev environment on an x86_64 Linux server.
# Usage: bash setup_server.sh [--full-build]
#   default      : editable install using the precompiled CPU wheel (Python-only changes)
#   --full-build : build csrc/cpu from source (needed for Stage 3+ kernel work)
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-$HOME/vllm}"
FULL_BUILD=0
[[ "${1:-}" == "--full-build" ]] && FULL_BUILD=1

echo "== ISA check (AMX changes which MoE/attention kernels run) =="
lscpu | grep -oE 'amx[^ ]*|avx512[^ ]*' | sort -u || echo "WARNING: no AMX/AVX512 detected"
echo "== NUMA layout =="
lscpu | grep -E '^(Socket|NUMA)' || true

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$VLLM_DIR" ]]; then
    git clone https://github.com/vllm-project/vllm "$VLLM_DIR"
fi
cd "$VLLM_DIR"

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install

if [[ "$FULL_BUILD" == "1" ]]; then
    echo "== Full source build (csrc/cpu) =="
    uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
    uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
    VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
else
    echo "== Editable install with precompiled CPU wheel =="
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
        uv pip install -e .
fi

uv pip install tlparse depyf

echo
echo "Done. Remaining manual steps:"
echo "  1. huggingface-cli login   (Llama-3.1-8B-Instruct is gated; request access on the HF page first)"
echo "  2. bash run_baselines.sh   (from this directory)"
