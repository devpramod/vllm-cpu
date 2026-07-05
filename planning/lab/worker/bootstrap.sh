#!/usr/bin/env bash
# Worker bootstrap. Runs ON the worker (labctl setup ships + executes it). Idempotent.
# Root fs on xeon is FULL: everything lives under REMOTE_ROOT on /mnt/nvme, and this
# script aborts if any derived path resolves to the root filesystem.
#
# Usage: bash bootstrap.sh <remote_root> <fork_url> <branch> [--full-build]
set -euo pipefail

REMOTE_ROOT="${1:?usage: bootstrap.sh <remote_root> <fork_url> <branch> [--full-build]}"
FORK_URL="${2:?fork url required}"
BRANCH="${3:?branch required}"
FULL_BUILD="${4:-}"

REPO_DIR="$REMOTE_ROOT/vllm-cpu"
RUNS_DIR="$REMOTE_ROOT/lab-runs"
ENV_FILE="$REMOTE_ROOT/lab.env"

mkdir -p "$REMOTE_ROOT" "$RUNS_DIR" "$REMOTE_ROOT/tmp" "$REMOTE_ROOT/datasets" \
         "$REMOTE_ROOT/.local" \
         "$REMOTE_ROOT"/.cache/{uv,huggingface,vllm,torchinductor,triton,ccache}

root_dev=$(df --output=source / | tail -1)
for p in "$REMOTE_ROOT" "$REMOTE_ROOT/tmp" "$REMOTE_ROOT/.cache"; do
    dev=$(df --output=source "$p" | tail -1)
    if [[ "$dev" == "$root_dev" ]]; then
        echo "FATAL: $p resolves to the root filesystem ($dev); root is full." >&2
        exit 1
    fi
done

cat > "$ENV_FILE" <<EOF
# Sourced at the start of every lab remote command. Pins ALL caches/scratch
# off the (full) root filesystem. bootstrap.sh owns this file.
export UV_CACHE_DIR=$REMOTE_ROOT/.cache/uv
export XDG_CACHE_HOME=$REMOTE_ROOT/.cache
export HF_HOME=$REMOTE_ROOT/.cache/huggingface
export VLLM_CACHE_ROOT=$REMOTE_ROOT/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$REMOTE_ROOT/.cache/torchinductor
export TRITON_CACHE_DIR=$REMOTE_ROOT/.cache/triton
export TMPDIR=$REMOTE_ROOT/tmp
export CCACHE_DIR=$REMOTE_ROOT/.cache/ccache
export PATH="$REMOTE_ROOT/.local/uv-bin:\$PATH"
EOF
source "$ENV_FILE"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$REMOTE_ROOT/.local/uv-bin" sh
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone --branch "$BRANCH" "$FORK_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" fetch origin
fi

cd "$REPO_DIR"
[[ -d .venv ]] || uv venv --python 3.12

if [[ "$FULL_BUILD" == "--full-build" ]]; then
    uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
    uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
    VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
else
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
        uv pip install -e . --torch-backend cpu --index-strategy unsafe-best-match
fi

SHAREGPT="$REMOTE_ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ ! -f "$SHAREGPT" ]]; then
    curl -L -o "$SHAREGPT" \
      https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

# Host probe consumed by run.json's "host" block.
.venv/bin/python - > "$REMOTE_ROOT/host.json" <<'EOF'
import json, re, subprocess
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
cpu = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines()
            if l.startswith("Model name")), "unknown")
numa = next((int(l.split(":", 1)[1]) for l in lscpu.splitlines()
             if l.startswith("NUMA node(s)")), 1)
isa = sorted(set(re.findall(r"\b(amx\w*|avx512\w*)\b", lscpu)))
try:
    import torch
    tv = torch.__version__
except Exception:
    tv = "unknown"
print(json.dumps({"cpu": cpu, "numa_nodes": numa, "isa": isa, "torch": tv}))
EOF

echo "bootstrap OK: repo=$REPO_DIR runs=$RUNS_DIR env=$ENV_FILE"
cat "$REMOTE_ROOT/host.json"
