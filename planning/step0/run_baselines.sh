#!/usr/bin/env bash
# Step 0 baseline matrix: {Llama-3.1-8B tp2, Qwen3-30B-A3B tp1} x {inductor, eager}.
# Mirrors vLLM's own CPU perf-CI parameters (.buildkite/performance-benchmarks/tests/*-cpu.json).
#
# Usage: bash run_baselines.sh [llama-latency|llama-throughput|qwen-serve|all]
# Env:   NUM_RUNS (default 3), VLLM_DIR (default ~/vllm), RESULTS_DIR (default ./results)
#
# NEVER set VLLM_CPU_CI_ENV here: it silently forces backend=eager and
# invalidates the inductor arm of every measurement.
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-$HOME/vllm}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
NUM_RUNS="${NUM_RUNS:-3}"
WHAT="${1:-all}"

source "$VLLM_DIR/.venv/bin/activate"
mkdir -p "$RESULTS_DIR"
SHA=$(git -C "$VLLM_DIR" rev-parse --short HEAD)
STAMP=$(date +%Y-%m-%d)

# CPU perf-CI environment (latency/throughput-tests-cpu.json)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=120
export VLLM_CPU_SGL_KERNEL=1
export VLLM_CPU_KVCACHE_SPACE=40
if [[ -n "${VLLM_CPU_CI_ENV:-}" ]]; then
    echo "ERROR: VLLM_CPU_CI_ENV is set; unset it (forces eager backend)." >&2
    exit 1
fi

LLAMA=meta-llama/Llama-3.1-8B-Instruct
QWEN=Qwen/Qwen3-30B-A3B
COMMON_ARGS=(--dtype bfloat16 --distributed-executor-backend mp --block-size 128
             --max-num-batched-tokens 2048 --max-num-seqs 256 --trust-remote-code
             --disable-log-stats)

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_llama_latency() {
    for mode in inductor eager; do
        extra=(); [[ $mode == eager ]] && extra=(--enforce-eager)
        for i in $(seq 1 "$NUM_RUNS"); do
            out="$RESULTS_DIR/$STAMP-$SHA-llama-latency-$mode-run$i.log"
            log "llama latency $mode run $i -> $out"
            vllm bench latency --model "$LLAMA" -tp 2 "${COMMON_ARGS[@]}" \
                --num-iters-warmup 5 --num-iters 15 "${extra[@]}" 2>&1 | tee "$out"
        done
    done
}

run_llama_throughput() {
    dataset="$RESULTS_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
    if [[ ! -f "$dataset" ]]; then
        log "downloading ShareGPT dataset"
        curl -L -o "$dataset" \
          https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    for mode in inductor eager; do
        extra=(); [[ $mode == eager ]] && extra=(--enforce-eager)
        for i in $(seq 1 "$NUM_RUNS"); do
            out="$RESULTS_DIR/$STAMP-$SHA-llama-throughput-$mode-run$i.log"
            log "llama throughput $mode run $i -> $out"
            vllm bench throughput --model "$LLAMA" -tp 2 "${COMMON_ARGS[@]}" \
                --dataset-name sharegpt --dataset-path "$dataset" \
                --num-prompts 200 --backend vllm "${extra[@]}" 2>&1 | tee "$out"
        done
    done
}

run_qwen_serve() {
    for mode in inductor eager; do
        extra=(); [[ $mode == eager ]] && extra=(--enforce-eager)
        srv_log="$RESULTS_DIR/$STAMP-$SHA-qwen-server-$mode.log"
        log "starting qwen server ($mode) -> $srv_log"
        vllm serve "$QWEN" -tp 1 "${COMMON_ARGS[@]}" "${extra[@]}" >"$srv_log" 2>&1 &
        srv_pid=$!
        trap 'kill $srv_pid 2>/dev/null || true' EXIT
        until curl -sf http://localhost:8000/health >/dev/null; do
            kill -0 $srv_pid 2>/dev/null || { echo "server died, see $srv_log" >&2; exit 1; }
            sleep 10
        done
        # First inductor boot compiles for minutes; grep the compile time afterwards:
        grep -iE 'compil.*(second|took|time)' "$srv_log" || true
        for i in $(seq 1 "$NUM_RUNS"); do
            out="$RESULTS_DIR/$STAMP-$SHA-qwen-serve-$mode-run$i.log"
            log "qwen bench serve $mode run $i -> $out"
            vllm bench serve --backend vllm --model "$QWEN" \
                --dataset-name random --random-input-len 128 --random-output-len 128 \
                --num-prompts 200 --ignore-eos --temperature 0 2>&1 | tee "$out"
        done
        kill $srv_pid; wait $srv_pid 2>/dev/null || true
        trap - EXIT
    done
}

case "$WHAT" in
    llama-latency)    run_llama_latency ;;
    llama-throughput) run_llama_throughput ;;
    qwen-serve)       run_qwen_serve ;;
    all)              run_llama_latency; run_llama_throughput; run_qwen_serve ;;
    *) echo "usage: $0 [llama-latency|llama-throughput|qwen-serve|all]" >&2; exit 1 ;;
esac

log "done. Summarize medians into results/$STAMP-baselines.md (see results-template.md)"
