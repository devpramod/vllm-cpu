#!/bin/bash
# Sequential queue: DFlash k sweep -> per-k step timing -> p2 TP2_PP3 -> p3.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python

wait_idle() {
    for _ in $(seq 120); do
        pgrep -f "bin/vllm (serve|bench)" >/dev/null || return 0
        sleep 5
    done
    pkill -9 -f "bin/vllm (serve|bench)"
    sleep 10
}

step() { echo "[$(date +%T)] $*" >> logs/queue_next.log; }

step "k sweep"
VLLM_KV_GROUP_SIZE=20 $PY sweep.py --phase p4 --no-accuracy \
    --configs TP4_DFLASH_HYB_K3 TP4_DFLASH_HYB_K4 TP4_DFLASH_HYB_K5 TP4_DFLASH_HYB_K7 \
    --workloads gsm8k sharegpt_chat --concurrency 1 4 16 32 >> logs/p4_ksweep.log 2>&1
wait_idle

for k in 3 4 5 7; do
    step "step timer k=$k"
    VLLM_KV_GROUP_SIZE=20 $PY profile_step.py --config TP4_DFLASH_HYB_K$k \
        --concurrency 1 --prompts 8 >> logs/profile_k.log 2>&1
    wait_idle
done

step "p2 TP2_PP3"
$PY sweep.py --phase p2 --configs TP2_PP3 >> logs/p2.log 2>&1
wait_idle

for v in "mnbt1024:--max-num-batched-tokens 1024" \
         "mnbt2048:--max-num-batched-tokens 2048" \
         "mnbt8192:--max-num-batched-tokens 8192" \
         "mns8:--max-num-seqs 8" "mns16:--max-num-seqs 16"; do
    tag=${v%%:*}; extra=${v#*:}
    step "p3 $tag"
    $PY sweep.py --phase p3 --tag "$tag" --serve-args "$extra" >> logs/p3.log 2>&1
    wait_idle
done
step "all done"
