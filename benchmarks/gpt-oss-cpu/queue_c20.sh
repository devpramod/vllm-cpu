#!/bin/bash
# Client sizing at 20 concurrent users, 16K context; best configs only.
cd "$(dirname "$0")"
PY=.venv/bin/python
CTX="--max-model-len 16384"
$PY sweep.py --phase p8 --serve-args "$CTX" > logs/p8.log 2>&1
./sync_results.sh >> logs/sync.log 2>&1
$PY replicas.py --layouts TP2x3 --concurrency 20 --tag c20 --serve-args "$CTX" \
    --workloads rand_2k_256 rand_8k_1k rand_1k_2k rand_12k_2k sharegpt_chat \
    > logs/p7_c20.log 2>&1
./sync_results.sh >> logs/sync.log 2>&1
$PY sweep.py --phase p9 --serve-args "$CTX" > logs/p9.log 2>&1
./sync_results.sh >> logs/sync.log 2>&1
$PY ocr/sweep_ocr.py --lane c20 > logs/ocr_c20.log 2>&1
$PY ocr/replicas_ocr.py --layouts TP1x6 --policies round_robin --concurrency 20 \
    --workloads omnidoc_greedy cord_greedy omnidoc > logs/ocr_replicas_c20.log 2>&1
./sync_results.sh >> logs/sync.log 2>&1
echo "C20 QUEUE DONE $(date +%T)"
