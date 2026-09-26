#!/bin/bash
cd "$(dirname "$0")"
M="--max-num-seqs 128"
.venv/bin/python sweep.py --phase p5 --concurrency 64 128 --prompts-per-user 2 --serve-args "$M" --tag mns128
.venv/bin/python sweep.py --phase p6 --workloads gsm8k humaneval --concurrency 64 128 --prompts-per-user 2 --serve-args "$M" --tag mns128 --no-accuracy
.venv/bin/python sweep.py --phase p6 --workloads mtbench --concurrency 64 --prompts-per-user 2 --serve-args "$M" --tag mns128 --no-accuracy
.venv/bin/python replicas.py --layouts TP1x6 TP2x3 TP2x2 TP2x2_TP1x2 TP4_TP1x2 TP4_TP2 --concurrency 64 128 --serve-args "$M" --tag mns128
./sync_results.sh
echo "QUEUE DONE $(date +%T)"
