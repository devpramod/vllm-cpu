#!/bin/bash
cd "$(dirname "$0")"
while kill -0 505194 2>/dev/null; do sleep 30; done
.venv/bin/python replicas.py --concurrency 64 128
.venv/bin/python replicas.py --layouts TP4x1 --concurrency 64 128 --serve-args "--max-num-seqs 128" --tag mns128
echo "QUEUE DONE $(date +%T)"
