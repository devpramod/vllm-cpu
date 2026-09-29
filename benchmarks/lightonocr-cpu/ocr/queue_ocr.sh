#!/bin/bash
# Socket-0 and socket-1 lanes in parallel, then configs needing both sockets.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY ocr/sweep_ocr.py --lane s0 > logs/ocr_lane_s0.log 2>&1 &
A=$!
$PY ocr/sweep_ocr.py --lane s1 > logs/ocr_lane_s1.log 2>&1 &
B=$!
wait $A $B
$PY ocr/sweep_ocr.py --lane all > logs/ocr_lane_all.log 2>&1
[ -x ocr/queue_ocr_replicas.sh ] && ./ocr/queue_ocr_replicas.sh
echo "OCR QUEUE DONE $(date +%T)"
