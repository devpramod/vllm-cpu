#!/bin/bash
# Stop p2 once TP4 is done, redo the TP4 points that overlapped the premature
# p4 start, then run p4. Everything runs strictly one after another.
set -u
cd "$(dirname "$0")"
P2_PID=104459

wait_idle() {
    for _ in $(seq 120); do
        pgrep -f "bin/vllm (serve|bench)" >/dev/null || \
            curl -s -m 2 localhost:8100/health >/dev/null || return 0
        sleep 5
    done
    pkill -9 -f "bin/vllm (serve|bench)"
    sleep 10
}

while ! grep -q "start TP2_PP3" logs/p2.log; do
    kill -0 "$P2_PID" 2>/dev/null || break
    sleep 5
done
kill "$P2_PID" 2>/dev/null
while kill -0 "$P2_PID" 2>/dev/null; do sleep 5; done
pkill -f "bin/vllm serve"
wait_idle
echo "[$(date +%T)] p2 stopped before TP2_PP3" >> logs/p2.log

rm -f results/p2/TP4/rand_1k_1k_c{8,16,32}.json
.venv/bin/python sweep.py --phase p2 --configs TP4 >> logs/p2.log 2>&1
wait_idle

.venv/bin/python sweep.py --phase p4 > logs/p4.log 2>&1
