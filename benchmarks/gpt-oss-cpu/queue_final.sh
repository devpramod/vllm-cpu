#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python sweep.py --phase p5
.venv/bin/python sweep.py --phase p6
echo "QUEUE DONE $(date +%T)"
