#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python replicas.py
echo "QUEUE DONE $(date +%T)"
