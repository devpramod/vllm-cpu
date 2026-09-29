#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python ocr/replicas_ocr.py > logs/ocr_replicas.log 2>&1
