#!/bin/bash
cd "$(dirname "$0")/.."
HF=.venv/bin/hf
$HF download lightonai/LightOnOCR-2-1B --local-dir models/LightOnOCR-2-1B
$HF download naver-clova-ix/cord-v2 --repo-type dataset --local-dir ocr/data/cord-v2
$HF download opendatalab/OmniDocBench --repo-type dataset --local-dir ocr/data/OmniDocBench
$HF download allenai/olmOCR-bench --repo-type dataset --local-dir ocr/data/olmOCR-bench
echo "DOWNLOADS DONE $(date +%T)"
