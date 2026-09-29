#!/bin/bash
# Mirror the bench harness and results into the vllm-cpu repo
# (branch gpt-oss-bench) and commit + push whenever something changed.
# Usage: sync_results.sh [interval_s]   (no interval = sync once)
set -u
SRC=/home/sysadmin/pramod/bench/
REPO=/home/sysadmin/pramod/vllm-cpu
DEST=$REPO/benchmarks/gpt-oss-cpu/
OCR_DEST=$REPO/benchmarks/lightonocr-cpu
export GIT_AUTHOR_NAME=devpramod GIT_AUTHOR_EMAIL=pramod.pai@intel.com
export GIT_COMMITTER_NAME=devpramod GIT_COMMITTER_EMAIL=pramod.pai@intel.com

sync_once() {
    rsync -a --delete \
        --exclude .venv/ --exclude models/ --exclude __pycache__/ \
        --exclude 'data/*' --exclude 'trace/' --exclude '*.pid' --exclude logs/sync.log \
        --exclude /ocr/ --exclude /results/ocr/ --exclude /results/ocr_size/ \
        "$SRC" "$DEST"
    mkdir -p "$OCR_DEST/results"
    rsync -a --delete --exclude data/ --exclude __pycache__/ "$SRC/ocr/" "$OCR_DEST/ocr/"
    rsync -a --delete "$SRC/results/ocr/" "$OCR_DEST/results/"
    rsync -a --delete "$SRC/patches/" "$OCR_DEST/patches/"
    cp "$SRC/.gitignore" "$SRC/sync_results.sh" "$OCR_DEST/"
    cd "$REPO" || return
    [ "$(git rev-parse --abbrev-ref HEAD)" = gpt-oss-bench ] || {
        echo "$(date +%T) not on gpt-oss-bench, skipping"; return; }
    git add -A benchmarks/gpt-oss-cpu benchmarks/lightonocr-cpu
    if git diff --cached --quiet; then
        return
    fi
    n=$(git diff --cached --name-only | wc -l)
    git commit -q -m "CPU bench (gpt-oss-20b, LightOnOCR-2-1B): sync results ($n files)" \
        -m "Co-authored-by: Cursor Agent <cursoragent@cursor.com>" \
        -m "Signed-off-by: devpramod <pramod.pai@intel.com>"
    git push -q origin gpt-oss-bench && echo "$(date +%T) pushed $n files"
}

sync_once
if [ $# -ge 1 ]; then
    while sleep "$1"; do sync_once; done
fi
