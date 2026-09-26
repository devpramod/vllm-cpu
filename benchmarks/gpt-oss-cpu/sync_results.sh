#!/bin/bash
# Mirror the bench harness and results into the vllm-cpu repo
# (branch gpt-oss-bench) and commit + push whenever something changed.
# Usage: sync_results.sh [interval_s]   (no interval = sync once)
set -u
SRC=/home/sysadmin/pramod/bench/
REPO=/home/sysadmin/pramod/vllm-cpu
DEST=$REPO/benchmarks/gpt-oss-cpu/
export GIT_AUTHOR_NAME=devpramod GIT_AUTHOR_EMAIL=pramod.pai@intel.com
export GIT_COMMITTER_NAME=devpramod GIT_COMMITTER_EMAIL=pramod.pai@intel.com

sync_once() {
    rsync -a --delete \
        --exclude .venv/ --exclude models/ --exclude __pycache__/ \
        --exclude 'data/*' --exclude 'trace/' --exclude '*.pid' --exclude logs/sync.log \
        "$SRC" "$DEST"
    cd "$REPO" || return
    [ "$(git rev-parse --abbrev-ref HEAD)" = gpt-oss-bench ] || {
        echo "$(date +%T) not on gpt-oss-bench, skipping"; return; }
    git add -A benchmarks/gpt-oss-cpu
    if git diff --cached --quiet; then
        return
    fi
    n=$(git diff --cached --name-only | wc -l)
    git commit -q -m "gpt-oss-20b CPU bench: sync results ($n files)" \
        -m "Co-authored-by: Cursor Agent <cursoragent@cursor.com>" \
        -m "Signed-off-by: devpramod <pramod.pai@intel.com>"
    git push -q origin gpt-oss-bench && echo "$(date +%T) pushed $n files"
}

sync_once
if [ $# -ge 1 ]; then
    while sleep "$1"; do sync_once; done
fi
