#!/bin/bash
# FALLBACK -- jobs now commit and push their own diagnostics when they end (_latest_log.sh). Use this if a compute node
# could not reach GitHub. Refresh the multi-source status file and push everything under correction/diagnostics/ (job logs, the status
# file, diagnostics) to GitHub, so it can be read from anywhere with `git pull`.
#
#   bash /work/GRDVULN/tp_qa/correction/scripts/push_diagnostics.sh            # after any step
#   bash /work/GRDVULN/tp_qa/correction/scripts/push_diagnostics.sh "01f done"  # with a note in the commit message
#
# Run it on a login node. Only touches correction/diagnostics/ -- never data, models, or code. The login shell is
# tcsh, so call it with `bash`.
ROOT=/work/GRDVULN/tp_qa
cd "$ROOT" || exit 1
[ -f "$ROOT/correction/.venv/bin/activate" ] && source "$ROOT/correction/.venv/bin/activate"
python "$ROOT/correction/scripts/multi_status.py" || echo "WARNING: multi_status.py failed; pushing the logs anyway"
git add correction/diagnostics
if git diff --cached --quiet; then
    echo "nothing new under correction/diagnostics/"
else
    git commit -q -m "diagnostics: ${1:-multi-source status} ($(date '+%Y-%m-%d %H:%M'))"
fi
git pull --rebase --autostash -q || { echo "pull --rebase failed; resolve, then rerun"; exit 1; }
git push -q && echo "pushed: $(git log -1 --format='%h %s')"
