#!/usr/bin/env bash
# Stop hook: after each Claude turn, commit whatever changed and push it, so
# work driven from the phone (Remote Control) can be pulled on another machine.
#
# Only acts on claude/* branches -- on master or any other branch it does
# nothing, so ordinary sessions are never auto-committed. Always exits 0: a
# failed push must not block Claude, and the commit stays local until the next
# turn's push picks it up.

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

branch=$(git symbolic-ref --short -q HEAD) || exit 0   # detached HEAD
case "$branch" in
  claude/*) ;;
  *) exit 0 ;;
esac

# Never commit into the middle of a merge or rebase.
gitdir=$(git rev-parse --git-dir)
if [ -e "$gitdir/MERGE_HEAD" ] || [ -d "$gitdir/rebase-merge" ] || [ -d "$gitdir/rebase-apply" ]; then
  echo "auto-push: merge/rebase in progress, skipping" >&2
  exit 0
fi

git add -A

# Size guard. GitHub rejects files over 100 MB, and this repo has already had
# 1.2 GB of imagery committed by accident once. Anything staged over the limit
# is unstaged and reported instead of committed -- if it belongs in git, commit
# it by hand; if not, add it to .gitignore.
limit=$((50 * 1024 * 1024))
git diff --cached --name-only --diff-filter=AM -z | while IFS= read -r -d '' f; do
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt "$limit" ]; then
    git reset -q -- "$f"
    echo "auto-push: skipped $f (over 50 MB)" >&2
  fi
done

if ! git diff --cached --quiet; then
  git commit -q -m "auto: claude session $(date '+%Y-%m-%d %H:%M')" || exit 0
fi

# Push if anything is unpushed -- this turn's commit, or one a previous turn
# failed to push. No upstream yet means the branch has never been pushed.
ahead=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo new)
if [ "$ahead" != "0" ]; then
  git push -q -u origin HEAD 2>&1 >&2 || echo "auto-push: push failed, will retry next turn" >&2
fi

exit 0
