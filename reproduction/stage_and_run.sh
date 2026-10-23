#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  stage_and_run.sh [--repo-root PATH] [--staging-parent PATH] -- COMMAND [ARGS...]
  stage_and_run.sh [--repo-root PATH] [--staging-parent PATH] --stage-only

Create a timestamped repository copy for reproducible research runs. When COMMAND
is provided, execute it from inside the staged copy.
USAGE
}

repo_root=""
staging_parent=""
stage_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      repo_root="${2:?--repo-root requires a path}"
      shift 2
      ;;
    --staging-parent)
      staging_parent="${2:?--staging-parent requires a path}"
      shift 2
      ;;
    --stage-only)
      stage_only=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$stage_only" -eq 0 && $# -eq 0 ]]; then
  echo "Missing command. Use --stage-only to only create a copy." >&2
  usage >&2
  exit 2
fi

if [[ -n "${RESEARCH_REPRO_STAGED_DIR:-}" ]]; then
  if [[ "$stage_only" -eq 1 ]]; then
    printf '%s\n' "$RESEARCH_REPRO_STAGED_DIR"
    exit 0
  fi
  exec "$@"
fi

if [[ -z "$repo_root" ]]; then
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi
repo_root="$(cd "$repo_root" && pwd)"

if [[ -z "$staging_parent" ]]; then
  staging_parent="$repo_root/staging"
fi
mkdir -p "$staging_parent"

now="$(date '+%Y%m%d_%H%M%S')"
salt="$(openssl rand -hex 3 2>/dev/null || date '+%N')"
commitid="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || echo nogit)"
stagedir="$staging_parent/${now}-${salt}-${commitid}-code"

mkdir -p "$stagedir"

echo "Staging repository:"
echo "  source: $repo_root"
echo "  target: $stagedir"

rsync -rlt --no-perms --no-owner --no-group \
  --exclude '/staging' \
  --exclude '/out' \
  --exclude '/outputs' \
  --exclude '/eval_outputs' \
  --exclude '/slurm_outputs' \
  --exclude '/wandb' \
  --exclude '/.cache' \
  --exclude '/__pycache__' \
  --exclude '/.venv' \
  --exclude '/venv' \
  "$repo_root/" "$stagedir/"

metadata="$stagedir/REPRODUCIBILITY_METADATA.json"
python3 - "$repo_root" "$stagedir" "$commitid" "$*" > "$metadata" <<'PY'
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

repo_root, stagedir, commitid, command = sys.argv[1:5]

def run(args):
    try:
        return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

data = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_repo": repo_root,
    "staged_repo": stagedir,
    "git_commit_short": commitid,
    "git_commit_full": run(["git", "rev-parse", "HEAD"]),
    "git_status_short": run(["git", "status", "--short"]),
    "command": command,
    "hostname": platform.node(),
    "cwd": os.getcwd(),
}
print(json.dumps(data, indent=2, sort_keys=True))
PY

echo "Staging complete: $stagedir"

if [[ "$stage_only" -eq 1 ]]; then
  printf '%s\n' "$stagedir"
  exit 0
fi

echo "Running command from staged repository:"
printf '  %q' "$@"
printf '\n'

cd "$stagedir"
export RESEARCH_REPRO_STAGED_DIR="$stagedir"
export RESEARCH_REPRO_SOURCE_REPO="$repo_root"
exec "$@"
