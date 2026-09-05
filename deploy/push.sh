#!/usr/bin/env bash
# Push the working tree to a host from deploy/hosts.yaml and run install.sh there.
# Usage: deploy/push.sh <host-name> [--env] [--no-install]
#   --env      also upload .env (DATA_DIR rewritten to the install dir, RAW_EXPORT_PATH
#              dropped, DRY_RUN=true forced for staging)
# Until the repo is on GitHub this is the deploy path; afterwards `git pull` in
# install.sh takes over and push.sh stays useful for .env updates.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"; shift || true
[ -n "$NAME" ] || { echo "usage: $0 <host-name> [--env] [--no-install]" >&2; exit 2; }
WITH_ENV=0; INSTALL=1
for arg in "$@"; do
  case "$arg" in --env) WITH_ENV=1 ;; --no-install) INSTALL=0 ;; *) echo "unknown flag $arg" >&2; exit 2 ;; esac
done

read -r TARGET PORT IDENT ROLE DIR < <(uv run --quiet --project "$HERE" python - "$HERE/deploy/hosts.yaml" "$NAME" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for h in data.get("hosts", []):
    if h["name"] == sys.argv[2]:
        target = h.get("ssh_alias") or f"{h.get('user', 'root')}@{h['host']}"
        role = h.get("role") or "-"
        default_dir = "/opt/twin" if role in ("bot", "staging") else "~/twin"
        print(target, h.get("port", 22), h.get("identity_file", "-"), role, h.get("install_dir", default_dir))
        break
else:
    sys.exit(f"host {sys.argv[2]} not in hosts.yaml")
PY
)
[ "$ROLE" != "-" ] || { echo "host $NAME has no role in hosts.yaml" >&2; exit 2; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT")
[ "$IDENT" != "-" ] && SSH+=(-i "${IDENT/#\~/$HOME}")
RSYNC_SSH="${SSH[*]}"

echo "[push] $NAME ($TARGET, role $ROLE) -> $DIR"
"${SSH[@]}" "$TARGET" "mkdir -p $DIR"
rsync -az --delete -e "$RSYNC_SSH" \
  --exclude '.git' --exclude '.venv' --exclude 'data' --exclude '.env' --exclude '.env.*' \
  --exclude '.cache' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude '.coverage' --exclude 'deploy/hosts.yaml' --exclude '.claude' \
  "$HERE/" "$TARGET:$DIR/"

if [ "$WITH_ENV" = 1 ]; then
  echo "[push] uploading .env (DATA_DIR -> $DIR/data)"
  tmp="$(mktemp)"
  grep -vE '^(DATA_DIR|RAW_EXPORT_PATH)=' "$HERE/.env" > "$tmp"
  printf 'DATA_DIR=%s/data\n' "$DIR" >> "$tmp"
  [ "$ROLE" = staging ] && printf 'DRY_RUN=true\n' >> "$tmp"
  rsync -az -e "$RSYNC_SSH" "$tmp" "$TARGET:$DIR/.env"
  rm -f "$tmp"
fi

if [ "$INSTALL" = 1 ]; then
  "${SSH[@]}" "$TARGET" "bash $DIR/deploy/install.sh $ROLE"
fi
