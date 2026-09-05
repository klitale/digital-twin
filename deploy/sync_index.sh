#!/usr/bin/env bash
# Copy the Chroma index (with index_manifest.json), the style profile and the dataset
# manifest from this machine to a host; --processed adds pairs.jsonl / holdout.jsonl for
# the jobs host. Usage: deploy/sync_index.sh <host-name> [--processed] [--restart]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"; shift || true
[ -n "$NAME" ] || { echo "usage: $0 <host-name> [--processed] [--restart]" >&2; exit 2; }
PROCESSED=0; RESTART=0
for arg in "$@"; do
  case "$arg" in --processed) PROCESSED=1 ;; --restart) RESTART=1 ;; *) echo "unknown flag $arg" >&2; exit 2 ;; esac
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
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT")
[ "$IDENT" != "-" ] && SSH+=(-i "${IDENT/#\~/$HOME}")
RSYNC_SSH="${SSH[*]}"
[ -f "$HERE/data/chroma/index_manifest.json" ] || { echo "no local index (run twin index)" >&2; exit 1; }

echo "[sync] index + style profile -> $NAME:$DIR/data"
"${SSH[@]}" "$TARGET" "mkdir -p $DIR/data/chroma $DIR/data/processed"
rsync -az --delete -e "$RSYNC_SSH" "$HERE/data/chroma/" "$TARGET:$DIR/data/chroma/"
rsync -az -e "$RSYNC_SSH" "$HERE/data/processed/style_profile.md" "$HERE/data/processed/dataset_manifest.json" "$TARGET:$DIR/data/processed/"
if [ "$PROCESSED" = 1 ]; then
  echo "[sync] pairs + holdout -> $NAME"
  rsync -az -e "$RSYNC_SSH" "$HERE/data/processed/pairs.jsonl" "$HERE/data/processed/holdout.jsonl" "$TARGET:$DIR/data/processed/"
fi
if [ "$ROLE" = bot ] || [ "$ROLE" = staging ]; then
  "${SSH[@]}" "$TARGET" "chown -R twin:twin $DIR/data 2>/dev/null || true"
  [ "$RESTART" = 1 ] && "${SSH[@]}" "$TARGET" "systemctl restart twin && sleep 2 && systemctl --no-pager --lines=0 status twin | head -3"
fi
echo "[sync] done"
