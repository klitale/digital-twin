#!/usr/bin/env bash
# Control the twin unit on a host. Usage: deploy/service.sh <host-name> <start|stop|restart|disable|enable>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"; ACTION="${2:-}"
case "$ACTION" in start|stop|restart|disable|enable) ;; *) echo "usage: $0 <host-name> <start|stop|restart|disable|enable>" >&2; exit 2 ;; esac
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
[ "$ROLE" = bot ] || [ "$ROLE" = staging ] || { echo "$NAME has no service (role $ROLE)" >&2; exit 2; }
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT")
[ "$IDENT" != "-" ] && SSH+=(-i "${IDENT/#\~/$HOME}")
case "$ACTION" in
  disable) CMD="systemctl disable --now twin" ;;
  enable) CMD="systemctl enable --now twin" ;;
  *) CMD="systemctl $ACTION twin" ;;
esac
"${SSH[@]}" "$TARGET" "$CMD; sleep 1; systemctl --no-pager --lines=0 status twin | head -3 || true"
