#!/usr/bin/env bash
# Read-only service check for a host from deploy/hosts.yaml: unit state, memory,
# last log lines. Usage: deploy/status.sh <host-name> [lines]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"; LINES="${2:-25}"
[ -n "$NAME" ] || { echo "usage: $0 <host-name> [lines]" >&2; exit 2; }
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
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT")
[ "$IDENT" != "-" ] && SSH+=(-i "${IDENT/#\~/$HOME}")
if [ "$ROLE" = jobs ]; then
  "${SSH[@]}" "$TARGET" "cd $DIR && .venv/bin/twin version && ls -la data/processed data/chroma 2>/dev/null | head -20"
  exit 0
fi
"${SSH[@]}" "$TARGET" "
systemctl --no-pager --lines=0 status twin | head -5
echo memory=\$(systemctl show twin -p MemoryCurrent --value | awk '{printf \"%.0f MB\", \$1/1024/1024}')
pid=\$(systemctl show twin -p MainPID --value); [ \"\$pid\" != 0 ] && echo rss=\$(awk '/VmRSS/{print \$2/1024 \" MB\"}' /proc/\$pid/status)
df -h $DIR | tail -1 | awk '{print \"disk_free=\" \$4}'
echo telegram_ipv4=\$(curl -4 -sS -m 8 -o /dev/null -w '%{http_code}' https://api.telegram.org/ 2>&1 | tail -c 60)
echo telegram_ipv6=\$(curl -6 -sS -m 8 -o /dev/null -w '%{http_code}' https://api.telegram.org/ 2>&1 | tail -c 60)
echo '--- journal (last $LINES) ---'
journalctl -u twin --no-pager -n $LINES -o cat
"
