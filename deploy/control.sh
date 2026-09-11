#!/usr/bin/env bash
# Run `twin control ...` or `twin poke ...` on a bot host without Telegram. The service is
# stopped first (a running bot would overwrite the state the command changes), the data
# is handed back to the service user, and the service is started again.
# Usage: deploy/control.sh <host-name> control <command...>
#        deploy/control.sh <host-name> poke <user_id> [opener|followup]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"; SUB="${2:-}"
case "$SUB" in control|poke) ;; *) echo "usage: $0 <host-name> control <command...> | poke <user_id> [kind]" >&2; exit 2 ;; esac
shift 2
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
ARGS="$(printf ' %q' "$@")"
"${SSH[@]}" "$TARGET" "systemctl stop twin; cd $DIR && $DIR/.venv/bin/twin $SUB$ARGS; rc=\$?; chown -R twin:twin $DIR/data; systemctl start twin; sleep 2; systemctl --no-pager --lines=0 status twin | head -3; exit \$rc"
