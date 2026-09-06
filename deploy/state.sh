#!/usr/bin/env bash
# Read-only dump of the bot's runtime state on a host from deploy/hosts.yaml:
# the business connection, the switches (including initiative), pauses and today's
# opener plan. Usage: deploy/state.sh <host-name>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-}"
[ -n "$NAME" ] || { echo "usage: $0 <host-name>" >&2; exit 2; }
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
STATE_JSON="$("${SSH[@]}" "$TARGET" "cat $DIR/data/state/bot_state.json 2>/dev/null || echo '{}'")"
export STATE_JSON
uv run --quiet --project "$HERE" python -c '
import json, os, time
state = json.loads(os.environ["STATE_JSON"] or "{}")
now = int(time.time())
print("enabled:", state.get("enabled"), "| mode:", state.get("mode") or "(from .env)",
      "| dry_run_override:", state.get("dry_run_override"))
print("features:", state.get("features") or "(all off)")
print("paused now:", {c: (t - now) // 60 for c, t in (state.get("paused_until") or {}).items() if t > now} or "none")
print("disabled chats:", {c: v for c, v in (state.get("chat_enabled") or {}).items() if not v} or "none")
print("opener plans:", state.get("opener_plan") or "none")
for label, key in (("last incoming", "last_incoming_ts"), ("last outgoing", "last_outgoing_ts"),
                   ("last bot reply", "last_bot_reply_ts"), ("last initiative", "last_initiative_ts")):
    values = state.get(key) or {}
    print(label + ":", {c: str((now - t) // 60) + " min ago" for c, t in values.items()} or "none")
'
