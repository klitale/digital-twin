#!/usr/bin/env bash
# Read-only inventory of the hosts in deploy/hosts.yaml: OS, CPU, RAM, disk, load,
# listening ports, Python, docker, existing twin install. Prints one block per host and
# a summary table; never changes anything on the servers.
#
# Usage: deploy/inventory.sh [hosts.yaml]
set -euo pipefail
HOSTS_FILE="${1:-$(dirname "$0")/hosts.yaml}"
[ -f "$HOSTS_FILE" ] || { echo "hosts file not found: $HOSTS_FILE" >&2; exit 2; }

# host lines: name|target|port|identity   (target = user@host or alias)
hosts_lines() {
  uv run --quiet python - "$HOSTS_FILE" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for h in data.get("hosts", []):
    if h.get("ssh_alias"):
        target, port, ident = h["ssh_alias"], "", ""
    else:
        target = f"{h.get('user', 'root')}@{h['host']}"
        port, ident = str(h.get("port", "")), h.get("identity_file", "")
    print("|".join([h["name"], target, port, ident]))
PY
}

REMOTE='
set +e
echo "hostname=$(hostname)"
echo "os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "kernel=$(uname -r)"
echo "arch=$(uname -m)"
echo "vcpu=$(nproc)"
echo "ram_mb_total=$(free -m | awk "/^Mem:/{print \$2}")"
echo "ram_mb_available=$(free -m | awk "/^Mem:/{print \$7}")"
echo "swap_mb_total=$(free -m | awk "/^Swap:/{print \$2}")"
echo "disk_root=$(df -h / | awk "NR==2{print \$2\" total, \"\$4\" free (\"\$5\" used)\"}")"
echo "load=$(awk "{print \$1, \$2, \$3}" /proc/loadavg)"
echo "uptime=$(uptime -p 2>/dev/null)"
echo "python3=$(python3 --version 2>&1)"
echo "python3.11=$(command -v python3.11 >/dev/null && python3.11 --version 2>&1 || echo none)"
echo "uv=$(command -v uv >/dev/null && uv --version 2>&1 || echo none)"
echo "docker=$(command -v docker >/dev/null && docker --version 2>&1 | cut -d, -f1 || echo none)"
echo "containers_running=$(docker ps -q 2>/dev/null | wc -l | tr -d " ")"
echo "listening=$(ss -Hltn 2>/dev/null | awk "{print \$4}" | sed -E "s/.*:([0-9]+)$/\1/" | sort -un | tr "\n" " ")"
echo "ufw=$(ufw status 2>/dev/null | head -1 || echo unknown)"
echo "sudo=$(sudo -n true 2>/dev/null && echo yes || echo no)"
echo "twin_installed=$([ -d /opt/twin ] && echo yes || echo no)"
echo "twin_service=$(systemctl is-active twin 2>/dev/null || true)"
echo "top_mem=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk "NR>1 && NR<=4{printf \"%s:%dMB \", \$2, \$1/1024}")"
'

summary=()
while IFS='|' read -r name target port ident; do
  [ -n "$name" ] || continue
  opts=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
  [ -n "$port" ] && opts+=(-p "$port")
  [ -n "$ident" ] && opts+=(-i "${ident/#\~/$HOME}")
  echo "=================== $name ($target)"
  if out=$(ssh -n "${opts[@]}" "$target" "$REMOTE" 2>&1); then  # -n: never read the host list from stdin
    echo "$out"
    vcpu=$(grep '^vcpu=' <<<"$out" | cut -d= -f2)
    ram=$(grep '^ram_mb_available=' <<<"$out" | cut -d= -f2)
    ramt=$(grep '^ram_mb_total=' <<<"$out" | cut -d= -f2)
    disk=$(grep '^disk_root=' <<<"$out" | cut -d= -f2)
    load=$(grep '^load=' <<<"$out" | cut -d= -f2)
    py=$(grep '^python3=' <<<"$out" | cut -d= -f2)
    summary+=("$name|$vcpu vCPU|$ram/$ramt MB free|$disk|$load|$py")
  else
    echo "UNREACHABLE: $out"
    summary+=("$name|unreachable||||")
  fi
done < <(hosts_lines)

echo
echo "=================== summary"
printf '%-8s | %-7s | %-22s | %-32s | %-16s | %s\n' host cpu ram disk load python
for row in "${summary[@]}"; do
  IFS='|' read -r a b c d e f <<<"$row"
  printf '%-8s | %-7s | %-22s | %-32s | %-16s | %s\n' "$a" "$b" "$c" "$d" "$e" "$f"
done
