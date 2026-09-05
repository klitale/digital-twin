#!/usr/bin/env bash
# Idempotent install on the server itself. Usage: deploy/install.sh <bot|staging|jobs>
#
# bot / staging (root): /opt/twin, system user `twin`, uv, `uv sync --no-dev`,
#   systemd unit `twin` from deploy/roles/<role>.service, restart if .env exists.
# jobs (any user, no sudo needed): ~/twin (or $TWIN_DIR), uv, `uv sync --no-dev`; no
#   service - eval, index builds and reports run on demand through the `twin` CLI.
#
# Code arrives either by `git pull` (when the checkout has .git) or by rsync from a
# workstation (deploy/push.sh); this script never fetches secrets.
set -euo pipefail
ROLE="${1:-}"
case "$ROLE" in
  bot|staging) DIR="${TWIN_DIR:-/opt/twin}"; SERVICE_USER="twin" ;;
  jobs) DIR="${TWIN_DIR:-$HOME/twin}"; SERVICE_USER="$(id -un)" ;;
  *) echo "usage: $0 <bot|staging|jobs>" >&2; exit 2 ;;
esac
log() { printf '[install:%s] %s\n' "$ROLE" "$*"; }

# --- uv -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then
    log "installing uv to /usr/local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
  else
    log "installing uv to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
log "uv: $(uv --version)"

# --- code -----------------------------------------------------------------------------
[ -f "$DIR/pyproject.toml" ] || { log "no code in $DIR (run deploy/push.sh or git clone first)"; exit 1; }
if [ -d "$DIR/.git" ]; then
  log "git pull --ff-only"
  git -C "$DIR" pull --ff-only
fi
mkdir -p "$DIR/data/state" "$DIR/data/chroma" "$DIR/data/processed"

# --- service user (root roles only) ---------------------------------------------------
if [ "$ROLE" != "jobs" ]; then
  [ "$(id -u)" = 0 ] || { log "bot/staging install needs root"; exit 1; }
  if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    log "creating system user $SERVICE_USER"
    useradd --system --home-dir "$DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
  fi
  chown -R "$SERVICE_USER:$SERVICE_USER" "$DIR"
  [ -f "$DIR/.env" ] && chmod 600 "$DIR/.env"
fi

# --- environment ----------------------------------------------------------------------
sync_cmd="cd '$DIR' && UV_CACHE_DIR='$DIR/.cache/uv' uv sync --no-dev --frozen && uv cache clean --quiet"
log "uv sync --no-dev (as $SERVICE_USER)"
if [ "$ROLE" = "jobs" ]; then
  bash -c "$sync_cmd"
else
  su -s /bin/bash "$SERVICE_USER" -c "export PATH=/usr/local/bin:\$PATH; $sync_cmd"
fi
log "twin: $("$DIR/.venv/bin/twin" version)"

# --- systemd --------------------------------------------------------------------------
if [ "$ROLE" = "jobs" ]; then
  log "jobs host ready: cd $DIR && .venv/bin/twin --help"
  exit 0
fi
install -m 644 "$DIR/deploy/roles/$ROLE.service" /etc/systemd/system/twin.service
systemctl daemon-reload
systemctl enable twin >/dev/null
if [ -f "$DIR/.env" ]; then
  log "restarting twin"
  systemctl restart twin
  sleep 3
  systemctl --no-pager --lines=0 status twin | head -3
else
  log "no $DIR/.env yet: unit enabled but not started (deploy/push.sh --env)"
fi
