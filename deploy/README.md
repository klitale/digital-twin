# Deployment runbook

Three small Ubuntu VPS, no Docker, no webhook, no open ports: the bot uses long polling.
Hosts and roles live in `deploy/hosts.yaml` (gitignored; template in
`hosts.example.yaml`).

| Role | What runs | Where |
|---|---|---|
| `bot` | `twin run` under systemd (`twin.service`), Chroma index, `data/state` | `/opt/twin`, system user `twin` |
| `staging` | a second bot instance with `DRY_RUN=true` forced by the unit. **Needs its own bot token and its own Telegram account**: one token allows one long-polling consumer and one Business account allows one chatbot. Installed on srv-a but disabled until a second bot exists | `/opt/twin` |
| `jobs` | index builds, evaluation runs, reports on demand through the CLI; no service, no sudo | `~/twin` |

This workload does not need three machines; the split is for isolation and learning.

## First install

```bash
deploy/inventory.sh                    # read-only facts, then assign roles in hosts.yaml
deploy/push.sh srv-b --env             # rsync code + .env, install uv, venv, unit; restarts twin
deploy/sync_index.sh srv-b --restart   # Chroma index, style profile, dataset manifest
deploy/push.sh srv-a --env             # staging (dry run forced)
deploy/push.sh srv-c --env             # jobs host: environment only
deploy/sync_index.sh srv-c --processed # pairs/holdout for evaluation
```

`push.sh` rewrites `DATA_DIR` to the install dir and drops `RAW_EXPORT_PATH`; the
export never leaves the workstation. Until the repository is on GitHub, `push.sh` is
the code path; with a checkout that has `.git`, `install.sh` does `git pull --ff-only`
instead, so the target state is `git pull && uv sync && systemctl restart twin`.

The bot starts in whatever `DRY_RUN` says in `.env` (`true` by default). Flip it with
`/twin dryrun off` in the bot's direct chat or in `.env` + restart once you connected the
account (Telegram → Settings → Business → Chatbots; restrict *Selected chats* to the two
allowed users).

## Update

```bash
deploy/push.sh srv-b            # code only, runs install.sh (uv sync + restart)
deploy/push.sh srv-b --env      # when .env changed
deploy/sync_index.sh srv-b --restart   # after `twin index --rebuild` or a new style profile
```

## Logs and state

```bash
ssh <host> journalctl -u twin -f          # structured JSON lines, one record per reply
ssh <host> journalctl -u twin --since -1h
ssh <host> cat /opt/twin/data/state/connection.json   # business connection
ssh <host> cat /opt/twin/data/state/bot_state.json    # switches, pauses, sent ids
```

Memory: `systemctl show twin -p MemoryCurrent`; measured at the first deploy (Chroma
index of 6.2k pairs loaded, embeddings via the gateway): about 230 MB.

Service control from the workstation: `deploy/service.sh <host> start|stop|restart|disable|enable`;
`deploy/status.sh <host> [lines]` shows the unit, memory, Telegram reachability and the log tail.

## Rollback

Code: `deploy/push.sh <host>` from a checkout of the previous commit (`git stash` /
`git checkout <sha>` locally, push, checkout back). State and index are untouched by
code pushes. Index: rerun `deploy/sync_index.sh` from a workstation that has the
previous `data/chroma`. Emergency stop: `/twin off` in the bot chat or
`ssh <host> systemctl stop twin`.
