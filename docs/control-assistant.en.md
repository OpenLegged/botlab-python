# BotLab Control CLI deployment runbook (for AI assistants)

> 中文版：[control-assistant.md](control-assistant.md)。两份内容同步维护，改其一必须改另一份。

This file is the **target document** of the BotLab web "CLI Control" panel prompt: the panel prompt
only needs one sentence pointing here. The reader is the AI assistant the user delegates to.
Following this file end to end brings up the "CLI → generic middleware → browser page" link.

**Goal**: connect the BotLab page the user already has open (editor or hosted page) to the local
generic middleware, then drive the robot simulation in the page via `botlab control` commands.

## 0. Link and prerequisites

- The user already has the browser page; the browser side **needs nothing installed**. The only
  local install is `botlab-python` (providing the `botlab` CLI and the generic middleware).
- The middleware is **generic**: one command, `botlab control serve`, with no project code inside.
  The project's capability declarations (capabilities), action tables, and control experience are
  **reported by the page at pairing time**, and the middleware is built from those descriptors —
  so **checking out the controlled project's source is never needed**.
- The middleware is a WS server the browser dials into, so it must run on the user's own machine
  (or a machine the user can reach).
- Confirm one thing with the user before acting: which address the BotLab page is opened from
  (this decides `--origin` in step 3).
- Project-bundled middleware (e.g. microduck's `python -m microduck_control serve`) is still
  supported: start it per that project's own instructions and the remaining steps are unchanged;
  this file defaults to the generic middleware.

## 1. Component check (read-only)

- `which botlab`; if present, read the script shebang to locate the owning Python environment and
  query the version the matching way (`python3 -m pip show botlab-python`).
- The version number cannot tell whether `control` is included; treat the existence of
  `botlab control --help` as the criterion.

## 2. Install / update

- Source: prefer PyPI `botlab-python>=0.4.1`; use `git+https://github.com/OpenLegged/botlab-python.git@main` only when you need the latest unreleased fixes from main.
- ⚠️ **Building from a git source requires the `BOTLAB_ALLOW_MISSING_UI=1` prefix**: the UI snapshot
  (`src/botlab/ui`) is a generated artifact kept out of git, otherwise the build hook refuses to
  produce a package. CLI control never touches the UI; this is an officially supported usage. PyPI
  installs are unaffected.
- If a local checkout contains `src/botlab/ui`, install straight from that directory instead (same
  commit, no flag needed, keeps the UI — strictly better).
- Install with the `serve` extra (the middleware needs aiohttp): `pip install "botlab-python[serve]"`;
  the control client itself is stdlib-only. Without the extra, `botlab control serve` fails with
  `missing_dependency` (exit 2) — install it as the message says.
- If the system pip is restricted, create a venv first; on Windows use `py -3 -m pip`.

## 3. Start the middleware and pair

```bash
botlab control serve --port 8776 --origin <this page's origin>

# Get the pairing info (websocket address + token):
botlab control connection --show-token --json
```

- The working directory sets the middleware's project anchor (discovery's `project_key`); it is
  unrelated to the page's project. Whichever directory you start it in is the directory you call
  the remaining commands from (or point `--project` at it).
- By default the middleware listens on 127.0.0.1 only and accepts only allow-listed origins.
  **A web page's origin is not 127.0.0.1**, so you must pass
  `--origin <scheme://host[:port] of the page>` (e.g. `--origin https://botlab.example.com`),
  otherwise the browser's WS connection is rejected.
- Each middleware instance accepts exactly one browser run; do not start two services against the
  same browser.
- Hand the websocket address and token from `connection --show-token` to the user **in the
  conversation**, and remind them to paste them into the browser's "CLI Control" panel and click
  "Connect and enable external control". Connecting only enables external commands; it does not
  move the robot.

## 4. Verify

```bash
botlab control state --json
```

The link is ready when `connected` and `external_enabled` are both `true`, `run_state` is `running`,
and `ready` is `true`. `ready=false` usually means the page is hidden or the simulation is paused —
keep the pairing; commands work again once it resumes.

## 5. Control rules (after takeover)

1. Before controlling, pull the project experience and capability table (both reported by the page):
   `botlab control experience --json` and `botlab control capabilities --json`;
   experience records measured parameters (e.g. the minimum speed that triggers walking) and
   pitfalls; capabilities do not.
2. Per action: read `state` → send the command → `wait --id` until `terminal:true` → check the
   physical feedback, then send the next one.
3. A failed tool-output parse does not mean the command did not run: check `state`'s
   `active_command_id` first; only resend after confirming it was not accepted.
4. Send parameters within the ranges declared by `capabilities`; action aliases come from the page's
   declaration and are not built into the generic CLI; timed moves are not precise navigation —
   judge actual effect by position and pose feedback.
5. A network timeout only means the outcome is unknown: first `wait` on the original request, and
   if needed explicitly `retry` the same request-id; use a new ID for new motion decisions.
6. Exit codes: `0` request accepted (including `pending`), `2` request/connection error, `3` task
   failed/cancelled/expired.

## 6. Security rules

Tokens are relayed in the user's conversation only — never written to files, logs, or commits; when
passed via `BOTLAB_CONTROL_TOKEN` or the hidden input prompt, the token never enters argv or shell
history. Keep the request-receipts directory (default `~/.local/state/botlab/control/receipts/`) at
0700/0600, do not copy receipts into project logs, and do not delete receipts to work around
conflicts.

## 7. Error discipline

Paste the full error message, analyze the root cause, then act — do not blindly switch ports or
reinstall. Quick reference of known pitfalls:

- browser panel cannot connect, middleware log shows origin rejected → missing `--origin`; add the page's origin per step 3;
- `state` shows `connected:false` → the user has not finished pairing in the panel; redo the step-3 handoff;
- `capabilities` / `experience` / command sending returns `no_project_paired` → the page has not paired and reported descriptors, or the page is too old to support dispatching; confirm the user is on a current page and reconnect;
- `serve` reports `missing_dependency` → missing serve extra; install aiohttp per step 2;
- after a middleware restart, all old commands are rejected → the instance identity (instance_id) has changed; this is expected. Re-pair; do not replay old session / episode / request IDs;
- CLI install reports `src/botlab/ui is missing` → the git source lacks the UI snapshot; prefix the install command with `BOTLAB_ALLOW_MISSING_UI=1`;
- port-related errors → check occupancy with `lsof -nP -iTCP:8776` first; do not just switch ports.

Protocol and hosting background: [control-protocol.md](control-protocol.md) and
[control-hosting.md](control-hosting.md) (in Chinese).
