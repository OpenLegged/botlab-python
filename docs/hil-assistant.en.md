# BotLab HIL device-link deployment runbook (for AI assistants)

This file is the **target document** of the BotLab web "Device Gateway" panel prompt: the panel
prompt only needs a one-sentence pointer here. The reader is the AI assistant the user delegates
to. Following this file end to end brings up the "browser ↔ gateway ↔ device agent" chain.

**Goal**: get the browser's "Device Gateway" panel connected to the gateway, with model inference
dispatched to the target device for execution.

## 0. Confirm the scenario and token choice before deploying

Before deploying, ask together about the deployment scenario and whether the user wants to
provide a custom connection token or have the AI generate a random one. Use information already
explicit in the conversation without asking again. Wait for answers to any unresolved choices;
until then, run read-only checks only. The deployment scenarios are:

- **A. Local trial (no board)**: gateway + agent on the same computer; the browser connects to
  `ws://127.0.0.1:50060`.
- **B. Real Linux board** (board HIL requires Python ≥3.11; if the system Python is older, ship a
  standalone CPython — step 4) **connected to the user's computer over Ethernet/LAN and reachable
  over SSH**: gateway + agent both run on the board's loopback; a single `ssh -L` tunnel on the
  user's computer surfaces the gateway on the local loopback, and the browser still connects to
  `ws://127.0.0.1:50060`. Board installs are **always offline** (step 4); the board is never
  required to reach the internet. Only boards that cannot be reached by direct SSH need the
  domain-server variant (see `hil.md`, "Gateway targets").

Follow the user's token choice:

- **Custom**: use the non-empty token the user provides. If it is missing or empty, ask and
  wait; do not switch to random generation on their behalf.
- **Random**: after the user chooses this option, generate the token once with a cryptographically
  secure generator such as `secrets.token_urlsafe(32)`.
- Use the same selected token for the gateway's `BOTLAB_HIL_GATEWAY_TOKEN`, the Agent's
  `BOTLAB_HIL_TOKEN`, and the browser connection. Every `<token>` below refers to this value.
  Pass it explicitly at startup instead of generating another one.

## 1. Component check (read-only)

- `which botlab-hil-gateway botlab-agent botlab`; if present, read the script shebang to locate
  the owning Python environment (`~/.local/share/uv/tools/<name>/bin/python` = a uv tool install),
  and query the version the matching way (`uv tool list` / `python3 -m pip show botlab-python`).
- If the machine has a checkout of this repository: `git fetch origin main`, then
  `git rev-list --count HEAD..origin/main` — update only if the count is > 0.
- Probe the tool environment for `websockets` / `numpy` / `onnxruntime` via
  `importlib.util.find_spec`; install whichever extras covers what is missing.

## 2. Install / update

- Source: PyPI `botlab-python>=0.4.1`; fall back to the git source
  `git+https://github.com/OpenLegged/botlab-python.git@main` only when an unreleased fix is needed.
- ⚠️ **git-source builds must be prefixed with `BOTLAB_ALLOW_MISSING_UI=1`**: the UI snapshot
  (`src/botlab/ui`) is a generated artifact that never enters git; without the prefix the build
  hook refuses to produce a package. HIL never touches the UI — this is a supported mode.
- A local checkout that contains `src/botlab/ui` can be installed from that directory directly
  (same commit, no toggle, keeps the UI — strictly better).
- Extras: `[gateway]` = websockets (gateway machine); `[target]` = numpy + onnxruntime +
  websockets (required for on-device inference; without it the agent registers but cannot run
  models); `[runtime]` = local MuJoCo simulation, not needed for HIL.
- **Boards and other internet-isolated machines install offline, always**: the assistant's machine
  builds this package's wheel, downloads dependency wheels tagged for the target, scps them over,
  then runs `pip install --no-index --find-links` (full commands in step 4). No installing over an
  improvised shared network or borrowed Wi-Fi.
- **HIL hard floor: Python ≥3.11**: `websockets>=17.1` declares Requires-Python ≥3.11, and the
  dial code (`hil/wsstream.py`) passes the `legacy=True` flag that only exists in 17.1 — dropping
  to 16.x fails immediately with TypeError. If the board's system Python is older, use a standalone
  CPython (step 4). Offline wheel cp/ABI tags must match the board venv's actual Python (pip
  silently ignores a cp310 wheel inside a 3.12 venv and reports `from versions: none`).
- With a restricted system pip, create a venv first; on Windows use `py -3 -m pip`; in a uv tool
  environment use `uv tool install --reinstall "botlab-python[...] @ <source>"`.

## 3. Scenario A commands (local trial)

```bash
# Before starting, confirm the port is free: lsof -nP -iTCP:50060
# Terminal 1 (use the token selected in step 0; --print-token prints it for handoff)
BOTLAB_HIL_GATEWAY_TOKEN=<token> botlab-hil-gateway --host 127.0.0.1 --port 50060 --print-token
# Terminal 2 (omit BOTLAB_HIL_TOKEN to type it via getpass interactively, keeping it out of shell history)
BOTLAB_HIL_TOKEN=<token> botlab-agent connect --gateway ws://127.0.0.1:50060 --device my-laptop
```

Keep both processes in the foreground. Note: the `registered` log line prints **before** the
gateway validates the token — a wrong token still shows registered first, then
`gateway link dropped`. Success criteria: the link stays up after registered (no drop) and
`lsof -iTCP:50060` shows ESTABLISHED; the final arbiter is the browser device window's Connect
succeeding. The assistant may run these on the user's behalf (as session background tasks), but
must tell the user the processes stop when the session ends, and must hand over the two terminal
commands to restart them.

## 4. Scenario B commands (board on the LAN + SSH tunnel)

Neither `ws://` leg ever leaves its own host's loopback (browser dials the user's loopback, agent
dials the board loopback); traffic over the cable is encrypted by SSH, so no domain, certificate,
or `wss://` is needed.

**Do the hardware match first**: once SSH works, identify the board —
`cat /proc/device-tree/model`, `. /etc/os-release && echo "$PRETTY_NAME"` — then check the
repository's [`skills/`](../skills/README.md) directory (read each `skills/*/SKILL.md`
frontmatter). On a hit, use that skill as this section's board-side supplement (default
networking, system quirks, and tested versions live there — load only the matching one); on a
miss, follow this section's generic commands, and once things work you are welcome to contribute
one back per `skills/README.md`.

```bash
# Assistant machine (internet + a checkout of this repo): build this package's wheel + download dependency wheels for the board's tags
# --python-version/--abi must equal the Python the board venv will use (example: 3.12/aarch64)
uv build --wheel   # the local checkout contains src/botlab/ui, so no BOTLAB_ALLOW_MISSING_UI needed
python3 -m pip download -d botlab-pkgs --no-cache-dir --only-binary=:all: \
  --python-version 3.12 --abi cp312 --implementation cp \
  --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 \
  --platform manylinux_2_27_aarch64 --platform manylinux_2_28_aarch64 \
  numpy "onnxruntime>=1.18" "websockets>=17.1" "pyyaml>=6.0"
cp dist/botlab_python-*.whl botlab-pkgs/
scp -r botlab-pkgs <user>@<board>:~/

# Board: create the venv — directly if system Python >= 3.11
python3 -m venv ~/botlab-venv
# If system Python < 3.11 (websockets>=17.1 needs >= 3.11), or venv complains about missing ensurepip:
# the assistant machine downloads cpython-3.12.*-<board arch>-install_only.tar.gz from the
# astral-sh/python-build-standalone releases (bundles ensurepip), scps it to the board, then:
mkdir -p ~/botlab-python && tar xzf cpython-*.tar.gz -C ~/botlab-python --strip-components=1
~/botlab-python/bin/python3 -m venv ~/botlab-venv

# Board: offline install (the board never needs internet access)
~/botlab-venv/bin/pip install --no-index --find-links ~/botlab-pkgs "botlab-python[gateway,target]"

# Board terminal 1: gateway (binds the board loopback only; use the token selected in step 0)
BOTLAB_HIL_GATEWAY_TOKEN=<token> ~/botlab-venv/bin/botlab-hil-gateway --host 127.0.0.1 --port 50060 --print-token

# Board terminal 2: agent (the gateway lives on the board; dial the board loopback)
BOTLAB_HIL_TOKEN=<token> ~/botlab-venv/bin/botlab-agent connect \
  --gateway ws://127.0.0.1:50060 --device <device-id>

# Local terminal 3: SSH tunnel (keep it running; if it drops, rerunning the same command restores it)
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 50060:127.0.0.1:50060 <user>@<board>
```

Success criteria: board terminal 2 shows `BotLab HIL agent: registered as <id> (onnxruntime)`
**with no** `gateway link dropped` after it (registered prints before token validation, so a wrong
token shows it too); the user's `lsof -nP -iTCP:50060` shows ssh listening on `127.0.0.1:50060`;
the final arbiter is the browser device window's Connect succeeding.

## 5. Security rules

Tokens are relayed only inside the conversation with the user — never written to files, logs, or
commits; `ws://` stays on each host's own loopback (the user's loopback in scenario A, the board's
loopback in scenario B), and scenario B's cross-machine traffic rides only the SSH tunnel; if a
remote `wss://` gateway is required, a strong token is still mandatory, with its source
following the user's choice in step 0.

## 6. Handoff

Once the chain is up, hand the user the three-item list — **gateway endpoint / device ID /
token** — and remind them: in the browser project tree, open the "Hardware" section and add a
device (fill in the endpoint and device ID; the name is theirs to choose), then click "Connect…"
on the device row and enter the token — the token only ever exists in the connect dialog's memory
and is never stored.

## 7. Error discipline

Paste the full error, analyze the root cause, then act — never blindly switch ports or reinstall.
Known-trap quick reference:

- `src/botlab/ui is missing` → the git source lacks the UI snapshot; prefix the install command
  with `BOTLAB_ALLOW_MISSING_UI=1`;
- Port-related errors → check occupancy with `lsof -iTCP:<port>` before deciding; don't just
  switch ports;
- The agent registers but inference errors → missing `[target]` extras (numpy/onnxruntime);
  install them;
- Offline install reports `from versions: none` → the wheel's cp/ABI tags don't match the board
  venv's Python; re-run `pip download` for the venv's actual version;
- `websockets>=17.1` won't install (17.x Requires-Python ≥3.11) → the board Python is too old;
  build the venv on a standalone CPython; do not downgrade to 16.x (the dial passes `legacy=True`
  and will TypeError);
- The browser can't connect although the board agent is registered → the local `ssh -L` tunnel
  dropped or local port 50060 is occupied; find out with `lsof -nP -iTCP:50060`, then rerun the
  tunnel command;
- Board-specific errors (system clock drift, remote `pkill` self-matching, system Python missing
  components, …) → check `skills/<board>/SKILL.md`'s tested-quirks list first, then fall back to
  the generic path.

Architecture and protocol background: [hil.md](hil.md) (Chinese).
