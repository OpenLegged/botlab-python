# Hardware-in-the-loop (HIL)

BotLab can run a project's ONNX models on a Linux target while the browser runs
MuJoCo, the signal graph, project Python and keyboard controls. Project `Model`
declarations are reused; neither a robot-specific network service nor a desktop
is required on the target.

The current execution type is **model-service**. It provides target preparation,
versioned model deployment and synchronous inference, not deployment of the
complete controller or a hard real-time scheduler. The feature is currently an
unreleased local development change. Install matching SDK and browser-runtime
builds; a published older wheel or cached editor may not contain these features.

## Prepare a target from a clean system

The built-in adapter supports Linux `aarch64` and `x86_64`, glibc 2.17 or later,
Python 3.10 or later, and an available SSH server. Compatible binary NumPy and
ONNX Runtime wheels must exist for the detected Python/glibc/architecture
combination. The Mac downloads these wheels; the board does not need Internet,
pip, virtualenv, BotLab, or project programs preinstalled.

Ubuntu alone is not a compatibility specification. Inspection reports the OS,
architecture, Python, glibc, storage, memory, network interfaces and available
network-management capability. The RDK X5 profile only adds factory-address
hints to discovery; it does not change the project protocol or inference backend.

In a project directory, start `botlab run` and open **Target manager** from the
compact HIL toolbar. The manager has three pages: **Target settings**, **Deployed
projects**, and **Network settings**. It manages configuration and deployed
projects; target connection is controlled from the main toolbar.

1. On **Target settings**, add a target and enter its name, hostname/address and
   SSH username. You can enter the SSH password to the right of the username,
   then click **Save target**. The password is remembered by the current
   `botlab run` host process and is not written to the target configuration or
   returned to the browser. Leaving it blank preserves a password already
   remembered for the same connection. Restarting `botlab run` clears this cache.
   For key-based login, set the identity-file path under **Advanced**. SSH is the
   default; Advanced also contains the port (22),
   installation root (`~/botlab`), backend (`onnxruntime`) and the existing-Agent
   TCP connection option. There is no board type to select: inspection detects
   the architecture and compatible environment.
2. Close the manager and click **Connect** in the main toolbar. For the first SSH
   connection, the connection dialog automatically checks the target using the
   saved session credentials or SSH key and displays its result. If no usable
   credentials are available or SSH authentication fails, a password dialog lets
   you enter a password and retry the current operation. A wrong password keeps
   the dialog open for another attempt; **Cancel** ends that operation. This
   prompt also applies to SSH management actions such as scanning projects and
   reading target network interfaces. Network failures and host-key errors are
   reported separately rather than requesting a password. For a new SSH host,
   verify the displayed fingerprint against a trusted source such as the board's
   console, then accept the verified key; inspection continues automatically.
   Existing trusted keys are also supported.
3. Review the check result. The dialog offers **Cancel** and **Connect**; when
   the Agent is missing, the latter becomes **Install and connect**. Installation
   uploads the Agent and matching Linux dependencies into the chosen BotLab root
   and connects afterward. A successful connection does not require changing
   network settings.
4. Choose **Run on Target** in the main toolbar. Its separate-operation menu
   also provides separate build, deploy, load, run and unload actions. After
   building and deploying to the connected target, **Load project** directly
   connects that exact build. **Run** in this menu starts the connected version
   without rebuilding, deploying or selecting another version.
   In the manager's **Deployed projects** page, choose a target on the left to
   automatically scan its projects. **Scan projects** refreshes the list;
   expanding a project shows each version's deployment time, short build ID,
   and loaded status. **Load version** selects an existing version of that
   project, or switches to a version of another project, on the connected target.
   Stop the running session before switching; selecting a version does not build
   or deploy. **Delete version** removes only that deployment; deleting its last
   version also removes the project. **Delete project** removes all its versions.
   Both deletion actions require confirmation. Stop and unload the loaded version
   before deleting it or its project; other inactive versions and projects can
   be deleted while a version remains loaded. Browsing a target does not change
   the main toolbar's selected target or active project. Connect the browsed
   target from the main toolbar before loading one of its versions.

Operation progress, completion and errors appear in the main **Output** terminal.
The manager does not repeat logs or completion banners. A failed connection shows
**Connection failed** on the main page; the Output terminal opens automatically
if hidden so the detailed error is visible.

If the address is unknown or SSH is unreachable, use **Network settings** as
described below, then return to **Target settings**.

Initialization uses the installed SDK's Python source and Linux dependency
wheels. It never copies the Mac's native modules or reuses an old robot-specific
`.deps` directory. Agent execution uses `python -S` with its own source and
dependency paths, excluding global Python site packages. Dependency import
checks complete before switching the active Agent release.

The Agent binds `127.0.0.1:50052` on the target. The SDK opens an authenticated SSH
forward and holds a remote login session for the connection lifetime. Connecting
starts the already-installed Agent if needed and waits for an actual Agent hello
through the tunnel before reporting success. Only SSH needs to be reachable from
the host.

### Network setup

Skip this page if the target already connects. Its three sections act on different
things:

- **Find a device** is read-only: list the Mac interfaces, select the USB/Ethernet
  interface connected to the board, then discover SSH neighbors on that link and
  choose an address. It does not change either device's IP or scan an entire subnet.
- **Mac network settings** changes the selected Mac network service's IPv4 mode
  to DHCP or a static address. macOS presents its own administrator authorization
  dialog; BotLab does not collect that password. Only the selected service is
  changed. A USB/Ethernet link may need a host address in the board's factory subnet
  before discovery can find it. This does not change the board's address.
- **Target network settings** changes the board's network. It requires SSH to be
  reachable first. Use **Read target network interfaces** in this section;
  BotLab uses saved session credentials or SSH keys, asks for a password when
  needed, and requires verification of a new host fingerprint. You can also
  enter the password in **Target settings** beforehand. Change the address only
  when the result reports supported NetworkManager capability. It cannot repair
  a device that has no usable SSH path.

Both network-setting sections are advanced, collapsible operations. Choose DHCP
or static explicitly before applying a change. Remote changes also require an
active NetworkManager-managed connection and either a root SSH account or
passwordless sudo. Other network managers are rejected before modifying the
configuration. For a static target address, leaving the gateway empty clears the
existing gateway; enter the gateway explicitly if it must be retained.

A target network change creates a 90-second NetworkManager checkpoint and applies
temporary settings. Reconnect through the new address and explicitly confirm to
make them persistent. Without confirmation, NetworkManager rolls back even if
the SSH session closes. For DHCP, discover and supply the new address before
confirming. Keep the SDK process alive while confirming; an expired transaction
must be retried from the restored address.

BotLab still needs a physical and management path to the device. Flashing an OS,
serial-console bootstrap, Wi-Fi credential setup and vendor-specific USB drivers
are not implemented. The CLI's optional `--profile` discovery hints remain
supported; hiding board type in the manager does not remove that compatibility.

### Command-line workflow

Install the matching development artifacts on the host:

```bash
pip install /path/to/botlab_python-<version>-py3-none-any.whl
botlab ui fetch --archive /path/to/botlab-browser-runtime-<version>.zip
```

Stop and restart any existing `botlab run` process after updating the SDK so the
new host-side credential and project-management behavior is loaded. Reload the
browser after installing its matching runtime.

Register and inspect a target. Replace placeholders with the actual SSH account,
address and host key; there is no built-in board password or automatic trust:

```bash
# Show local adapters, then discover neighbors on the chosen adapter.
botlab target discover
botlab target discover --interface <adapter-name> --profile generic-linux
# For an RDK X5, --profile rdk-x5 adds its factory-address hints.

botlab target add my-board --ssh-host <address-or-hostname> --user <login-user>
botlab target identity my-board

# After independently verifying the displayed SHA256 fingerprint, save it.
botlab target add my-board --ssh-host <address-or-hostname> --user <login-user> \
  --host-key 'SHA256:<verified-fingerprint>'

botlab target inspect my-board --ask-password
botlab target provision my-board --ask-password
botlab target list
```

For key-based login, add `--identity-file /path/to/private-key` when registering
the target and omit `--ask-password`. Standard SSH-agent/configuration behavior
is available. Passwords are requested through a terminal prompt, not a command
argument, and are not saved in target profiles. Each separate CLI invocation
needs credentials again unless SSH key authentication is available.

Deploy without running, inspect deployed projects, or launch the editor:

```bash
botlab deploy /path/to/my-project --target my-board --ask-password
botlab target projects my-board --ask-password
cd /path/to/my-project
botlab run --hil my-board
```

`botlab deploy` builds and installs the current model-service artifact. The
existing `botlab build` command still creates the ordinary browser release ZIP;
it is not a separate HIL compile command. Use **Build** in the HIL run menu to
build without deploying.

`botlab run --hil my-board` preselects the target. It does not immediately execute
or install anything. Connect from the main toolbar; optionally enter and save
an SSH password in **Target settings** first, or supply it when prompted. A normal
`botlab run` starts with local simulation selected and also provides the target
manager.

```bash
# Remove only the local saved target; does not delete board files.
botlab target remove my-board
```

Target settings live in `~/.config/botlab/targets.json`; `BOTLAB_TARGETS_FILE`
overrides this location. They contain connection settings and accepted public
host-key fingerprints, never passwords. Deleting a deployed project and all its
versions is a separate, confirmed action in Target manager.

Password caching lasts for the current host process, including browser reloads,
not across `botlab run` restarts. Long-term **Remember password** storage is not
implemented. A future OS credential-store integration could offer a similar user
experience to a browser password manager, but the current implementation does
not use macOS Keychain, Windows Credential Manager or a Linux secret service.

The SSH helper currently writes a password into a permission-restricted `0600`
file in a temporary directory for `SSH_ASKPASS`. Ordinary SSH commands clean up
that directory when they finish; an SSH tunnel retains it until the tunnel closes.
This is temporary file-based authentication, not a guarantee that passwords never
touch disk. The password is excluded from target profiles and operation logs.

## One-click and separate operations

After preparing a target, **Run on Target** connects when necessary, builds the
current project, deploys it, loads that exact revision, and starts inference.
The Agent verifies and reuses an already-deployed revision. The current transport
still sends the artifact during deployment; it does not yet implement a
missing-file-only transfer protocol.

| Action | Effect |
| --- | --- |
| Connect target | Open SSH forwarding and the Agent management connection. |
| Build | Validate the local project and package its declared models and manifest. |
| Deploy | Install the last successful build as an immutable target release without rebuilding or executing it. Requires a completed build and a connected target. |
| Load project | Directly load the last successful local build on the connected target and acquire its management ownership. That exact build must already be deployed there. Models load as the browser requests them during execution. |
| Run on Target | Build/deploy/load the current editor project and start the inference session. |
| Run (separate-operation menu) | Start only the currently loaded project/version, after checking that the local source and models match it. No build, deployment or project switch occurs. |
| Stop | End inference and release model execution resources; deployed files remain and project selection stays loaded. |
| Unload project | Release project ownership after stopping; the target connection remains. |
| Disconnect target | Close the connection and release its runtime/project ownership; deployed files remain. |
| Load version (Target manager) | Select an existing version on the connected target after stopping. May switch within one project or to another project; never builds or deploys. |
| Delete version | From Target manager, remove one inactive deployed version after confirmation. Deleting the last version also removes its project. Stop and unload first if this is the loaded version. |
| Delete project | From Target manager, remove the entire deployed project and all its versions after confirmation. Stop and unload first if this is the loaded project. |

For step-by-step execution, choose **Build → Deploy → Load project → Run**
in the run menu. Deploy remains disabled until a build succeeds. It
sends that build even if local files subsequently change; choose Build again to
include those edits. Build artifacts live in the current host process, so
restarting `botlab run` requires another build before deploying.

The run menu's **Load project** directly connects the current host process's
last successful build. Connect the target, then build and deploy that version
before loading it. Load project remains disabled until that exact version is
available on the connected target. If you build a newer version, deploy it before
loading again; this action does not fall back to an older deployment.

To choose another project or a historical version, open Target manager's
**Deployed projects** tab, expand a project and choose **Load version** for that
exact deployment. Browsing or expanding a project does not connect it or change
the local editor files. The toolbar shows the connected project and its version.

Several projects and versions can remain deployed, but only one project/session
can own execution at a time. The manager provides independent target browsing,
project scanning, version details, version/project switching, and deletion of a
single version or an entire project. Switching requires the browsed target to
be the connected target, and rejects stale target settings or deleted versions
without changing the current selection. Deletion refreshes both the manager and
the connected target's project inventory. Viewing or loading a deployed version
does not change the local editor source: **Run on Target** always prepares the current editor project.
**Run** in the separate-operation menu preserves the chosen version and requires matching
local source and model files; a mismatch leaves that selection intact and asks
you to open the corresponding local project revision. To reproduce an older
controller/model combination, restore the matching local project revision first.

Model execution occurs in a separate worker process so a stalled native backend
can be terminated while the management Agent remains available. Timeouts are
failure bounds, not real-time scheduling guarantees. A disconnect stops the
session; this model-service mode does not continue a standalone controller on
the board.

## Target files and project identity

The default root is `~/botlab`; set another dedicated root when adding the target
with `--root`. Do not use a symlink or a general-purpose system directory.

```text
~/botlab/
├── agent/
│   ├── releases/<agent-content-id>/
│   │   ├── package/botlab/          # SDK Python source
│   │   ├── deps/                    # Matching Linux binary/pure-Python wheels
│   │   └── manifest.json
│   ├── current -> releases/<id>
│   ├── start-agent.py
│   ├── agent.pid
│   └── agent.log
├── projects/<project-id>/
│   ├── project.json
│   ├── current                      # Selected build ID, stored as JSON
│   ├── releases/<build-id>/
│   │   ├── manifest.json
│   │   ├── .deployment.json
│   │   └── models/<sha256>.onnx
│   └── runs/<session-id>.jsonl       # Run lifecycle events
├── staging/                         # Incomplete uploads
├── cache/models/                    # Legacy direct-model session cache
└── environments/                    # Reserved; per-project environments not implemented
```

The first HIL build writes `.botlab/project.json` in the local project to give it
a stable ID. Preserve this file when updating that project; it distinguishes
same-name projects. Releases are content-addressed and checked before loading.
Deployment stages files, verifies their hashes and embedded weights, then
atomically creates the release directory. Failed uploads leave previous releases
available. Project deletion and local target removal are independent.

All deployed model-service projects currently share the provisioned Agent's
fixed ONNX Runtime environment. Arbitrary project requirements and independent
per-project dependency environments are not yet supported. The `environments`
directory does not imply those features are available. Run event logs and Agent
logs are stored on disk. The main Output terminal shows host operation logs;
it does not tail the target Agent log file.

If an existing systemd user manager is available, initialization installs and
enables a `botlab-agent-<root-id>.service` user unit. It starts at user login;
startup at boot without a login requires user lingering to already be enabled.
BotLab reports `autostart` accurately and does not enable lingering implicitly.
Without lingering, the persistent SSH login keeps the user service alive while
BotLab is connected; it may stop after the last login ends. Reconnecting starts
the installed service again, without reinstalling or changing system permissions.
When the user manager is unavailable, provisioning starts a detached managed
process and reports `autostart: false`; a later connection can start the installed
launcher after reboot. A running Agent is reused and is not restarted by another
connection. Neither service mode automatically starts a deployed project.

## Architecture and protocol

```text
Browser: physics + controller graph
    ↕ local authenticated HTTP
Mac: botlab-python Host, build/deploy, target inventory
    ↕ verified SSH tunnel
Linux: BotLab Agent, project store, isolated inference worker
    ↕
ONNX Runtime CPU
```

- `botlab.hil.devices`: SSH trust/authentication, discovery, system inspection,
  provisioning, network adapters and local forwarding.
- `botlab.hil.host`: local HTTP coordination, operations, sessions and metrics.
- `botlab.hil.artifacts`: deterministic builds, persistent releases and validation.
- `botlab.hil.agent` / `worker`: target management and bounded inference execution.
- `botlab.hil.backends`: model execution backend interface.
- Browser toolbar/runtime: source lives in the BotLab engine repository and is
  exported as the SDK's separately installed runtime archive.

The underlying wire format remains length-prefixed JSON protocol version 1,
with capabilities negotiated separately for project management. The Host and
updated Agent advertise `deploymentVersions: 1`; manager version switching and
individual version deletion require this Agent capability and otherwise ask you
to update the Agent first.
Whole-project deletion remains available with older project-management Agents.
A rejected project selection preserves its existing lease when the updated
Agent explicitly confirms that validation made no changes. Transport failures
and other operation errors retain the existing disconnect behavior. Models and
builds use SHA-256 identities. Named tensors carry dtype, shape and data; session and
sequence IDs reject stale inference results. Errors never silently switch to
local inference. Metrics measure real successful backend calls and latency.

The generic Linux adapter can be reused across compatible boards. A new
accelerator requires a compatible build/conversion path and execution backend,
not just another IP address. The existing direct-agent backend interface is:

```python
describe()  # {backend, provider, dtypes, ...}
load(path)  # (handle, input_ports, output_ports)
infer(handle, inputs)  # named wire tensors
close(handle)
```

Factories can be supplied to a manually installed Agent as
`--backend vendor_module:create_backend`. Automatic provisioning and the current
persistent deployment format only support `onnxruntime`; custom accelerator
build/dependency plugins are not implemented by that extension point alone.

### Compatibility with direct TCP targets

Existing manually prepared targets still work:

```bash
# On the target, with a matching SDK build and its [target] dependencies installed:
botlab agent serve --host <trusted-development-interface> --port 50052 --root ~/botlab
# On the host:
botlab target add legacy-board --endpoint <target-address>:50052
```

Direct TCP does not provide authentication or encryption. Use it only on a
trusted isolated development link; the SSH route is the standard onboarding path.
Direct targets cannot use SSH inspection, provisioning or network-management
operations. Older Agents without project capabilities retain direct model-session
behavior rather than gaining persistence automatically.

### Gateway targets (outbound device link)

The board can also dial out to a device gateway, so it never needs an inbound
port — the shape used for hosted/web HIL. The gateway relays opaque HIL v1
frames; the protocol, browser UI and acceptance rules are unchanged, and the
inference path carries no receipts or retry semantics.

```bash
# Wherever the gateway runs (loopback dev, or wss:// behind/direct TLS):
botlab-hil-gateway --host 127.0.0.1 --port 50060 --print-token
# Token via BOTLAB_HIL_GATEWAY_TOKEN; there is no tokenless mode.

# On the board: resident agent registers and redials after link loss:
BOTLAB_HIL_TOKEN=<token> botlab-agent connect \
  --gateway wss://boards.example.com --device rdk-x5-desk \
  --backend onnxruntime --root ~/botlab

# On the host:
botlab target add desk-x5 --gateway wss://boards.example.com --device rdk-x5-desk
BOTLAB_HIL_TOKEN=<token> botlab run --hil desk-x5
```

Rules: `ws://` endpoints are loopback-only; remote gateways must be `wss://`
(private CA via `BOTLAB_HIL_CA_BUNDLE`, verification never disabled). The token
is never stored in target settings and never passed as a CLI argument. Native
clients send it in the `Authorization` header; browsers, which cannot set
handshake headers, send the same credential as one leading text control frame
`{"type": "gateway-auth", "token": ...}` before the binary relay begins (any
text frame after that is still a protocol error). One
agent per device (re-registration takes over, closing the displaced pairing —
device restart discards stale sessions), one coordinator per device (a second
gets `device busy`), and either side dropping releases the HIL lease exactly
like a TCP disconnect.

Latency acceptance has two tiers: functional (the model executes correctly
through the gateway — `tests/test_hil_gateway.py`) and real-time (wall-clock
loop rate at the project's `control_hz`). Measure the latter with:

```bash
BOTLAB_HIL_TOKEN=<token> botlab target bench desk-x5 \
  --model policy.onnx --iterations 200 --budget-ms 20
```

which reports RTT percentiles, effective Hz and whether the serial inference
loop fits the 20 ms budget of a 50 Hz project.

Browser runs default to `127.0.0.1`. Explicit `--host 0.0.0.0` serves ordinary
local simulation on the LAN with HIL disabled. `--hil` requires loopback binding
and cannot be combined with native/headless/export/custom-adapter modes. The
local bridge uses a per-process token plus Host/Origin validation.

## Current limits

- Supported automatic target setup: Linux aarch64/x86_64 + glibc + Python 3.10+
  + SSH + compatible ONNX Runtime CPU wheels. No MCU/RTOS support or OS flashing.
- Static positive tensor dimensions, float32, named multi-input/output models.
  Dynamic shapes, rank-zero graph tensors and ONNX external weight files are
  unsupported. At most 32 model declarations and 32 MiB of unique model data per
  deployment; aggregate project storage is limited to 512 MiB. The legacy model
  cache has a separate 256 MiB limit.
- One active owner/session; all projects use the same provisioned runtime
  dependencies. No arbitrary target-side project Python or `requirements.txt`.
- No BPU conversion, TensorRT build, full-controller deployment, hardware actuator
  output, synchronized hardware clock or hard real-time deadline guarantee.
- Network changes support macOS host services and NetworkManager targets only;
  fresh devices still need a reachable bootstrap channel and valid credentials.

## Updating the browser engine

After changing the HIL UI or browser model execution, build the engine, export
and install its runtime together with the matching SDK build:

```bash
# In the engine checkout:
npm run build:dev
# With the updated SDK installed:
botlab ui export --botlab /path/to/BotLab --out /tmp/botlab-browser-runtime.zip
botlab ui fetch --archive /tmp/botlab-browser-runtime.zip
```

The archive contains `botlab-capabilities.json` with `hilProtocol: 1`. The CLI
rejects `--hil` with incompatible cached runtimes. Installing a Python wheel alone
does not update the browser code. No online BotLab deployment is needed.
