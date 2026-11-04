# BotLab Control API v1

`botlab control` is a robot-independent client. A project supplies its own middleware,
capability declarations and execution backend. The client uses Python's standard library;
it does not import project code, a robot adapter, aiohttp, MuJoCo or ONNX Runtime.
The browser execution protocol is separate and is not changed by this API.

## CLI

```sh
botlab control --project . instances --json
botlab control --project . capabilities --json
botlab control --project . experience --json
botlab control --project . sessions --json
botlab control --project . state --json
botlab control --project . call joint.position --args '{"positions":[0.1,-0.1]}' --request-id my-request --json
botlab control --project . wait --id my-request --timeout 1 --json
botlab control --project . retry --id my-request --json
botlab control --project . stop --request-id my-stop --json
botlab control serve --port 8776 --origin https://your-botlab-page.example
```

`call` accepts an object via `--args` or `--args-file`. `move --vx V --vy V
--yaw-rate W --duration SECONDS` requires the declared `motion.twist` capability and
submits `vx`, `vy`, `yaw_rate`, `duration_s`. `action NAME` resolves a unique alias
from capabilities. Neither command knows robot-specific limits or action names.
Unsupported capabilities return `unsupported_capability`; there is no fallback action.
The service is the authoritative validator. The CLI checks finite JSON and common
schema constraints (type, required/properties, additionalProperties=false, enum/const,
numeric bounds and array items/length), not all of JSON Schema.

Options `--project`, `--endpoint`, `--instance`, `--session` and `--json` may appear
before or after the subcommand. The project defaults to the current working directory.
Multiple live instances require explicit selection. `connection` reports the selected
address and pairing guidance; only explicit `--show-token` exposes the pairing token.
`serve` starts the generic middleware below; apart from it the CLI does not start a
project middleware or manipulate a browser window.

JSON stdout contains one object, including on argument errors. Exit codes are 0 for
success or pending, 2 for parsing/connection/request errors, and 3 for a cancelled,
failed or expired command. A pending wait does not cancel a command. Wait timeout is
0..30 seconds; network timeout leaves the execution outcome unknown.

## Discovery and authentication

Canonical project identity is `str(Path(project).resolve())`; `project_key` is the
first 16 hex characters of SHA-256 of its UTF-8 encoding. The default discovery root is
`Path(tempfile.gettempdir()) / ('botlab-control-' + user_id)`, where user_id is the
POSIX numeric UID, or SHA-256 of `getpass.getuser()` truncated to 12 hex characters.
`BOTLAB_CONTROL_RUNTIME_DIR` overrides the root.

Each middleware writes `<root>/<project_key>/<instance_id>.json`, with 0700 directories
and a 0600 file owned by the current user. Symbolic-link directories/files are rejected.
A descriptor contains:

```json
{
  "schema": 1,
  "api_version": 1,
  "protocol": "botlab-control",
  "project": "/absolute/project/path",
  "project_key": "path-hash",
  "instance_id": "new-uuid-for-each-service-start",
  "endpoint": "http://127.0.0.1:8776",
  "websocket": "ws://127.0.0.1:8776/runtime",
  "token": "temporary-private-pairing-token",
  "pid": 123
}
```

All HTTP requests carry `Authorization: Bearer <token>`. Credentials are never sent
in query strings. Only loopback HTTP origins with an explicit port are accepted;
URL credentials, paths, queries and fragments are refused. Environment HTTP proxies
and HTTP redirects are disabled. Explicit endpoints still require a matching private
descriptor for this project. Every descriptor is verified against `info` before use;
the CLI does not select targets by process start time. Stale descriptor files may be
listed as unavailable and are not removed by the client.

## HTTP endpoints

All paths are under `/control/v1`. Successful bodies contain `ok:true`; errors contain
`{"ok":false,"error":{"code":"stable_code","message":"description"}}`.
New optional fields are compatible; unknown API major versions are rejected.

| Method / path | Meaning |
| --- | --- |
| GET `/info` | `api_version:1`, `protocol:'botlab-control'`, project, project_key, instance_id, backend `{id,kind}`, `extensions` |
| GET `/capabilities` | api_version, instance_id, capabilities_revision, capabilities array |
| GET `/experience` | instance_id, capabilities_revision and the project's text experience document |
| GET `/sessions` | instance_id and available sessions/targets; no FSM requirement |
| GET `/state` | instance_id, project_key and current state |
| POST `/commands` | submit the complete command envelope |
| GET `/commands/{request_id}?instance_id=...&timeout=1` | query or bounded wait for the original instance |
| POST `/stop` | submit the same envelope with `capability:'control.stop',args:{}` |

`control.stop` is a mandatory protocol operation, also accepted at `/commands`.
The generic client uses `/commands` so stop shares the same request registry and retry
behavior. Control API v1 requires `command_id` to equal `request_id` for every
command. Submission responses and subsequent queries preserve this identity; queries
use that same ID. Querying a different instance
returns `wrong_instance`; unknown IDs return `not_found`; reclaimed results return
`result_evicted` without permitting replay.

Each capability declares `id`, `version:1`, optional `aliases`, `input_schema`, `units`
(parameter-name to unit mapping), `frame` (string or null), `max_duration_s` (positive
number or null), and nonempty `completion_semantics` and `stop_semantics`. Unitless
actions use an empty units object and null frame. Current readiness is independent of
capability support. Services may publish only joint or end-effector capabilities,
without movement, sitting, a floating base, or a scalar policy state.

The state envelope includes `connected`, `external_enabled`, `ready`, `run_state`,
`session_id`, `episode_id`, sampling/progress timestamps or steps, `state_age_ms`,
`progress_age_ms`, and `active_command_id` where applicable. Robot measurements belong
in `observations`, with schemas/frames supplied by the middleware. Connection heartbeats
do not count as advancing physics. The middleware enforces freshness and robot limits.

## Command identity and retries

```json
{
  "request_id": "client-generated-stable-id",
  "instance_id": "service-start-id",
  "session_id": "connected-runtime-session",
  "episode_id": "current-reset-episode",
  "capabilities_revision": "adapter-capabilities-revision",
  "capability": "joint.position",
  "args": {"positions": [0.1, -0.1]}
}
```

The service validates all identity fields and uses one deduplication registry for all
public and compatibility routes. An identical ID/content returns the original command;
different content conflicts. Service restarts create a new instance_id and never replay
old motion. Heavy results may be reclaimed, but lightweight ID tombstones must survive
for the instance lifetime. A service at its published identity limit rejects new tasks
instead of forgetting IDs and allowing duplicate motion.

Before a first submission, the CLI persists the exact envelope, endpoint and normalized
intent under `~/.local/state/botlab/control/receipts/<project_key>/<sha256(request_id)>.json`.
When state supplies `control_generation`, the receipt preserves it separately from the
public command envelope as an additional retry precondition.
`BOTLAB_CONTROL_RECEIPTS_DIR` overrides the root. Directories are 0700; files are 0600.
Publication is atomic and exclusive, using a fully written/fsynced temporary file plus
a no-overwrite hard link. Filesystems lacking this operation cause submission to fail
before any command is sent. Receipts contain no token, are not auto-replayed, and are
not auto-deleted: deleting one removes the ability to safely retry that request.

`retry --id` and repeated calls with an existing request ID read the original receipt,
verify its project/instance/endpoint and query the original task first. Terminal or
pending tasks are returned without submission. Only an explicit unknown-task response
from the same instance allows checking the original session, episode and capabilities
revision and resending the exact envelope. Missing receipts, replaced instances,
changed identities, unavailable services and reclaimed results refuse resubmission.
For an unknown task with a recorded control generation, the current generation must
still match; a changed or absent generation returns `state_changed` without resending.
This prevents a pre-pause request that never reached the service from executing after
resume. Middleware without this optional state field remains compatible. Existing task
results can still be queried after generation changes.
Timeout/error output after submission includes request_id and instance_id. A changed
intent cannot reuse the request ID. A new movement decision requires fresh state and a
new ID; `retry` is not a resume operation.

## Completion and physical evidence

Command responses contain `command_id`, `instance_id`, `session_id`, `episode_id`,
`status`, `terminal`, `initial_observation`, `completion_observation` and `current_state`.
Statuses are accepted, applied, completed, cancelled, failed and expired.
Wait responses add `pending`; accepted/applied are not completion.

`initial_observation` labels its capture stage (typically request acceptance), not an
unverified physical start instant. `completion_observation` is a committed snapshot
matching the completion acknowledgment's episode and step; if unavailable it is null,
with an explicit absence reason. `current_state` is query-time feedback and must not
be substituted for completion evidence. Already-satisfied actions identify that basis.

Completion is capability-specific: a bounded velocity command expiring and zeroing
does not prove arrival at a position; entering a posture state does not prove physical
stability. Stop cancels pending movement according to backend semantics and is not
necessarily an emergency freeze or interruption of a running dynamic action. Local
middleware/backend deadlines and leases, not model or CLI response timing, enforce
bounded execution. Pause/reset/disconnection never implicitly replay old commands.

## Generic middleware: `botlab control serve`

`pip install "botlab-python[serve]"` adds the generic middleware. The client CLI stays
standard-library-only: `serve` imports its server module lazily, and without the extra
reports `missing_dependency` (exit 2) instead of crashing.

```sh
botlab control serve --port 8776 --origin http://127.0.0.1:5199
```

`--origin` (repeatable) allowlists the BotLab page origin for browser pairing and is
required. Discovery registration, permissions (0700/0600), the pairing token
(`--print-token` or the descriptor) and remote enrollment (`--public-endpoint`, HTTPS
only) follow the rules above unchanged. `/info` declares `extensions:
["project-descriptor"]`; clients ignore extensions they do not know.

The middleware itself is project-agnostic: the paired page uploads the project's
control semantics at connect time, and the middleware compiles them into capabilities.
Until then `/capabilities`, `/experience`, `/commands` and `/stop` answer
`no_project_paired` (HTTP 409), while `/info` and `/state` keep working.

Pairing sequence over the `/runtime` WebSocket (the v1 message envelope is unchanged):

```
hello {token}                              page → middleware
welcome {profile: null, descriptor:{version:1}}   middleware → page (negotiation bit)
descriptor {profile, motion, actions, states?, experience?}   page → middleware
descriptor_ack {capabilities_revision, profile}   middleware → page
... then state / command / renew / result exactly as before
```

A middleware without the `descriptor` negotiation bit never receives a descriptor
frame, so old pages and old middlewares interop unchanged. On reconnect the retained
descriptor is echoed in `welcome.profile` and re-uploads are idempotent (same revision);
a second descriptor on one connection closes it with `invalid_message`. Structural
validation failures close with `invalid_descriptor` (or `descriptor_too_large` for an
experience document over 32768 bytes); state frames before a descriptor are
`invalid_message`. The middleware performs structural validation only — cross-checks
against the actual project (the robot exists, the state node and parameters are
consumed, state codes cover the FSM) stay on the page, which owns the project manifest.

Descriptor fields (frame ≤ 64 KiB): `profile` with `robot_id`, `state_node_id`,
`motion_parameters` mapping exactly `vx` and `yaw_rate` to signal names,
`event_parameter` and nonzero unique `event_values`; `motion` with `vx`/`yaw_rate`
`{min<max, unit}` and `allowed_states`; `actions` as an array of
`{name, event_value, allowed_states, target_states, capability?, idempotent?}` —
names match `^[a-z][a-z0-9_]{0,63}$`, capability ids are dotted-lowercase with
`motion.twist`/`control.stop` reserved, and two actions may share an event value (a
toggle distinguished by `allowed_states`); optional `states` mapping canonical decimal
codes to names; optional `experience` text. Target states are declared, not derived
from FSM transitions: steady-state targets are reached via after-step chains.

The compiled surface fixes `motion.twist` (descriptor bounds, `vy` constant 0,
`duration_s` 0.05..10) and `control.stop`; each action becomes a capability
(`action.<name>` unless overridden) carrying
`preconditions {allowed_state_codes, target_state_codes}`. The middleware self-checks
the compiled table against the frozen capability contract.
`capabilities_revision` is `"gen-" + sha256(canonical descriptor)[:16]`, so any
descriptor change forces clients to re-read capabilities.

## Initial integration boundary

MicroDuck is the first adapter; `botlab control serve` is the generic path for browser
projects, with the paired page supplying the project's semantics. Project-authored
adapter modules remain the escape hatch for headless/native execution, where no page
exists to upload a descriptor. MicroDuck's middleware retains the existing browser
WebSocket v1, old HTTP routes and `python -m microduck_cli` entry point. New and old
clients share one task registry. Installing the SDK alone does not upgrade a running
middleware; deploy the compatibility service first, then install the CLI and verify
actual feedback.
The joint-only HTTP fixture in `tests/test_control_cli.py` demonstrates protocol
independence; it is not a hardware driver or a validated physical robot backend.
