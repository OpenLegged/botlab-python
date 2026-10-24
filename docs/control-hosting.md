# Hosting the Control API (Route A, milestone M0)

This document freezes the contract decisions every later hosting milestone
(M1 remote control, M2 accounts, H1 web HIL, M3 platform) implements against.
It records *decisions*, not aspirations: each one is enforced by a named test
suite listed at the end. The hosting goal itself — browser editor served from
a URL, server-side middleware per session, outbound CLI — is out of scope here.

## Architecture target

Two independent chains, deliberately not merged:

- **Control chain** (low-frequency commands, cloud relay acceptable):
  one-shot CLI / external agent → HTTPS → cloud control service ← outbound
  WSS ← the BotLab page (physics, signal graph, project Python stay in the
  user's browser).
- **HIL chain** (in-loop inference, separate acceptance): page → device
  gateway ← outbound WSS ← resident board agent running ONNX. The board still
  runs an agent; the user's machine runs at most a browser and an optional CLI.

## Version and extension strategy (frozen)

- The v1 command envelope is **frozen at exactly eight fields** (see
  `ENVELOPE_FIELDS` in `botlab/control/protocol.py`). Hosted identity, routing
  and tenancy never add envelope fields; they travel in discovery descriptors
  and `/info`. A middleware must reject unknown envelope fields (this is what
  `validate_envelope` does), so an old middleware cleanly refuses a newer
  client's extension instead of executing a partially understood command.
- `/info` declares `"extensions": [...]` — the negotiation surface. Clients
  ignore extensions they do not understand; local v1 declares none
  (`PROTOCOL_EXTENSIONS == ()`). Hosted features ride named extensions.
- `API_VERSION` bumps are reserved for incompatible envelope or semantic
  changes, and require a full discovery→submit compatibility matrix across
  old/new clients and middlewares. Adding an extension is not a version bump.

## Hosted identity model (to be implemented in M1/M2)

A hosted identity namespace binds, at minimum: **service environment + account
or tenant + project ID + execution instance**. Consequences:

- Receipt directories are keyed by the full namespace, never by a bare project
  name, so two deployments' same-named projects never share receipts.
- Descriptors need an explicit instance routing design (path segment, subdomain
  or routing field): several descriptors sharing one public origin cannot
  address an instance today. The stable public address is recorded in receipts.
- Local and hosted modes are selected explicitly; hosted mode must not require
  downloading project source.
- `project_key` (path digest) remains the *local* identity; the hosted
  `project_ref` is a separate, explicitly negotiated field.

## Restart and deduplication semantics (frozen)

Safety rests on three things holding together: CLI receipts on the user's
machine, server-side deduplication records, and instance identity changing on
restart.

- A worker restart produces a **new `instance_id`**. Requests bound to the old
  instance are refused (`original_instance_unavailable`); nothing is
  auto-recovered or replayed. This is a feature, not a limitation.
- The in-memory task table may evict results under retention pressure; evicted
  IDs keep tombstones and answer `result_evicted` — never a re-execution.
- **Prohibition:** an identity-preserving failover (same `instance_id` on a new
  worker) must not be built without persistent deduplication, atomic accept
  and executor mutual exclusion. A lost task table must never surface as a
  resend-permitting `not_found`.
- Server-side task and audit records are *not* CLI receipts. CLI receipts stay
  on the CLI machine (`0600`, never uploaded) — both statements hold at once.

## Dependency rules

- `botlab.control.protocol` and the client are **standard-library only**.
  Server-side dependencies (aiohttp, later the middleware base class) never
  enter the CLI import chain.
- Project middleware imports `botlab.control.protocol` directly; the resulting
  package import cost (~80 ms) is accepted for now, deferring a lazy
  `botlab/__init__` to a later milestone if it ever matters.
- The middleware base class (M3) unifies envelope validation, task tables and
  state machines, but is **not** a third-party code isolation boundary: real
  isolation needs separate restricted workers. First hosted phase supports
  platform-maintained adapters only.

## HIL boundary (for H1)

HIL inference rides its own chain and must not inherit Control-command
semantics: no receipt/retry on the inference path, and acceptance splits into
a functional tier (model executed correctly on the board) and a real-time tier
(wall-clock loop rate at the project's `control_hz`, e.g. 50 Hz with serial
multi-model calls). Device restart discards stale inference results.

## Frozen by these suites

- `tests/test_control_protocol.py` — frozen constants, envelope validation,
  extension surface, `project_key`.
- `tests/test_control_cli.py` — client identity, retry and restart refusal
  semantics (dedup, eviction, control generation, unsafe discovery).
- microduck-project `tests/test_control_api.py`, `tests/test_control_service.py`
  — middleware conformance, including extension-surface and frozen-envelope
  rejection tests.
