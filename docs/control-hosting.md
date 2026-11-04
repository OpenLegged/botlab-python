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

## M1: remote control over HTTPS (implemented)

M1 ships the remote transport without any account system, using **restricted
developer credentials** (the middleware token, shared out of band):

- **Endpoint rules** (`botlab.control.discovery.endpoint`): loopback HTTP
  keeps the explicit-port rule; HTTPS is accepted for any host with an
  optional port. Plaintext HTTP never leaves the loopback.
- **Enrollment** (`botlab control enroll --endpoint https://…`): the CLI
  probes `/info`, pins the reported `project_key`/`project` identity and the
  `instance_id` into a `0600` discovery descriptor marked `remote: true`.
  Every later command — including `wait`/`retry` — flows through the normal
  descriptor, receipt and restart-refusal semantics with zero changes.
  Identity comparison for remote descriptors uses the enrollment pin, not the
  local checkout path (trust-on-first-use; M2 replaces this with accounts).
  Re-enrolling the same endpoint replaces its previous descriptor; a second
  endpoint reusing an instance identity fails with `instance_conflict`.
- **Credentials**: the token reaches the CLI via `BOTLAB_CONTROL_TOKEN` or a
  hidden prompt — never argv. For private CAs set `BOTLAB_CONTROL_CA_BUNDLE`
  to the CA file; TLS verification is never disabled.
- **Middleware exposure** (MicroDuck reference): `serve --host 0.0.0.0`
  behind a TLS-terminating reverse proxy, `--public-host` to allowlist the
  proxy's Host header, `--public-endpoint https://…` to record the public
  origin in the descriptor, `--print-token` for the operator channel. The
  browser panel connects to `wss://<public>/runtime` with the same token.

Acceptance for M1 is encoded in `tests/test_control_remote.py`: enrollment,
pinned-identity control, restart refusal until re-enrollment, identity-pin
mismatch, self-signed rejection without the CA, and multi-instance selection —
all over real TLS with a generated dev CA. Genuine cross-machine acceptance
(two physical machines) remains a manual step: same commands, `--endpoint` on
the other machine's origin.

Explicitly **not** in M1: accounts and remote discovery (M2), instance pools
and isolation (M3), web HIL (H1 — separate chain, never rides these
semantics).

## H1: web HIL device chain (implemented)

HIL inference rides its own chain and inherits **no** Control-command
semantics: no receipts, no retry, no task table on the inference path. H1
turns the board agent's inbound TCP listener into an **outbound** link so the
board never needs an inbound port:

- **Gateway** (`botlab-hil-gateway`, `botlab.hil.gateway`): relays opaque HIL
  v1 frames between a coordinator (`/device/<id>`) and a resident agent that
  dialed out (`/agent/<id>`). It never parses HIL frames. One shared bearer
  token (operator channel, generated at startup or via
  `BOTLAB_HIL_GATEWAY_TOKEN`, revealed with `--print-token`); there is no
  tokenless mode. Native clients present it in the `Authorization` header;
  browsers cannot set handshake headers, so they send the same credential as
  one leading `{"type": "gateway-auth", "token": ...}` text control frame
  before the binary relay begins. `ws://` is loopback-dev only; remote gateways
  are `wss://` (direct via `--tls-cert/--tls-key` or a TLS reverse proxy).
  Compression is off and links dial directly (no ambient system proxy).
- **Agent** (`botlab-agent connect --gateway wss://… --device <id>`):
  registers, serves the same dispatch loop as TCP mode, and redials after
  lease termination or link loss (`--retry-interval`, default 5 s). The token
  arrives via `BOTLAB_HIL_TOKEN` or a hidden prompt — never argv. Private CAs
  use `BOTLAB_HIL_CA_BUNDLE`; verification is never disabled.
- **Coordinator** (`transport: "gateway"` targets): `ws://` endpoints stay
  loopback, `wss://` accepts any host — the same shape as the Control M1
  endpoint rule (`gateway_endpoint` in `botlab.hil.protocol`). The token is
  never stored in target settings; the run server reads `BOTLAB_HIL_TOKEN`.
- **Pairing semantics**: one agent per device (re-registration takes over and
  closes the displaced pairing — device restart), one coordinator per device
  (a second gets `device busy`), and either side disappearing closes the
  other, which releases the HIL lease exactly like a TCP disconnect. Stale
  sessions are refused by the agent and the rejection terminates the lease;
  nothing is replayed.

**Acceptance is two tiers** (`tests/test_hil_gateway.py` covers the first):
functional — the model executes correctly through the gateway, exclusive
pairing, takeover and stale-result rejection, over loopback `ws://` and real
`wss://` with a generated dev CA; real-time — wall-clock loop rate at the
project's `control_hz` (50 Hz → 20 ms budget) measured on real hardware via
`botlab target bench <name> --model policy.onnx --budget-ms 20`, which
reports RTT percentiles, effective Hz and a budget verdict. Serial
multi-model inference (one Promise chain in the browser) is the loop shape
that must fit the budget.

Explicitly **not** in H1: per-device credentials and accounts (M2), multi-
tenant routing, and any receipt/retry semantics on the inference path.

## Frozen by these suites

- `tests/test_control_protocol.py` — frozen constants, envelope validation,
  extension surface, `project_key`.
- `tests/test_control_cli.py` — client identity, retry and restart refusal
  semantics (dedup, eviction, control generation, unsafe discovery).
- `tests/test_hil_gateway.py` — H1 functional tier: relay transparency,
  exclusive pairing, takeover, stale-session rejection, endpoint rules, TLS.
- microduck-project `tests/test_control_api.py`, `tests/test_control_service.py`
  — middleware conformance, including extension-surface and frozen-envelope
  rejection tests.
