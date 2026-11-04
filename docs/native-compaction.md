# Native offline compaction and durable receipts

Scope: [issue #8](https://github.com/whayn/hermes-ariadne/issues/8). The installed
Hermes engine **still only archives/searches**; this module does not activate live
compaction, replace the harness, import personal history or run a background agent.

## The native seam is deliberately restricted

Hermes documents `ctx.llm` as the supported live-plugin LLM interface. Inspection
of the installed auxiliary implementation found that its high-level recovery
ladder may retry, change routes or omit output limits. Plugin results also do not
expose all finish/usage-presence information needed for these receipts. Do not
assume a `max_tokens` argument or SDK retry setting establishes a physical-call
bound on that high-level path.

This first implementation instead runs in a **dedicated sequential offline
worker**. It obtains the existing client through Hermes's native auxiliary
factory, retaining its routing, authentication and HTTP transport. It supports
only validated synchronous OpenAI SDK Chat Completions clients and Hermes's
native Codex wrapper. It does not create a second provider client/HTTP transport,
reimplement authentication, or replace native text/finish normalization.

Compatibility is intentionally narrow: OpenAI SDK **2.24.0**, HTTPX **0.28.1**, and
the inspected native client shapes. Other versions/adapters fail closed. The
fingerprint records the actual installed `auxiliary_client.py` hash, not a guessed
historical Git revision. Do not downgrade or modify a managed Hermes environment
to bypass a refusal. Live integration needs a separate host-compatible strict
request boundary; **never attach these hooks to a live/shared concurrent agent**.
This lower-level lane does not provide the full `ctx.llm` trust/audit surface.

## Dispatch, budget and usage truth

- The native request hook runs **after native content encoding**. It validates
  endpoint/model/cap against bounded decoded JSON, but hashes/reserves the exact
  encoded HTTP body delivered to transport. Native Codex zstd compression stays
  enabled; unsupported/corrupt encodings fail closed. Both decoded and encoded
  bodies must fit the configured byte bound. It then commits the reservation.
  If validation/reservation fails, no request is sent. A second send/redirect or
  concurrent dispatch is rejected.
- SDK retries are temporarily disabled and restored with request/response hooks
  in `finally`. The auxiliary recovery/fallback ladder is not invoked.
- The planner predicts full ASCII-JSON requests plus a framing margin. The wire
  guard independently checks actual serialization; prediction is not wire proof.
- Hermes's native token counter is a **rough full-prompt estimate**, checked
  against an explicit configured planning window. Capped requests reserve at least
  `max(configured reserve, output-token cap)`; explicitly uncapped requests retain
  only the configured planning reserve. This is neither exact provider
  tokenization nor a monetary ceiling.
- Chat routes require an output-token cap. Codex's endpoint rejects that field:
  it requires both `max_output_tokens=None` and explicit unbounded-output consent.
  Short-summary instructions are not a hard token ceiling. The demo's Codex
  exception was separately authorized; do not silently generalize that consent.
- Missing usage remains `None`. For Codex, bounded raw SSE metadata observation
  preserves absent/null/partial usage and cache/reasoning details instead of
  trusting synthetic zeros from compatibility normalization. Observe SSE bytes
  **after HTTPX content decoding**, not the raw gzip/zstd transport stream.
  Hermes still owns all text and completion-state normalization. Individual metadata frames are
  limited to 1 MiB; this is not a global output-token limit.
- A normalized `finish_reason="stop"` is not Codex completion proof: the native
  wrapper can synthesize it at EOF. Require an observed valid successful terminal
  event before publishing success. Text followed by missing/malformed terminal
  evidence remains charged and uncertain, with no automatic retry. Completion
  validation identity is part of the frozen campaign route. An absent
  `Content-Type` is allowed because the pinned native SDK supports headerless
  SSE; it never substitutes for terminal proof. Explicitly incompatible media
  types, headerless JSON and text-without-terminal still fail closed.
- Empty, refused, incomplete, invalid-UTF8 or oversized generated text is not
  committed as a node. Known invalid replies consume attempts. There is no
  clipping or fabricated fallback summary.

## Durable state and crash boundary

One owned SQLite ledger is **one immutable campaign**: projection/session/event
identities, kernel settings, prompt content hash/version, planner, route and
policy are bound to frozen limits. It is separate from the canonical archive and
disposable kernel checkpoint. Existing unverified manual summaries are rejected;
small literal leaves are checked directly against their exact source and are
not represented as model-generated output.

Plans bind stable node/chunk/merge slots to their first immutable context and
input lineage. A later frontier cannot mint a fresh budget or erase uncertain
work. Counters survive restart; failed/uncertain reservations are never refunded.
The ledger enforces campaign attempts, exact total/per-request body bytes,
per-job attempts and total jobs. Chunk count/reduction limits also bound a root
plan's job count. There is no currency cap without trustworthy price/charge data.

```text
planned → dispatching → succeeded + immutable receipt
                    → known failure (bounded retry possible)
                    → uncertain (no automatic retry)
```

A reservation and dispatch marker commit before send. A valid receipt commits
before the callback returns generated text. On restart, unfinished dispatches
become uncertain. A crash between marking and sending can therefore block a job
that was never billed: that conservative ambiguity is intentional. The system
cannot guarantee exactly-once remote execution/charging. There is no blind
uncertainty reset/refund/migration API in this milestone.

Completed receipts rebuild a lost kernel checkpoint without making model calls.
Changing an immutable namespace/limit requires an explicit different campaign;
the runner never silently creates one to bypass exhaustion. Local reduction
aliases reference completed model results, not invented replacement summaries.

The ledger uses WAL/FULL synchronization and a POSIX exclusive writer lock,
private files/directories, exact schema ownership and hash validation. Its entire
filename family is checked bidirectionally against protected archive/kernel
families, including aliases/hardlinks. Foreign WAL-backed databases are inspected
through private scratch copies rather than recovered in place. POSIX `flock` is
required; malicious path replacement/non-cooperating writers are outside scope.
Copies and the in-memory derived catalog are not large-history performance claims.

## Long source records

The runner freezes contiguous, non-overlapping spans over the complete escaped
JSON input serialization. Each chunk has offsets/hash; adjacent completed results
are reduced through bounded model jobs with ordered dependency hashes. All chunks
and reductions inherit the node's original historical context cutoff. Optional
context may lose **whole parts under an explicit policy**, never source coverage.
Any missing/failed/uncertain chunk prevents publishing the leaf/its parent.

This does **not** make archive reading streaming: the existing source getter and
canonical page interface materialize full records. Planner source/chunk limits
are explicit. Exceeding them yields incomplete work, not a partial-success summary.
Canonical sources and exact byte-paged `zoom` remain independently recoverable.
Coverage proves which data was processed, not semantic retention or injection
resistance. Model summary quality and autonomous recall need separate evaluation.

## Reproduce without a real provider

Use the repository's isolated development setup and inspected host dependencies:

```bash
.venv/bin/python -m pytest tests/test_receipts.py tests/test_compaction.py -q
HERMES_SOURCE=/path/to/hermes HERMES_PYTHON=/path/to/hermes/.venv/bin/python \
  bash scripts/test.sh tests/test_native_compactor.py tests/test_compaction_transport.py -q
```

Native tests use real factory/SDK HTTP transport against **scripted loopback**
responses with isolated HOME/HERMES_HOME, whitelisted environment and network
restrictions. These are not claimed real-model summaries. The combined transport
test builds the tree and starts another process to rebuild from receipts with no
additional HTTP requests. Keep full-suite/CI verification separate from unit counts.

## Explicitly authorized synthetic real-model demo

The demo can spend provider quota. Use a **NEW scratch output directory** and an
explicitly selected, dedicated development Hermes profile, never the inherited
live/default home. Prepare/authenticate that profile through Hermes's normal
mechanisms; do not copy credentials into the demo. Shared native authentication
may still be used/refreshed by Hermes. Profile separation is not an OS sandbox.
The demo does not install a plugin, activate Ariadne, edit profile configuration
or import conversation history. Its records are fixed synthetic code/garden text.

```bash
export HERMES_SOURCE=/path/to/hermes
export HERMES_PYTHON=/path/to/hermes/.venv/bin/python
# Select a dedicated, authenticated named development profile explicitly.
# Inherited HERMES_HOME is never accepted for build; no default-profile fallback.
export ARIADNE_DEV_HOME=/path/to/hermes-home/profiles/ariadne-dev

# Chat-Completions-compatible configured compression route, with cap:
bash scripts/compactor-demo.sh "$TMPDIR/ariadne-native-demo" \
  --phase build --hermes-home "$ARIADNE_DEV_HOME" --allow-model-calls

# OR explicit Codex route, ONLY with separately authorized uncapped output:
bash scripts/compactor-demo.sh "$TMPDIR/ariadne-native-codex" \
  --phase build --hermes-home "$ARIADNE_DEV_HOME" --allow-model-calls --allow-unbounded-codex-output \
  --provider openai-codex --model YOUR_CONFIGURED_CODEX_MODEL

# New process, fresh derived state, saved receipts; no host auth/model/network:
bash scripts/compactor-demo.sh "$TMPDIR/ariadne-native-codex" --phase rebuild
```

Build requires `--hermes-home PATH` (or API `build(..., hermes_home=PATH)`). The
canonical existing path must be `ROOT/profiles/NAME`, with a separate `dev` or
`development` token in its name (hyphens/underscores delimit tokens), and a regular,
non-symlink `config.yaml`. Missing/default/non-development selections and aliases
resolving outside an acceptable canonical development profile fail before native
imports or output creation; aliases to acceptable canonical profiles are allowed.
The fresh worker binds HOME/HERMES_HOME
and removes inherited config/env/profile overrides. The launcher disables Python
site/.pth startup before selection. This is an operator-selected development
boundary, not proof that arbitrary plugins or profile contents are harmless.
Receipt-only rebuilding needs no selected profile.

If the managed runtime uses a different SDK pair, do **not** downgrade it. Create
a disposable compatibility venv with OpenAI 2.24.0/httpx 0.28.1, select its Python
as `HERMES_PYTHON`, and append the managed dependency site-packages read-only via
caller `PYTHONPATH`; the wrapper places the selected compatibility site first.
[CI compatibility](native-ci-compatibility.md) records the same deliberate test
override and the separately tested pristine Hermes baseline. SDK and profile
isolation are separate requirements.

Each demo campaign allows at most **3 physical model requests**, **8,000 body
bytes per request**, and a fixed total request-byte reservation. Failed requests
still consume the allowance. Rerunning build refuses an existing directory;
rebuild is receipt-only and cannot dispatch. Creating another paid campaign is
an explicit new authorization, not an automatic retry workaround. A pending
result exits nonzero; inspect its ledger/status rather than inventing success.

The rebuild phase navigates generated summaries to the code identifier, then
reassembles canonical archive bytes and verifies exact code/event identity.
The manifest stores identities and a verification hash, not a source copy.
A successful tiny smoke test does not certify general memory quality or a
production/live deployment.
