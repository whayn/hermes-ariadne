# v0.1 verification evidence

These are actual local executions, not inferred results. Re-run with the commands
in README; test identifiers and timings naturally differ between runs.

## Environment

- Inspected Hermes source: `/opt/hermes`; source contract details and repository
  revisions are in `integration-findings.md`.
- Test Python: 3.14.9, isolated `/opt/data/workspace/hermes-ariadne/.venv`.
- Host dependencies reused read-only from Hermes's site-packages. No install or
  edit was made to the managed Hermes environment.
- All stateful integration runs used disposable homes under the scratch directory.
  No Ariadne plugin was installed in `/opt/data/plugins/`.

## Full automated run

```text
HERMES_SOURCE=/opt/hermes HERMES_PYTHON=/opt/hermes/.venv/bin/python \
  bash scripts/test.sh -q -s

62 passed in 78.60s (0:01:18)
```

Coverage includes exact Unicode/whitespace/structured snapshots, repeated and
concurrent ingestion, branch/session scope, preserved revisions and tool identity
fields, missing identities, atomic rollback, schema/ownership refusal without
foreign-file changes, lone/paired surrogate fidelity, literal search without
escape aliasing, session lifecycle and reset-only rebind, clone isolation,
archive failure reporting, directory discovery/activation, configuration-preserving
installer refusal, abrupt process exit, and the real Hermes restart/tool loop.

Initialization coverage includes 100 rounds of eight simultaneous constructors
against nonexistent archives, a deterministic mixed-metadata interleaving, real
SQLite WAL contention, and bounded-retry exhaustion. Validation/initialization
now share a serialized transaction; journal-mode switching follows commit.
Installer coverage rejects inside/outside/dangling `plugins` symlinks and preserves
all existing final destinations. These regressions were added after independent
review reproduced both issues; the full result above includes the fixes.
Final independent reviews approved the storage concurrency fix and installer
containment fix with no remaining security or logic blockers in those scopes.

Final focused storage/crash run after the initialization-race fix:

```text
.venv/bin/python -m pytest tests/test_store.py tests/test_crash.py -q
40 passed in 25.71s
```

The initial surrogate binding regression was caught in independent review.
Additional test-first failures exposed standard ASCII-JSON's pair/scalar alias
and a derived-search escape alias. The final payload/search BLOB encoding uses
UTF-8 with surrogatepass; neither alias survives the regression tests. This is
an internal archive codec, not a new provider serialization layer. Invalid
Unicode codepoint sequences remain subject to Hermes/JSON transport semantics
when returned over a provider's tool protocol.

## Separate-process restart evidence

The successful full run's integration worker produced:

```json
{
  "seed_pid": 108135,
  "restart_pid": 108216,
  "same_message_uid": "aca556ab150941508a77c6c681f23864",
  "event_id": "a745ee5cf177a5b1687d4d999387f9fecd5a22934335a7680395fbeb601f4cb8",
  "exact_content": "ARIADNE-RESTART-MARKER café\n\t  preserve this spacing  ",
  "recovered_missing_hook": 1,
  "search_turn_api_calls": 2,
  "archive_error": null
}
```

Both workers instantiate the installed Hermes `AIAgent`, load the copied plugin
via real discovery, bind a real `SessionDB`, and use real HTTP/SSE transport to a
scripted loopback provider fixture. The first exits with `os._exit(0)`, deliberately
skipping session-end cleanup after an additional host row is committed. The
second reconciles that row, resumes history, receives a provider tool-call fixture,
dispatches `ariadne_search` through Hermes, and sends its exact stored result back
to the fixture before receiving the final answer fixture. No paid LLM was called.

The fixture also confirms the host's existing outgoing whitespace normalization
can differ from the stored user original: Ariadne preserves the durable original,
not the provider-facing sanitized version. Identity fields stay off the provider
message envelopes; quoted archived identity metadata in tool-result data is
intentional.

## Packaging and rollback smoke test

```text
.venv/bin/python -m build
Successfully built hermes_ariadne-0.1.0.tar.gz and
hermes_ariadne-0.1.0-py3-none-any.whl

uv pip install --python <new scratch venv>/bin/python <wheel>
Installed 1 package: hermes-ariadne==0.1.0

<scratch venv>/bin/ariadne-install --home <new scratch home>
configuration unchanged

HERMES_HOME=<scratch home> <Hermes Python> -m hermes_cli.main \
  config set context.engine ariadne
wheel-only directory discovery and activation: PASS

HERMES_HOME=<scratch home> <Hermes Python> -m hermes_cli.main \
  config set context.engine compressor
rollback to built-in compressor: PASS

No Ariadne installation in the live profile: PASS
```

A fresh Python environment installed the wheel with no Ariadne development-source
path. The copied directory then loaded in Hermes's existing interpreter without
installing Ariadne into it. Activation and rollback were read back through the
real configuration loader and `_select_context_engine`, not inferred from CLI
exit codes alone.

```text
.venv/bin/ruff check src tests
All checks passed!
.venv/bin/ruff format --check src tests
12 files already formatted
git diff --check
(no output)
```

Build emitted byte-compilation-disabled warnings from this runtime, but produced
both artifacts and the installed wheel smoke test passed. No claims are made
about the entire upstream Hermes test suite, UniiChat's test suite, live gateways,
other providers, plugin-host isolation or production power-loss testing.

## Offline kernel evidence (issue #6)

The deterministic kernel demo is separate from live Hermes context selection.
No private corpus or real model provider was used.

- Regenerated `tree_golden.json` by executing the pinned original `Memory.ts`
  (`0500bf6a8ea4bf0ce047488562022b1b6532cc30`), not a handwritten reference
  reimplementation. Node 22.18.0 was required for TypeScript transform mode;
  its downloaded archive was checked against the official SHA-256 manifest.
- Oracle output: 77 scenarios, 359 transitions, 11 root cases, 5 part cases and
  11 text cases; regenerated JSON matched the checked-in fixture byte-for-byte.
- `.venv/bin/python -m pytest tests/test_tree.py -q`: **164 passed**.
- Source CLI demo and a freshly installed wheel in a separate scratch venv both
  archived **65 synthetic messages**, built **128 ready nodes**, and left **0
  pending nodes**. The view used **188 raw-summary UTF-8 bytes**. The simulator's
  complete proposal measured **3,232 ASCII-JSON bytes**, with the retained suffix
  unchanged; these are different measurements, not provider token estimates.
- A separate recall process navigated persisted summaries to the original code,
  reconstructed **49 canonical-byte pages**, and confirmed exact code and event
  identity. Archive row count remained **65** before and after recall. The recall
  phase performs no ingestion, summary callbacks or synthetic source generation.
- Original and recovered code SHA-256:
  `e860ae7edd9cc07de83377c442bba23fa55d737d37d39d748cff8c6589411d13`.
- Distribution checks confirmed reference JSON/MJS fixtures in the sdist and the
  offline modules plus Apache NOTICE in the wheel. The initial packaging check
  failed because non-Python fixtures were missing; `MANIFEST.in` now includes them.

The final post-review full regression run returned **304 passed in 287.88s**,
including real Hermes integration, both safety fixes and the existing 1,000-round
concurrent-initializer test. Hosted CI/Codex status is recorded on the PR.
These executions prove the synthetic offline data path, not semantic recall
quality, provider caching, live compaction or production-scale performance.
