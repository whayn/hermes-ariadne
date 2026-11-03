# Hermes Ariadne v0.1

An installable, Hermes-native **ContextEngine development plugin**. It archives
verbatim message snapshots in SQLite and exposes `ariadne_search`. It is not a
new harness, frontend, or Hermes fork. No Hermes core patches are required.

**Do not enable this milestone on an important/live profile.** v0.1 does not
compact, prune, or select replacement context. It takes the ContextEngine slot,
so it displaces the built-in compressor; long chats can hit the provider context
limit. This is the durable-storage vertical slice, not yet a production
long-horizon memory engine.

## What works

- Exact JSON-value snapshots: Unicode, whitespace, structured content, nulls,
  tool calls/results, host identity fields and metadata are preserved.
- Host `message_uid` identifies a logical message. A deterministic SHA-256
  `event_id` identifies each exact snapshot. Repeated ingestion is idempotent;
  edits and metadata changes append versions, never overwrite originals.
- SQLite transactions, WAL, FULL synchronous writes, dedicated archive directory;
  no long-lived connection or thread-affine state.
- Session-scoped reconciliation using the **bound host's read API** at start,
  turn completion, session end, and search. This also catches retained rows from
  failure paths whose turn-complete callback did not fire.
- `ariadne_search` returns exact stored snapshots, not generated answers/snippets.
- No `select_context` override; provider requests and tool dispatch remain owned
  by Hermes. No model clients, embeddings, summaries or background agent calls.

See [inspected contracts and architecture](docs/integration-findings.md).

## Project and contributions

Maintained by [Whayn](https://github.com/whayn), under [Apache-2.0](LICENSE).
The long-term direction is a UniiChat-inspired memory/context engine **inside
Hermes**. The installed runtime implements the archival foundation described
above; a separate offline tree/view/zoom kernel is available for development.
There is no stable release or PyPI publication.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for commit/branch naming, PRs, issue triage,
verification and release policy. [AGENTS.md](AGENTS.md) carries repository-local
agent guidance. Report vulnerabilities through [SECURITY.md](SECURITY.md), not a
public issue. Public reports and fixtures must use synthetic data, never private
life-chat transcripts or memory archives.

Repository conventions draw on Hermes Agent, T3 Code and hermes-lcm; the pinned
references and deliberate differences are recorded in
[repository-conventions.md](docs/repository-conventions.md). UniiChat remains the
memory-algorithm reference; the offline pure tree/view port is attributed in
[NOTICE](NOTICE). It is not wired into the installed runtime.

## Installation (isolated home first)

Prerequisites: an installed Hermes with the inspected ContextEngine/identity
contracts. Tested against the installed source described in the findings doc,
with Python 3.14.9. Archive/installer code requires Python >=3.11, but Hermes's
own version requirements still apply. There is no claimed compatibility with
older Hermes releases missing `message_uid` or the observation hooks.

Use a **separate Ariadne environment**; do not mutate Hermes's managed Python
installation. With `uv` (or equivalent `venv` + pip in your disposable env):

```bash
cd /path/to/hermes-ariadne
uv venv --python /path/to/hermes/.venv/bin/python .venv
uv pip install --python .venv/bin/python -e '.[dev]'

# Pick a NEW path, not your normal HERMES_HOME or another existing profile.
export ARIADNE_DEV_HOME="$PWD/.dev-home"
.venv/bin/ariadne-install --home "$ARIADNE_DEV_HOME"

# Run the installed Hermes CLI with this explicit isolated home.
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set context.engine ariadne
```

If `hermes` is not on PATH, the tested source entry point is:

```bash
HERMES_HOME="$ARIADNE_DEV_HOME" /path/to/hermes/.venv/bin/python   -m hermes_cli.main config set context.engine ariadne
```

The installer **only copies** four runtime files into
`$ARIADNE_DEV_HOME/plugins/ariadne/`; it refuses to overwrite any existing
file/directory/symlink at that path and rejects a pre-existing `plugins` symlink.
The explicit home itself is canonicalized (an intentionally symlinked root is
allowed). Malicious concurrent directory replacement is outside its threat model.
It does not activate the engine, copy
credentials, import conversations, edit settings, or install into Hermes's
Python environment. A wheel install works too:

```bash
.venv/bin/python -m build
uv pip install --python /path/to/disposable/venv/bin/python dist/hermes_ariadne-0.1.0-py3-none-any.whl
/path/to/disposable/venv/bin/ariadne-install --home /path/to/new/dev-home
```

Restart only the **development** Hermes process after installation/configuration.
A general `plugins.enabled` entry is not needed. The tool must be allowed by the
platform's enabled toolsets: if you use an explicit list, retain all your desired
existing toolsets and include `context_engine`. An empty list intentionally
exposes no engine tools. `skip_memory` does not disable this ContextEngine.
No extra Ariadne configuration is needed in v0.1.

Archive location: `$HERMES_HOME/ariadne/events.sqlite3` (and SQLite WAL/SHM files).
The host `state.db` remains authoritative; Ariadne never writes to it. By default
only the session currently bound to the engine is read. No profile-wide backfill,
ancestor crawling, cross-session search or cross-profile memory is implemented.

## Search

Tool name: `ariadne_search`

```json
{"query": "a literal phrase", "limit": 20, "offset": 0}
```

- Case-sensitive **literal substring** search over stored `content`, not SQL
  LIKE patterns, a query language, FTS ranking, or semantic retrieval.
- `limit`: integer 1–100, default 20; `offset`: nonnegative integer, default 0.
- Result: `results`, page `count`, and `session_id`. Each hit carries `event_id`,
  `message_uid`, the exact `message` mapping, `origin` and observed `position`.
  Position is the index in the source snapshot, not a globally stable identifier.
- Search includes **historical versions**, including retained inactive/rewound
  host rows. One message may therefore yield multiple snapshots. Metadata changes
  also count as versions; a copied durable row can yield a new snapshot while
  retaining the same logical UID. A hook-only and durable snapshot are distinct
  provenance representations, not silently reconciled into a guessed original.
- Page count is not a total or a promise of more results. Offset pagination is
  deterministic for a fixed archive; concurrent ingestion can shift pages.
- Archived text is untrusted historical data, not new instructions. Exact results
  can be large; use a selective query and small limit. v0.1 does not clip records.

Ordinary bound agents archive decoded `SessionDB.get_messages` row mappings with
`origin=hermes_db`. This preserves the authoritative persisted content, which can
differ from injected or sanitized provider-visible text. With no bound session
store, observation falls back to identified hook mappings (`origin=hook`). No
UID is invented for missing/legacy identities; the scan reports skipped rows.

## Tests

No paid provider or credentials are required. The integration fixture serves
scripted OpenAI-compatible JSON/SSE responses on loopback; the **real Hermes
agent, HTTP transport, persistence, discovery and tool loop** are exercised.
These responses are test fixtures, not claimed model-generated output.

```bash
# Storage tests can run without importing Hermes.
.venv/bin/python -m pytest tests/test_store.py tests/test_crash.py -q

# Full suite, using installed Hermes dependencies READ-ONLY.
export HERMES_SOURCE=/path/to/hermes
export HERMES_PYTHON=/path/to/hermes/.venv/bin/python
bash scripts/test.sh -q

# Print two-process restart evidence (UIDs, process ids, exact text).
bash scripts/test.sh tests/test_restart.py -q -s

.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m build
```

The Ariadne test Python must have the **same Python minor version** as the Hermes
runtime when reusing its compiled dependencies. `scripts/test.sh` points
`PYTHONPATH` at its site-packages rather than installing into it. Tests isolate
`HERMES_HOME` before host imports; integration children additionally use a
whitelisted environment, a disposable HOME, explicit session id, and no live
credentials. Temporary files use the caller's TMPDIR. Never run integration tests
by repointing them to a production home.

## Disable / rollback / upgrade

```bash
# Only the development home, not your normal profile.
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set context.engine compressor
```

Restart that development Hermes process. Hermes now uses its built-in compressor
and does not inject `ariadne_search`. The plugin directory may then be moved out
of `$ARIADNE_DEV_HOME/plugins/` or deleted. No DB migration or transcript restore
is necessary. Keep the archive if you want to retain the snapshots. For an
upgrade, disable first, move the old plugin directory aside, install anew and
re-test before re-enabling; the installer deliberately has no force-overwrite.

The wheel environment can be removed independently; the installed directory is
self-contained and has only Hermes/stdlib runtime imports. Archive backups must
use SQLite's backup API or be made with all writers stopped (do not copy only the
main DB while WAL writes are active). No backup/export/migration CLI is shipped.

## Reliability, privacy and limitations

- **Not yet a complete lossless-history guarantee.** A mutation or deletion
  before Ariadne first observes it cannot be reconstructed. A missed callback is
  healed only if the persisted row still exists and that session is reconciled
  again. SQLite protects completed archive transactions, not an unobserved turn.
- Lifecycle ingestion failures fail open, leave live context untouched, log only
  the exception type, and appear as `archive_error` in engine status/search.
  A successful later scan retries missing snapshots; never infer archive success
  solely from a successful Hermes turn. Missing identities are separately counted
  in `archive_skipped_last_scan`.
- Inactive/rewound rows are deliberately archived. Hermes rewind/delete is NOT
  deletion of Ariadne's copies. There is no erasure command or retention policy.
- Snapshots can include secrets, tool payloads, URLs, reasoning and attachment
  references. The directory is created private and the DB with mode 0600, but
  there is **no encryption or secret redaction**. Backups inherit this sensitivity.
  External attachment bytes are not copied; exact references are not durable media.
- Full session rescans and literal scans are intentionally simple and may become
  expensive with large histories. No cursor optimization, FTS index or derived
  index rebuild command yet. The offline summary tree is a separate rebuildable
  projection over these authoritative snapshots, not a replacement for them.
- There is no automatic current-version resolver or revision chronology API. Results are
  ordered by observed position and snapshot id, not by edit time. The archive
  retains versions but does not infer an edit sequence from timestamps.
- The tested path is the in-process directory loader and local chat-completions
  transport. Plugin-host isolation, remote worker/store variants, every provider,
  gateway/desktop UI lifecycle, power-loss hardware guarantees and multiprocess
  long-running production loads are not certified by this milestone.

## Offline selection simulator (development only)

A separate, stdlib-only [context-selection simulator](docs/context-selection-simulator.md)
now exercises synthetic summary coverage, explicit retention boundaries, complete
tool sequences, measured budgets and logical cache-prefix invariants. Run it with:

```bash
.venv/bin/python -m pytest tests/test_simulation.py -q
```

It is not called by the engine or copied by the runtime installer. Its default
measurement is canonical ASCII-JSON bytes, **not provider tokens**, and it neither
creates summaries nor implements UniiChat's tree/view algorithm. The warning at
the top of this README remains in force: live context is still not compacted.

## Offline memory kernel (development only)

The [archive-backed tree/view/zoom kernel](docs/offline-memory-kernel.md) now connects
an explicitly selected source projection to persisted hierarchical summaries,
a bounded view and paged exact-source retrieval. Its pure tree/view algorithm is
ported from a pinned UniiChat revision and checked against executable reference
fixtures. It is not copied by the runtime installer or called by the live engine.

Run the two-process synthetic demonstration in a NEW scratch directory:

```bash
.venv/bin/python -m hermes_ariadne.demo "$TMPDIR/ariadne-kernel-demo" --phase build
.venv/bin/python -m hermes_ariadne.demo "$TMPDIR/ariadne-kernel-demo" --phase recall
.venv/bin/python -m pytest tests/test_tree.py tests/test_kernel.py tests/test_kernel_demo.py -q
```

The deterministic topic-label provider is a fixture, **not a real AI compactor
or recall evaluation**. The second process navigates persisted summaries and
recovers old code from the archive, without re-ingestion or source regeneration.
See the kernel document for byte accounting, exact page codec, source scoping,
single-writer/checkpoint limitations and what was deliberately not ported.

**Next milestone:** a real, measured compactor using Hermes model routing, then
isolated live selection/zoom integration. Keep live `select_context` inherited
until the host/provider boundary is tested. Proactive recall remains a later stage.
