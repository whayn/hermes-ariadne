# Automatic rolling context (experimental, opt-in)

`ariadne-rolling` is a separate development engine. On a validated successful
completed turn, it can ask **Hermes's own model connection** for a summary of an
older part of this conversation. Later requests can send a bounded summary view
instead of that older part, followed by the unchanged recent conversation and
unchanged host instructions. It can prepare a first view and advance that view
as more completed exchanges accumulate. Ordinary `ariadne` remains passive;
`ariadne-prepared` still requires operator-prepared artifacts.

This slice supplies packaging, registration and pure contract tests. **Actual
installed-host rolling runtime proof follows separately.** Fake-context and
AST-extracted-host tests are not native discovery, provider compatibility,
long-horizon recall, daily-use readiness or live activation evidence. Do not
activate this on an important/live profile.

## What is retained, and what leaves future requests

The host's session DB stays authoritative and is never rewritten. Exact archived
JSON-value snapshots, including supported older opaque Codex continuation state,
remain in `$HERMES_HOME/ariadne/events.sqlite3`. Once a complete summary boundary
has passed structural/source validation, that older state may stop being sent in
**future** requests with the summarized prefix. It is not converted into readable
reasoning. Recent replay state and recent complete tool groups remain unchanged
in the assembled request suffix. Archive availability is not a claim of identical
provider continuation behavior.

Summaries are lossy. Coverage hashes, exact source identities and complete
exchange checks establish structural correspondence, **not semantic completeness
or accurate remembering**. Navigation is a bounded flat summary view, not the
older offline binary tree. Evidence tools treat history as untrusted data:

- `ariadne_find` searches the session archive with bounded output; it is not
  limited to just the summary's prefix and can include historical versions.
- `ariadne_rolling_page` pages the pinned generation's ordered source descriptors
  (event references and roles, not full source text).
- `ariadne_zoom` resolves exact event references and serves paged canonical bytes.
  Binary-tree node references are unavailable in rolling mode. Reassemble byte
  pages before decoding UTF-8; a page can split a code point.
- Rolling generation references expire when that generation is no longer the
  selected view; they never silently resolve to a newer summary. Exact event
  references are separately source/hash checked and session scoped.

The runtime validates current active sources again before planning and before
publication. Missing persistence, interrupted/failed completion, revision,
rewind, unsupported replay/restriction carriers or unsafe archive paths can
prevent preparation/selection. Pending, failed or exhausted work retains the
previous usable view rather than publishing partial output. A view is pinned for
a turn's requests; it is not replaced in the middle of a tool exchange.

## Install separately; do not activate implicitly

Use the project's standalone environment, never install dependencies into the
managed Hermes environment. Choose a **new disposable development home**:

```bash
export ARIADNE_DEV_HOME=/absolute/path/to/new/ariadne-dev-home
.venv/bin/ariadne-install --home "$ARIADNE_DEV_HOME" --rolling
```

The installer copies a strict 17-file runtime allowlist into
`$ARIADNE_DEV_HOME/plugins/ariadne-rolling-plugin/`. It does not copy the whole
package, caches, private configuration, credentials or history. It does not call
a model, alter configuration or activate anything. `--rolling` and `--prepared`
are mutually exclusive; no flag still installs the four-file passive runtime,
and `--prepared` still installs its existing 11-file runtime. Existing final
files/directories/symlinks and a symlinked `plugins` parent are refused. An
explicit symlinked home is canonicalized; partial copy failures remove the new
partial plugin. Concurrent malicious directory replacement is out of scope.

### Native registration contract

The **general-plugin ID is `ariadne-rolling-plugin`**, while the **engine name is
`ariadne-rolling`**. Keep that directory/name distinction: Hermes tries a matching
named directory engine before the general manager's cloned engine. Do not rename
this installation to `plugins/ariadne-rolling`. The limited directory collector
has no LLM/config facilities; the entry refuses it and exports no engine class
for its fallback scan.

A full general `PluginContext` supplies `ctx.get_config(key, default=None)`,
`ctx.llm`, and `ctx.register_context_engine(engine)`. The entry reads only
`get_config("rolling", {})`, validates an immutable `RollingSettings`, retains
the exact facade and returns the registrar's result unchanged. Bad/unknown
rolling limits raise; registry errors propagate. A `None` return is not guessed
to mean rejection (host versions differ); inspect native plugin logs/selection.
Hermes permits only one general registered context engine, so do not enable a
competing general engine. Cloning retains the facade/settings and allocates fresh
mutable agent state. Import, construction and tool schemas do no archive/model I/O.

The manifest is JSON-formatted YAML, schema version 2, `kind: standalone`, with
no general tools/hooks or required secrets. Tools and lifecycle callbacks belong
to the ContextEngine, not general-plugin event subscriptions.

### Explicit configuration for a later development-host test

These are instructions, **not actions performed by the installer or this slice**:

```bash
# Review local plugin code and the native enablement/trust prompt first.
HERMES_HOME="$ARIADNE_DEV_HOME" hermes plugins enable ariadne-rolling-plugin
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set plugins.isolation in_process
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set context.engine ariadne-rolling
```

`hermes plugins enable` maintains `plugins.enabled` and `plugins.disabled`;
the disabled list wins. Do not bypass native trust/admission checks. Changing
`plugins.isolation` affects other user plugins **in this development home**;
never change a live/global isolation policy for Ariadne. Only `in_process` with a
live `SessionDB` is supported. In `host` isolation that binding is opaque: the
engine stays unavailable, performs zero preparation LLM calls, and does not
silently change isolation. It is not an OS sandbox.

Restart only the disposable development process after explicit setup. If the
platform uses an explicit enabled toolset list, retain desired existing toolsets
and include `context_engine`; an empty list exposes no navigation tools. Native
routing/authentication stays Hermes-owned; there is no separate provider/client,
credential copying, replacement harness or core fork.

## Limits and accounting

Settings live under
`plugins.entries.ariadne-rolling-plugin.settings.rolling`. For example:

```bash
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set plugins.entries.ariadne-rolling-plugin.settings.rolling.max_logical_jobs 8
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set plugins.entries.ariadne-rolling-plugin.settings.rolling.timeout 60
```

`ctx.get_config` cannot enumerate the entire plugin settings root: the supported
configuration is the single `rolling` mapping, not sibling settings or flat
limit keys. Unknown keys **inside that mapping** and invalid values refuse
registration; values are not silently coerced/defaulted. Absent fields use:

| Setting | Default |
| --- | ---: |
| `max_logical_jobs` | 64 |
| `max_request_bytes` | 65536 |
| `max_total_request_bytes` | 2097152 |
| `max_readable_delta_bytes` | 32768 |
| `max_delta_exchanges` | 4 |
| `trigger_bytes` | 4096 |
| `max_summary_bytes` | 4096 |
| `max_navigation_bytes` | 8192 |
| `max_candidate_bytes` | 65536 |
| `reserve_tokens` | 4096 |
| `max_prefix_events` | 100000 |
| `max_checkpoint_bytes` | 16777216 |
| `max_tokens` | 2048 |
| `timeout` | 60 |

Every limit must be an explicit integer from 1 through 2147483647 (not a bool,
string, float or null). These are development defaults, not recommended daily-use
or billing settings. `trigger_bytes` is the serialized UTF-8 length of one virtual
canonical JSON array containing the readable_delta descriptors for newly aged,
contiguous completed exchanges after the published frontier (excluding parent summary,
prompt wrappers, previously summarized events, opaque replay values, and the newest
completed exchange). This prefix-accumulation threshold is distinct from the per-job
bounds: once triggered, preparation selects the earliest contiguous delta admitted by
`max_delta_exchanges` and byte limits. `max_logical_jobs` counts durable logical
reservations and request byte limits count serialized facade work, including cumulative
charged work across advances/restart. **They are not physical HTTP attempt/cost caps.**
Hermes may retry or fall back internally within one facade call. Observed usage
is bounded, allowlisted metadata, not verified billing; missing usage is unknown,
not measured zero. Interrupted reservations stay charged/uncertain, not free retries.

Maintenance is synchronous and can delay turn completion. The facade timeout is
forwarded, not a promise of cancellation or a strict callback latency deadline;
work assumes the host call returns. There is no background executor or guaranteed
scheduler. Message-byte estimates and an operator token reserve do not certify
full provider fit (tool schemas, output runway and provider overhead also matter).
A declined `select_context` returns `None`: Hermes can send the original request,
which can overflow. This is **not a dispatch veto** or a universal size guarantee.

**Known performance limit (#19).** Every request and every rolling navigation call
reopens the session journal and revalidates the pinned view. Validation is
memoized within one journal operation only: each archive event is read once, and
each generation is validated and cross-checked against the journal once per open.
There is no cross-request cache, so each request still revalidates every published
generation. Archive reads per open are linear in source events. CPU time grows
roughly quadratically, because each generation canonicalizes its own source
prefix. Measured on a synthetic aarch64 journal at the default 64-job bound: one
open takes 128 archive reads and about 1.3 s (previously 187,200 reads and 129 s).
A later `load_view` takes about 0.7 s and one view revalidation about 0.5 s. These
are component timings, not an end-to-end request measurement. Whether this
per-request cost is acceptable, or a separately invalidated cross-request cache is
needed, is still open; activation for normal use remains blocked on that decision.
If a session exceeds `max_prefix_events` before the trigger threshold is reached,
or the logical-job budget is exhausted, rolling status reports `unavailable` while
any retained view keeps being served; the label is coarser than the cause.

## Durable state, rollback and evidence

Rolling journal schema v2 retains work/charge lineage. Existing v1 journals fail
closed, without reset, migration or spending-counter forgiveness. Operator reset
or recovery is out of scope; do not delete journals to obtain automatic retries.
Archives/journals contain sensitive source material and require private paths;
there is no encryption, erasure or cross-profile retrieval feature.

To stop using the experimental engine in the same disposable home:

```bash
HERMES_HOME="$ARIADNE_DEV_HOME" hermes config set context.engine compressor
HERMES_HOME="$ARIADNE_DEV_HOME" hermes plugins disable ariadne-rolling-plugin
```

Restart that process. Retain archive/journal files for evidence; the installer has
no overwrite/upgrade/migration mode. Packaging checks and pure tests do not prove
actual agent selection, two automatic frontier advances, restart continuation,
wire preservation, model-backed semantic quality or production reliability.
Those require separate native-host fixtures and explicitly authorized pilots.

Pure verification (read-only AST extraction, no host bootstrap):

```bash
ARIADNE_PURE_HOST_SOURCE=/absolute/path/to/hermes .venv/bin/python -m pytest \
  tests/test_rolling_plugin.py tests/test_install.py \
  -k 'not installed_directory_loads' -q
```

The omitted pre-existing installer test imports native host discovery; it belongs
to the separately isolated native test run, not this pure command.
