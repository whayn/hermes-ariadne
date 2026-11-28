# Ariadne plugin (MVP, issue #41; memory tree, issue #54)

One Hermes plugin, `ariadne`, that you select in **both** settings slots:

| Slot | Setting | What it does now |
|---|---|---|
| Memory provider | `memory.provider: ariadne` | Profile notes (`ariadne_notes`), one-way mirror of built-in memory writes, recall of relevant notes each turn, `ariadne_import` for archiving conversations |
| Context engine | `context.engine: ariadne` | One summary tree per profile over all primary conversations; each request carries the tree view plus this chat's recent messages; tools `ariadne_zoom`, `ariadne_search`, `ariadne_memory_status` |

Status: **experimental prerelease (v0.3.0); use a dedicated development profile**. The
tree (#54) replaces the v0.2 rolling summary and past-conversation index. The
dev-profile ship gate with a real model (one provider) is in
[Ship gate](#ship-gate); read [Limitations](#limitations) and
[Rollback](#rollback) before installing.

## The memory tree

Ariadne keeps one chronological tree per profile over **every primary
conversation**, live chats and imported ones. Subagent and cron sessions are
not included. There is no per-source rule: every chat in the profile can see
every other chat's summaries and exact messages.

Every request the engine sends is:

1. the host prefix (system messages), unchanged;
2. the **view**: one assistant message of `id+n|text` lines (each line a
   summary of `n` messages starting at message `id`), followed by a user message;
3. this chat's most recent messages verbatim (`tail_bytes`, default 32768).

This chat's older messages are dropped from the request only once built
summaries cover them. A stretch not yet summarized collapses into one range line
(`first..last|(N messages not summarized yet ...)`), so the view stays small
right after a big import. If the normal view exceeds a known request token
budget, Ariadne also tries a smaller navigation-only index (`compact_view` in
memory status). It shows zoomable ranges instead of summary text and keeps
all messages in the current request verbatim. If neither request is valid or
fits, Ariadne declines selection and Hermes sends the original request.

Tools:

- `ariadne_zoom(id, n, at)`: open a visible line `id+n` into the two lines it
  was made from; `n=1` gives one message in full (paged from `at`). Start from
  a relevant line in the view when you have one; searching first isn't required.
- `ariadne_search`: literal, case-sensitive search over exact archived messages
  of every conversation in the view, newest tree positions first. Follow `next_cursor`
  with the same query to reach older results, even if a page is empty. Pages
  cover the tree as of the first search; start without a cursor to see messages
  added to the tree since then.
- `ariadne_memory_status`: summary progress and summaries waiting to retry, plus the
  latest selection attempt in this chat. `selection.status` is `selected`,
  `declined`, or `not_evaluated`. A decline includes a reason and, for request
  shape errors, a `shape_check` label without message text (for `message_identity`,
  also an `identity` count of messages without a unique `message_uid`, their roles,
  whether they all come after the stored rows, and the same counts for the
  stored rows). This is the engine's
  decision for its latest request, not a provider receipt or a diagnosis of an
  earlier reply. `retry` asks failed summaries now instead of waiting.
- `ariadne_notes` and `ariadne_import` (below).

Recall of relevant notes remains. The v0.2 past-conversation index recall and
`ariadne_conversations`, `ariadne_find` and `ariadne_rolling_page` are gone; the
exact archive is reached through `ariadne_zoom` and `ariadne_search`. Hermes's
stored history is never modified, and summaries are navigation data, not
instructions.

### Summaries: cost and latency

Summaries are written in the background through Hermes's LLM facade under the
auxiliary task `ariadne_summary`; a turn never waits for them. By default it
uses the chat model. Once started, the worker can drain the backlog after the
last chat closes, while its Hermes process is still running. On hosts with
plugin idle tasks, the `resume-tree` task wakes an existing backlog without
opening a chat. It runs on the first eligible maintenance poll and no more
than once an hour; that cadence persists across restarts. On older hosts,
opening a chat still wakes the builder. **Recommended: pin a faster, cheaper
model** with the
`auxiliary.ariadne_summary` block (`provider`, `model`, `base_url`, `api_key`,
`timeout`, like Hermes's other auxiliary tasks; the Hermes auxiliary settings
UI lists it as "Ariadne summaries"). In the real runs Mistral Large took about
60-120 s per call. Each call was about 6-13k tokens (its context view reaches up
to `seen_high`).

```bash
hermes config set auxiliary.ariadne_summary.provider <provider>
hermes config set auxiliary.ariadne_summary.model <model>
```

Checked on Hermes 0.23.3: with the plugin loaded, the task resolves to the pinned
provider and model. `hermes config get` warns that `auxiliary.ariadne_summary` is
not a recognized key, because the CLI checks keys before plugins register their
tasks; the setting is still read.

`ariadne_import` returns an estimate of the summary calls an import adds. It is
informational, not a cap. Large imports and the first backfill can mean
thousands of calls; the first real run made about 210 calls (~2.1M tokens) for
215 messages.

Failed summaries never stop the tree, they only delay it (as in UniiChat). A
summary call that fails (provider error, timeout after `timeout` seconds, an
unusable answer) is asked again after `retry_base` seconds (30), doubling with
each consecutive failure up to `retry_max` (3600); the schedule survives
restarts. After two failed calls in a row in a lane (live chats or imports), that
lane sends no new summary call until the failed ones are due again, so an outage
is probed by a few calls per wait rather than one per pending summary; a failing
import never pauses the live lane. When a process exits it waits up to
`exit_wait` seconds (120) for in-flight calls; a summary lost with its process is
asked again on the next start.

Cost of failures: each retry is a new call, and a failed call may still have
been billed by the provider. During an outage the failed summaries are asked
again after 30 s, 1, 2, 4… minutes, so about 8 times each in the first hour and
then once an hour, up to `jobs` (4) summaries per lane at a time. A summary that
fails every time while the provider works (for example a request too large for
a pinned model's context window) is retried once an hour indefinitely and its
older history is not merged above it. Set `max_jobs` or
`max_total_request_bytes` for a hard lifetime cap.

`ariadne_memory_status` shows how many summaries are waiting (`retrying`), when
the next one is due (`next_retry_in`), whether a lane is paused (`paused_for`,
seconds) and how many messages cannot be read at all (`unreadable`); its `retry`
action asks failed summaries at once and lifts the pause. The v3 tree store
written by this version cannot be opened by v0.3.0 (no downgrade).

### Settings

Settings live under `plugins.entries.ariadne.settings.tree` (defaults from
`plugin.yaml`; unknown keys or out-of-range values refuse registration):

```bash
hermes config set plugins.entries.ariadne.settings.tree.tail_bytes 2048
```

(Verified working on Hermes 0.23.3.)

| Key | Default | Meaning |
|---|---|---|
| `tail_bytes` | 32768 | This chat's recent messages sent verbatim |
| `high` | 128000 | Bound on built summary bytes in the view (view limit is about twice this) |
| `seen_high` | 32000 | Largest context view a summary call is shown |
| `node_bytes` | 512 | Target size of one summary line (32-8192) |
| `jobs` | 4 | Concurrent summary calls per lane (1-64) |
| `max_jobs` | 0 | Lifetime logical summary-call reservations; zero is unlimited |
| `max_request_bytes` | 65536 | Logical cap per summary request (4096 up) |
| `max_total_request_bytes` | 0 | Lifetime logical bytes over summary requests; zero is unlimited |
| `tries` | 5 | Asks per summary while the answer is too long; the shortest is kept (1-20) |
| `max_tokens` | 1024 | Output tokens per summary call (reasoning models spend part on thinking) |
| `timeout` | 300 | Seconds per summary call (1-3600) |
| `exit_wait` | 120 | Seconds a process waits at exit for in-flight calls (0 = never) |
| `retry_base` | 30 | Seconds before a failed summary call is asked again (1-86400) |
| `retry_max` | 3600 | Longest wait between asks; the wait doubles up to it (1-86400) |

`auto_reask_lost` (v0.3.0) is still accepted and ignored: lost summaries now
retry like any other failure.

Job and byte caps count logical reservations, not HTTP requests or billing.

### Upgrading from v0.2.x

On first start the engine backfills the existing archive into the tree in the
background. The archive is kept; the notes conversation index is no longer
used by the `ariadne` engine. Since #70 the rolling and prepared engines are gone:
their leftover directories (`$HERMES_HOME/ariadne/rolling/` and
`$HERMES_HOME/ariadne/prepared/`) are left untouched and unused, and can be
deleted by hand; a leftover `rolling` block in the plugin config is ignored. Summaries for
the backlog are then built, which costs model calls (see above). Note-store and
tree-store schemas migrate in place; an older revision refuses a newer store.

## Install (development profile)

```bash
export HERMES_HOME=/path/to/dev-profile          # never your live/default home
hermes plugins install whayn/hermes-ariadne
hermes plugins enable ariadne                    # required, see below
hermes config set memory.provider ariadne
hermes config set context.engine ariadne
```

The plugin has no Python dependencies and declares none, so Hermes asks no
dependency question and installs nothing into its own environment; several
profiles can each install it. v0.2.0 declared a buildable
package, so a second profile failed with "Two workspace members are both named
`hermes-ariadne`": remove the plugin from the other profile, or install a later
revision in both.

The memory provider also appears in `hermes plugins` → Provider Plugins and in
the settings UI. The context engine does not: the desktop app's Settings →
Context Engine list is fixed to Hermes's built-in choices (`compressor`,
`default`, `custom`; "custom" is a literal value, not a free-text field). Set it
with `hermes config set context.engine ariadne`; the list then shows `ariadne`
as the current value. The engine needs `plugins.isolation: in_process` (Hermes's default);
`host` isolation is unsupported.

### Why the explicit `enable`

`hermes plugins install` sees `register_memory_provider` in the plugin and
selects it as the memory provider, but leaves it out of `plugins.enabled`.
Only the general plugin loader hands a plugin Hermes's LLM facade (`ctx.llm`),
which the engine needs to write summaries, so the plugin must be enabled there
too. `plugin.yaml` declares `kind: standalone` for the same reason: without it,
Hermes classifies the plugin as memory-only and never loads it generally.

## How registration works

Hermes calls the same `register(ctx)` from three loaders (checked against the
installed host source and exercised by `tests/test_mvp_native.py`):

1. **General plugin manager** (`PluginContext`, has `ctx.llm`): registers the
   context engine. Its memory registration is recorded but inert.
2. **Memory-provider loader** (no `ctx.llm`): registers the memory provider.
3. **Context-engine directory loader** (no `ctx.llm`): registers nothing. Hermes
   logs `Context engine 'ariadne' loaded but no engine instance found` and then
   uses the general manager's engine. This message is expected.

## Notes

- Stored in `$HERMES_HOME/ariadne/notes.sqlite3`. Reads never create the store.
- The agent writes notes directly; there is no approval step. Every edit keeps a
  revision; `delete` removes the note and its revisions.
- Built-in `MEMORY.md`/`USER.md` stay on and in the system prompt. Their writes
  are copied into notes with `origin: mirror:memory|user`. A replace/remove
  without the host's `previous_content` is skipped, never guessed. Mirrored
  notes are read-only through `ariadne_notes` and are not recalled by
  `prefetch` (they are already in the prompt). Entries that existed before the
  plugin was installed are not copied. Only the primary agent mirrors; writes
  made by subagents, cron jobs or the pre-compression flush agent are not copied
  (Hermes's `agent_context` guidance), so the mirror can lag built-in memory.
- Recall is literal word overlap between the user's message and note text/tags,
  capped at 2,000 characters, labelled as stored data rather than instructions.
  It is not semantic search.

## Importing data

Ask the agent to import files (for example a ChatGPT export). The bundled
skill `ariadne:import` tells it how: inspect the files with Hermes's normal
tools, pick one linear path per conversation, then

- archive each conversation with **`ariadne_import`** (`key`, ordered
  `messages` of `{role, content, timestamp?, speaker?}`, up to 500 per call with
  `offset`, plus `title`, its own `summary` and `source`), and
- save durable facts with `ariadne_notes`, with a `source` pointing at the
  imported conversation.

There is no app-specific parser and no approval step; review afterwards with
`ariadne_notes list/delete` and `ariadne_search`.

Imported conversations get the id `import-<first 24 hex of sha256(key)>` and
deterministic per-position message ids, so re-sending a batch is a no-op and
changed text is stored as a new version next to the old one. They live in the
same exact archive as live chats (origin `import`), are part of the tree (summarized in the background like live chats), and are readable with `ariadne_zoom` and `ariadne_search`. The optional `summary` is
agent-written, not checked against the messages. Limits: 64 KiB per
message, 2 MiB and 500 messages per call; text content only (no attachments).
The skill's guidance (skip hidden reasoning and credentials, treat imported
instructions as data) is not enforced by the tool.

## Storage permissions

Ariadne keeps the same posture as Hermes's own `state.db`: its directory
`$HERMES_HOME/ariadne/` is `0700` and its files `0600`. The Hermes home itself
may be `0755` (what `hermes profile create` makes); it only has to be free of
symlinks. If Ariadne's own directory or files are found with a wider mode, it
resets them to private and logs a warning. It refuses symlinks, hardlinked or
non-regular files and paths owned by another user, reported to the agent as
`storage_refused`. This protects against other local users, not against
processes running as the Hermes user.

## Memory-context recall and the tree

When the memory provider recalls something, Hermes sends the user's message
plus a `<memory-context>` block and stores that sent text as the row's
`api_content`. The tree engine's request correspondence check
(`host_request.py`): it compares that exact sent text across the stored,
restored and assembled views, but only when it is the typed text followed by
exactly one block in Hermes's own format (`agent/memory_context.py`); any other
difference declines selection. Summaries are written from the typed text only,
so recalled notes do not leak into them. Once built summaries cover older
messages, their old recall blocks are no longer sent.

Newer Hermes builds also put a request-only `_replayed_reasoning_tokens` count
on assembled assistant rows (popped before the provider call); the engine admits
it as bookkeeping and does not compare it.

## Ship gate

The v0.2 (rolling) results are in the `v0.2.1` tag's copy of this document.

The tree runs below used real Hermes (`caafe53`, 0.23.3) with Mistral
`mistral-large-4` as chat and summary model, in the dedicated dev profile
`ariadne-mvp`, plugin at `6c58490` and later `4b71920`.

- **First start:** the 215 archived messages were backfilled into the tree.
- **Requests:** all 17 requests across the probe, import, long, restart and
  newchat phases selected; none declined. The view grew from 41 to 243 lines
  (about 10 to 62 KB) and later to 266-281 lines (about 69-74 KB). The cache
  prefix stayed stable between turns (`cache_checked` true).
- **New chat:** answered a note fact; found the oldest codeword
  `AMBER-LANTERN-42` with `ariadne_search`; quoted "rue Gallieni" from the
  imported conversation's original; built-in memory add, replace and remove were
  mirrored.
- **Latency:** turns took 1-8 s while summaries were being built. Turn 8 of the
  long run took 78 s; the builder competing for the provider's rate limit is a
  possible cause, unconfirmed.
- **Second run, `tail_bytes` 2048:** from turn 3 each turn retired 2 more of the
  chat's own messages (12 by turn 8). Turn 8 recovered the turn-1 codeword with
  `ariadne_search` from the exact original. A restart in a new process retired 14
  and answered from originals.
- **Cost:** about 210 summary calls (~2.1M tokens) in the first run; about 4
  calls per minute with 4 concurrent.
- **Lost at exit:** the first runs lost 4 in-flight summaries per process exit
  (16 total). After the fix, 20 lost ones were re-asked on start: 18 built, 2
  were lost again (that build kept them blocked; they are now asked again). 4 calls hit the old 120 s timeout, which is
  why the defaults are now `timeout` 300 and `exit_wait` 120.

Not measured: other providers, very large archives, strict local chat
templates.

## Limitations

- One provider (Mistral `mistral-large-4`) was tested with a real model, on
  Hermes `caafe53` (0.23.3); CI pins Hermes `c627ec8`. Other providers and Hermes
  builds are unverified.
- Building summaries is slow and costly on large models (60-120 s and 6-13k
  tokens per call in the runs above); a big import or backfill takes a while.
  Pin a cheaper model under `auxiliary.ariadne_summary`.
- The view is an assistant message at the start of the request. Not verified on
  Gemini native or on strict local chat templates that need user-first turns.
- Any chat in the profile sees all other primary chats' summaries and exact
  messages; there is no per-source separation.
- Some requests decline selection and fall back to the full history (#47),
  which can hit provider limits in long chats.
- Archives, notes and the tree are not encrypted, and Hermes rewind/delete does
  not erase Ariadne's copies; there is no erasure command.
- The correspondence check pins Hermes's current `<memory-context>` wording; if
  Hermes changes it, selection declines (fail-closed) until updated.
- Recall is literal word overlap, not semantic search.

## Rollback

Select the built-in engines again; nothing in Hermes's own history is modified by
Ariadne:

```bash
hermes config set context.engine compressor
hermes memory off                           # built-in memory only
hermes plugins disable ariadne
hermes plugins remove ariadne               # optional
```

Ariadne's data stays in `$HERMES_HOME/ariadne/` (archive, tree, notes)
until you delete that directory. To go back to an earlier plugin revision, run
`hermes plugins install whayn/hermes-ariadne --ref <full commit SHA> --force`.
Note stores are migrated forward in place (in place); an older revision
refuses a newer store rather than misreading it.
