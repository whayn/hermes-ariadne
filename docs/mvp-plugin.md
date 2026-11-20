# Ariadne plugin (MVP, issue #41)

One Hermes plugin, `ariadne`, that you select in **both** settings slots:

| Slot | Setting | What it does now |
|---|---|---|
| Memory provider | `memory.provider: ariadne` | Profile notes (`ariadne_notes`), one-way mirror of built-in memory writes, recall of relevant notes and past conversations each turn, `ariadne_conversations` to read past chats |
| Context engine | `context.engine: ariadne` | The rolling engine (automatic summary of older turns through Hermes's LLM facade, page/search/zoom tools) plus the conversation index |

Status: **experimental; use a dedicated development profile**. Phases 1, 3 and
4 of #41. The MVP keeps rolling's single cumulative summary with exact zoom; the
UniiChat summary tree is deferred until after the MVP. The dev-profile ship gate
with a real model passed once (see [Ship gate](#ship-gate)).

## Install (development profile)

```bash
export HERMES_HOME=/path/to/dev-profile          # never your live/default home
hermes plugins install whayn/hermes-ariadne --yes-deps
hermes plugins enable ariadne                    # required, see below
hermes config set memory.provider ariadne
hermes config set context.engine ariadne
```

Both choices also appear in `hermes plugins` → Provider Plugins and in the
dashboard. The engine needs `plugins.isolation: in_process` (Hermes's default);
`host` isolation is unsupported. Do not also run the development
`ariadne-install` in the same profile: it targets the same
`$HERMES_HOME/plugins/ariadne` directory with the old passive engine.

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

## Past conversations

The engine keeps a small index in the same private store
(`conversations` table): each archived conversation's first user message as a
preview, its Hermes title when available, and the latest summary rolling
published. It is derived navigation data; the exact messages stay in
`$HERMES_HOME/ariadne/events.sqlite3`, and index failures never affect the chat.

- **First turn of a chat:** recall adds the 8 most recently updated other
  conversations. **Every turn:** up to 3 other conversations whose title,
  preview or summary shares words with the message (searched among the 50 most
  recent). Capped at 1,500 characters, labelled as historical data.
- **`ariadne_conversations`:** `list`, `search` (literal, over the index and
  every archived message of other conversations), `open` (summary/metadata),
  `read` (exact archived messages in first-observed order, 4,000 characters per
  message) and `zoom` (one full message, paged by characters). The archive is
  opened read-only and never created by these tools; payloads are rehashed
  against their ids and reasoning/replay fields are left out of results.
- Edited messages appear as separate versions with the same `message_uid`.
- Every chat that runs with the engine is indexed, including delegated
  subagent and cron sessions. Conversations from before the plugin was
  installed are not indexed.

Evidence: `tests/test_mvp_native.py` installs the plugin with the real
installer, runs chat A (the agent saves a note through the real tool loop),
then a new chat B in another process against a loopback provider. Chat B's
request contains the note and chat A's index entry, and its
`ariadne_conversations` read returns chat A's exact original message. The tools
were exposed directly in that configuration; the deferred `tool_call` path is
not exercised. This is plumbing evidence with a scripted provider, not a claim
about how well a real model uses the recall.

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
`ariadne_notes list/delete` and `ariadne_conversations list/open`.

Imported conversations get the id `import-<first 24 hex of sha256(key)>` and
deterministic per-position message ids, so re-sending a batch is a no-op and
changed text is stored as a new version next to the old one. They live in the
same exact archive as live chats (origin `import`), appear in the index and in
recall, and are readable with `ariadne_conversations`. The `summary` is
agent-written index text, not checked against the messages. Limits: 64 KiB per
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
processes running as the Hermes user. The development engines
(`ariadne-install`) keep their stricter rule requiring a `0700` home.

## Recall and summaries together

When the memory provider recalls something, Hermes sends the user's message
plus a `<memory-context>` block and stores that sent text as the row's
`api_content`. The engine compares that exact sent text across the stored,
restored and assembled views, but only when it is the typed text followed by
exactly one block in Hermes's own format (`agent/memory_context.py`); any other
difference still declines selection. Summaries are written from the typed text
only, so recalled notes do not leak into them. On summarized turns the old
recall blocks are no longer sent.

Newer Hermes builds also put a request-only `_replayed_reasoning_tokens` count
on assembled assistant rows (popped before the provider call); the engine admits
it as bookkeeping and does not compare it.

## Ship gate

`scripts/mvp_ship_gate.py` drives real Hermes (`AIAgent`, `SessionDB`, the
profile's provider) in a marked disposable profile, one process per phase, and
only observes engine hooks. Run of 2026-11-19: plugin installed with
`hermes plugins install whayn/hermes-ariadne --ref b274bb4` into a fresh `0755`
profile, Hermes `ae2a81c` (not the CI pin), Mistral `mistral-large-4` with
`reasoning_effort: low` as both chat and `ctx.llm` route, rolling
`trigger_bytes=1024`, `max_delta_exchanges=2`, `max_logical_jobs=12`.

| Phase | Result |
|---|---|
| Import | Skill loaded; both synthetic conversations archived (abandoned branch skipped) with summaries and sources; existing notes found, not duplicated. 19 s, 7 requests |
| Long chat (8 turns) | Summaries published after turns 3, 5, 7; requests from turn 4 on carried the summary view. Turn 8 asked for the turn-1 codeword: the model used `ariadne_find` + `ariadne_zoom` on this conversation and quoted it exactly |
| Restart (new process) | Adopted the published view, selected, published again; answered the turn-2 sensor names from `ariadne_find` |
| New chat | Answered an imported fact from notes and an earlier chat's codeword through `ariadne_conversations` |
| Built-in memory | `memory` tool write mirrored as a `mirror:user` note |

Latency: `select_context` 13–56 ms per request. Summary preparation runs
synchronously in turn completion and added 6–43 s to the turns that triggered
it (the user waits for it before the next turn starts). Not measured: large
chats near the default limits (#19), other providers.

Rerun on 2026-11-20 after the review fixes, installed at `0cb7ec2` into the same
profile (its note store was still at pre-release schema version 1 and migrated to
3 in place): import (existing notes found, not duplicated), long chat (summaries
after turns 3/5/7, selected from turn 4; turn 8 quoted the run's codeword from
the original via `ariadne_find`), restart (answered both sensor names from the
originals), new chat (imported fact from notes, earlier codeword via
`ariadne_conversations`, run-unique built-in memory write mirrored). One
mid-turn request in the import turn was declined with "missing or duplicate
logical identity" while no summary existed yet; Hermes sent the original
request. The stored rows had unique identities; the in-memory message lacking
one was not captured.

Observed confusion: in one run the past-conversation index showed earlier runs'
summaries containing a similar codeword, and the model briefly described them
as a lossy summary of the current chat. The answer was still taken from the
exact original.
