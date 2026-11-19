# Ariadne plugin (MVP, issue #41)

One Hermes plugin, `ariadne`, that you select in **both** settings slots:

| Slot | Setting | What it does now |
|---|---|---|
| Memory provider | `memory.provider: ariadne` | Profile notes (`ariadne_notes`), one-way mirror of built-in memory writes, recall of relevant notes and past conversations each turn, `ariadne_conversations` to read past chats |
| Context engine | `context.engine: ariadne` | The rolling engine (automatic summary of older turns through Hermes's LLM facade, page/search/zoom tools) plus the conversation index |

Status: **experimental; use a dedicated development profile**. Phases 1 and 3
of #41. The MVP keeps rolling's single cumulative summary with exact zoom; the
UniiChat summary tree is deferred until after the MVP. Not yet implemented: the
import skill (phase 4).

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

- Stored in `$HERMES_HOME/ariadne/notes.sqlite3` (directory `0700`, file `0600`;
  symlinks, shared permissions and foreign databases are refused). Reads never
  create the store.
- The agent writes notes directly; there is no approval step. Every edit keeps a
  revision; `delete` removes the note and its revisions.
- Built-in `MEMORY.md`/`USER.md` stay on and in the system prompt. Their writes
  are copied into notes with `origin: mirror:memory|user`. A replace/remove
  without the host's `previous_content` is skipped, never guessed. Mirrored
  notes are read-only through `ariadne_notes` and are not recalled by
  `prefetch` (they are already in the prompt). Entries that existed before the
  plugin was installed are not copied.
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
