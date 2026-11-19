# Ariadne plugin (MVP, issue #41)

One Hermes plugin, `ariadne`, that you select in **both** settings slots:

| Slot | Setting | What it does now |
|---|---|---|
| Memory provider | `memory.provider: ariadne` | Profile notes (`ariadne_notes` tool), one-way mirror of built-in memory writes, relevant notes recalled each turn |
| Context engine | `context.engine: ariadne` | The rolling engine (automatic summaries of older turns through Hermes's LLM facade, page/search tools) |

Status: **experimental; use a dedicated development profile**. Phase 1 of the
plan in #41. Not yet implemented: the UniiChat summary tree and zoom for the
engine (phase 2), the past-conversation index in new chats (phase 3), and the
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
