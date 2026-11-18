# Agent-driven memory ingestion (experimental)

The optional `ariadne-library` plugin lets Hermes turn supplied material into
source-linked notes. The agent reads and interprets the material using its existing
tools. Ariadne checks and stores the proposed notes and selected source bytes; it
does not parse any application's export, call a model, or replace the active
ContextEngine. Chat exports are just one possible input.

## Try it in a disposable profile

Use a private, existing development profile on Linux (profile directory mode
`0700`). Use a Hermes build that supports native general plugins, plugin skills and
slash commands; the integration test exercises the pinned host in the CI workflow.
Review the Ariadne revision before installing native plugin code. In that profile,
set `ARIADNE_REVISION` to the reviewed commit SHA, then use the normal Hermes Git
plugin installer:

```sh
HERMES_HOME=/absolute/dev-profile hermes plugins install whayn/hermes-ariadne --ref "$ARIADNE_REVISION"
HERMES_HOME=/absolute/dev-profile hermes plugins enable ariadne-library
```

Use a revision containing the root `plugin.yaml` for `ariadne-library`. Restart the
profile's agent/gateway or start a fresh session to see the `ariadne_memory` toolset
and `ariadne-library:ariadne-import` skill. The normal Git installer is exercised
in tests; Desktop installation and host isolation mode are not verified. Do not
change an existing profile's isolation policy or switch its context engine for
this experiment.

1. Supply a small local file and ask Hermes: “Use the Ariadne import skill to
   remember the decisions and preferences in this file. Show me what you'd keep.”
2. Hermes inspects the file and proposes notes tagged as quotations, summaries or
   inferences. Check what's retained, what's skipped, and which ranges back each
   note. A draft already stores the selected bytes privately on disk, but no draft
   notes appear in search.
3. To publish, enter `/ariadne-approve proposal:<sha256>` yourself using the exact
   reference you reviewed. The approval command is not a model tool. This is a
   workflow boundary, not a sandbox against an agent with unrestricted shell access.
4. Open a new chat in the same profile and ask Hermes to search Ariadne for a
   distinctive word from an approved note and cite its source. The library is
   looked up on demand, not appended to every prompt.

## Storage and limits

- Selected source ranges (up to 1 MiB each) are kept byte-for-byte in a separate,
  private `ariadne-library/memory.sqlite3` under the profile. The source file and
  Hermes conversation database are not rewritten by the plugin. This is not
  encryption, a full conversation migration, or automatic archive scanning.
- Approved reads return note text, classification, source references, labels and
  byte ranges. Raw bytes and filesystem paths are not returned by retrieval tools.
  An operator can recover bytes with `Library.original(source_ref)`. Hashes verify
  retained bytes, not the accuracy or completeness of the agent's interpretation.
- Exact repeats deduplicate; changed notes coexist. Search is literal and case
  sensitive, not semantic. Review/read pagination uses character offsets. Search
  offsets are stable only while approved memory is unchanged.
- Start with a few small files. The source-byte budget is 128 MiB, with at most
  1,000 proposals and 10,000 published notes. There is no per-note editor, discard
  command, or draft garbage collection yet. Disabling the plugin stops access
  without deleting the database. To reset a disposable trial, stop its Hermes
  processes and move the whole `ariadne-library` directory out of the profile.
- Do not supply credentials, hidden reasoning or restricted material. The agent's
  inspection uses normal Hermes tools and model routing, so material it reads can
  reach its configured model. Source text and derived notes are untrusted. This
  plugin does not certify sanitization, restriction compliance or prompt-injection
  resistance for arbitrary files.
- Use source and profile directories you control. Symlinks are refused, but
  hostile concurrent replacement of parent directories is outside this trial's
  threat model.
- The agent must select only files you explicitly supplied. The prepare tool
  does not independently enforce that consent: it can read and retain bounded
  ranges from other local files that Hermes can read. Approval controls note
  publication, not filesystem access or draft retention. Do not enable this
  draft in a profile with access to sensitive files.
- Quotations must occur as literal UTF-8 bytes in the selected range. Use a
  labeled summary rather than a quotation for decoded or escaped text that
  does not occur verbatim in the source bytes.

The scripted native integration test exercises Git installation, skill discovery,
Hermes tool dispatch, the registered approval handler, and read-back from a new
conversation in a fresh process. It does not establish semantic extraction quality
or real-provider recall.
