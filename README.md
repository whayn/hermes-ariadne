# Hermes Ariadne

A memory and context plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/docs),
inspired by UniiChat. It gives Hermes durable notes, recall of past conversations,
and automatic summaries of long chats that can always be traced back to the exact
original messages. It is a plugin, not a fork: no Hermes core patches are needed.

> [!WARNING]
> Experimental prerelease. Use a dedicated development profile, never your live
> or default Hermes home. See the [limitations](docs/mvp-plugin.md#limitations)
> and [rollback steps](docs/mvp-plugin.md#rollback) before installing.

## What it does

Select `ariadne` as both the memory provider and the context engine, and:

- **Notes:** the agent saves durable facts directly; you can list, edit and delete
  them. Writes to Hermes's built-in memory are mirrored one way into Ariadne.
- **New chats** recall relevant notes and an index of past conversations, and can
  read those conversations' exact archived messages.
- **Long chats** have older turns replaced by an automatic summary for each
  request. Search and zoom recover the exact originals; Hermes's stored history is
  never modified.
- **Import:** the bundled `ariadne:import` skill lets the agent turn chat exports
  you supply (any format) into archived conversations plus notes.

## Install

```bash
export HERMES_HOME=/path/to/dev-profile          # never your live/default home
hermes plugins install whayn/hermes-ariadne --yes-deps
hermes plugins enable ariadne
hermes config set memory.provider ariadne
hermes config set context.engine ariadne
```

To pin a release, add `--ref <full commit SHA>` from the
[releases page](https://github.com/whayn/hermes-ariadne/releases). The
[plugin guide](docs/mvp-plugin.md) covers why `enable` is required, storage,
importing, how recall and summaries interact, and the real-model test results.

## Status

Tested with one provider (Mistral) on short chats; other providers and very large
chats are unverified, and writing a summary can add tens of seconds to the turn
that triggers it. Archives and notes are stored unencrypted under
`$HERMES_HOME/ariadne/`, and there is no erasure command yet. Releases are
prereleases; there is no stable release or PyPI package.

## Development engines

The repository also contains earlier development engines and offline tooling.
They are not needed to use the plugin above:

- [Passive archive engine](docs/passive-engine.md): `ariadne-install`, verbatim
  snapshots and `ariadne_search`.
- [Prepared context](docs/prepared-context.md) and
  [rolling context](docs/rolling-context.md): opt-in selection engines
  ([rolling storage](docs/rolling-storage.md)).
- [Context-selection simulator](docs/context-selection-simulator.md),
  [offline memory kernel](docs/offline-memory-kernel.md) (the UniiChat tree port)
  and [native compactor](docs/native-compaction.md).
- [Inspected Hermes contracts](docs/integration-findings.md) and
  [verification notes](docs/verification.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for branches, commits, tests and release
policy; [AGENTS.md](AGENTS.md) holds guidance for coding agents. Report
vulnerabilities through [SECURITY.md](SECURITY.md), not a public issue. Use
synthetic data only in reports and fixtures, never real chat transcripts or
memory archives.

Maintained by [Whayn](https://github.com/whayn) under [Apache-2.0](LICENSE). The
UniiChat tree/view port is attributed in [NOTICE](NOTICE); repository conventions
and their references are in [repository-conventions.md](docs/repository-conventions.md).
