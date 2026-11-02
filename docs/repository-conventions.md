# Repository conventions: sources and decisions

These are repository-management references, not claims that Ariadne implements
any other project's memory engine. Snapshots below were inspected when setting
up the public repository; upstream policy can change.

## Sources

- **Hermes Agent** (`d9397d7b559f75e60395af4e5f1a1f593ee8358c`):
  [CONTRIBUTING](https://github.com/NousResearch/hermes-agent/blob/d9397d7b559f75e60395af4e5f1a1f593ee8358c/CONTRIBUTING.md),
  [AGENTS](https://github.com/NousResearch/hermes-agent/blob/d9397d7b559f75e60395af4e5f1a1f593ee8358c/AGENTS.md),
  [plugin guidance](https://github.com/NousResearch/hermes-agent/blob/d9397d7b559f75e60395af4e5f1a1f593ee8358c/plugins/AGENTS.md).
  Adopt conventional titles/branches, search-before-work, tests and the standalone
  plugin boundary. Its core test runner is not Ariadne's test runner.
- **T3 Code** (`ac8dce0cc3abd8a337f2d806b5082a6834b9b81a`):
  [CONTRIBUTING](https://github.com/pingdotgg/t3code/blob/ac8dce0cc3abd8a337f2d806b5082a6834b9b81a/CONTRIBUTING.md),
  [PR template](https://github.com/pingdotgg/t3code/blob/ac8dce0cc3abd8a337f2d806b5082a6834b9b81a/.github/pull_request_template.md).
  Adopt one-problem PRs and concrete verification, including what was not checked.
  Do not import its restricted-contribution policy, model exclusions or approval
  bureaucracy into a small new plugin.
- **hermes-lcm**, an independently maintained Hermes extension
  (`a13e709c25476fa9dc145c61f733f1f4a0443a05`):
  [CONTRIBUTING](https://github.com/stephenschoettler/hermes-lcm/blob/a13e709c25476fa9dc145c61f733f1f4a0443a05/CONTRIBUTING.md),
  [PR template](https://github.com/stephenschoettler/hermes-lcm/blob/a13e709c25476fa9dc145c61f733f1f4a0443a05/.github/PULL_REQUEST_TEMPLATE.md),
  [CI](https://github.com/stephenschoettler/hermes-lcm/blob/a13e709c25476fa9dc145c61f733f1f4a0443a05/.github/workflows/ci.yml).
  Adopt lightweight issue-first design discussion, correctness-first review and
  explicit scope/compatibility limits. Its stub-based test coverage is not a
  substitute for Ariadne's real-Hermes discovery/restart tests.

## Our deliberate choices

- Conventional Commits, optional scope, 72-character subjects.
- Focused `type/slug` branches, optional issue number; one `main` branch.
- Squash-only merges, required CI, resolved review threads, no forced self-review.
- A small label taxonomy and two issue forms rather than a large triage system.
- Evidence-first PRs, responsible AI assistance and synthetic-only public fixtures.
- Apache-2.0 licensing. Hermes Agent remains MIT-licensed and UniiChat Apache-2.0;
  importing actual source in future requires retaining the applicable notices.
- No claim of perfect memory, production readiness, or universal compatibility.
- No publishing workflows, package releases or deployment from ordinary PRs.

The upstream projects sometimes duplicate checklists or contain differing commands
between templates and guides. Here the PR template is the submission checklist;
CONTRIBUTING explains policy and links to it rather than copying the whole form.
