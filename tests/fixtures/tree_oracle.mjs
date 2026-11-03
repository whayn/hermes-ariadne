// Copyright 2026 Higher Order Company
// SPDX-License-Identifier: Apache-2.0
// Differential fixtures for the Python adaptation of UniiChat Memory.ts.
// Exact source: VictorTaelin/UniiChat at
// 0500bf6a8ea4bf0ce047488562022b1b6532cc30, Memory.ts (Apache-2.0).
// Run with Node 22.18 (tested; Node 26 removed transform mode):
// node tests/fixtures/tree_oracle.mjs /scratch/pinned/Memory.ts /scratch/tree.json
// Obtain ONLY the public pinned source, e.g. with curl -fL using:
// https://raw.githubusercontent.com/VictorTaelin/UniiChat/0500bf6a8ea4bf0ce047488562022b1b6532cc30/Memory.ts
// The input file is required and its SHA256 is checked. No Uniichat home, app
// configuration, imports, model, or network is accessed. Synthetic text only.
// Extraction: remove ONLY top-level import lines; transform the ENTIRE remaining
// source with node:module.stripTypeScriptTypes(mode:'transform'); import data URL.
// Runtime adaptations: save is a no-op; read throws (filesystem safety guard).
// fit runs unchanged with an inert second view ending at zero. Each step rebuilds
// mem.nodes from all supplied ready texts (no eviction in our API) and recomputes
// the incoming size via original Memory.size, equivalent to source store updates.
// Expected states come from ORIGINAL reach/fit/merge; render output comes from
// ORIGINAL Memory.view, with only its <chat> wrapper and trailing LF removed.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

const sha = "0500bf6a8ea4bf0ce047488562022b1b6532cc30";
const sourceSha256 = "a243258fa3be6a6a593709dc5009baaf9d9c6fc11db7758d1d80914e71289389";
const [sourceFile, outputFile] = process.argv.slice(2);
if (!sourceFile || !outputFile) throw new Error("pass pinned Memory.ts and output JSON paths");
const source = readFileSync(sourceFile, "utf8");
if (createHash("sha256").update(source).digest("hex") !== sourceSha256) {
  throw new Error("Memory.ts does not match the pinned source hash");
}
const code = stripTypeScriptTypes(source.replace(/^import .*;\r?\n/gm, ""), { mode: "transform" });
const { Memory } = await import("data:text/javascript;base64," + Buffer.from(code).toString("base64"));
Memory.save = () => {};
Memory.read = () => { throw new Error("oracle must not read a Uniichat home"); };

const leaves = (n, text = "abcdefghij") => Array.from({ length: n }, (_, i) => [0, i, text]);
const parents = (level, n, text = "p") => Array.from({ length: n }, (_, i) => [level, i, text]);
const scenarios = [
  { name: "append_prefix_stable", high: 128000, nodes: leaves(8), steps: [1, 2, 3, 4, 5, 8].map(end => ({ end, next_message: true })) },
  { name: "pending_leaves_and_parents", high: 20, nodes: [], steps: [
    { end: 4 }, { end: 4, add: [[0, 0, "😀😀"], [0, 2, "abcdefghi"]] },
    { end: 4, add: [[0, 1, "éééé"], [0, 3, "abcdefghij"]] },
    { end: 4, add: [[1, 1, "late"]] }, { end: 4, add: [[1, 0, "first"], [2, 0, "all"]] },
  ] },
  { name: "delayed_batch_then_next_message", high: 40, nodes: leaves(8), steps: [
    { end: 8 }, { end: 8, add: [[1, 0, "123456789"]] },
    { end: 8, add: parents(1, 4, "123456789") },
    { end: 8, add: parents(2, 2, "xy") },
    { end: 8, next_message: true },
    { end: 9, next_message: true, add: [[0, 8, "new"]] },
  ] },
  { name: "background_batch_must_reach_half", high: 50, nodes: leaves(6), steps: [
    { end: 6 }, { end: 6, add: [[1, 0, "p"], [1, 1, "q"]] },
  ] },
  { name: "message_commits_partial_batch", high: 50, nodes: leaves(6), steps: [
    { end: 6, next_message: true },
    { end: 6, next_message: true, add: [[1, 0, "p"]] },
  ] },
  { name: "equal_due_prefers_older", high: 200,
    parts: [[2, 0], [2, 1], [1, 4], [1, 5], [0, 12], [0, 13], [0, 14]],
    nodes: [[2, 0, "a".repeat(100)], [2, 1, "b".repeat(100)], [1, 4, "c".repeat(10)], [1, 5, "d".repeat(10)], ...leaves(15).slice(12), [3, 0, "A"], [2, 2, "B"]],
    steps: [{ end: 15, next_message: true }] },
  { name: "mixed_level_last_message_due_not_first", high: 200,
    parts: [[2, 0], [2, 1], [1, 4], [1, 5], [0, 12], [0, 13], [0, 14], [0, 15]],
    nodes: [[2, 0, "a".repeat(100)], [2, 1, "b".repeat(100)], [1, 4, "c".repeat(10)], [1, 5, "d".repeat(10)], ...leaves(16).slice(12), [3, 0, "A"], [2, 2, "B"]],
    steps: [{ end: 16, next_message: true }] },
  { name: "truncate_inside_summary_then_append", high: 1000,
    parts: [[2, 0], [1, 2], [0, 6]], nodes: [...leaves(12), [2, 0, "old\n prefix"], [1, 2, "middle"]],
    steps: [{ end: 7 }, { end: 5 }, { end: 9 }, { end: 2 }, { end: 0 }, { end: 3 }] },
  { name: "unicode_raw_bytes_not_formatted_cost", high: 22,
    nodes: [[0, 0, "  é\n\t😀  "], [0, 1, "中\r\n   文"], [1, 0, "é😀"]],
    steps: [{ end: 1 }, { end: 2 }, { end: 2, next_message: true }] },
  { name: "exact_high_does_not_merge", high: 20, nodes: [...leaves(3), ...parents(1, 1)],
    steps: [{ end: 2 }, { end: 3 }] },
  { name: "ready_parent_with_pending_children", high: 5, nodes: [[0, 0, "abcdef"], [1, 0, "p"]],
    steps: [{ end: 2, next_message: true }] },
  { name: "empty_summaries_are_built", high: 1, nodes: [[0, 0, ""], [0, 1, "xx"], [1, 0, ""]],
    steps: [{ end: 1 }, { end: 2 }] },
  { name: "nonshrinking_ready_parent", high: 10, nodes: [...leaves(2), [1, 0, "X".repeat(30)]],
    steps: [{ end: 2 }, { end: 2, next_message: true }] },
];

// Seeded coverage of valid mixed partitions, budgets, missing nodes, truncations,
// appends and summary arrivals. PRNG generates INPUTS, never expected outputs.
let seed = 19073;
const rand = n => { seed = (seed * 48271) % 2147483647; return seed % n; };
for (let caseNo = 0; caseNo < 64; caseNo++) {
  const n = 4 + rand(29);
  const all = [];
  for (let l = 0; 2 ** l <= n; l++) {
    for (let i = 0; (i + 1) * 2 ** l <= n; i++) {
      all.push([l, i, ["é", "😀", "z", "x\n \ty"][rand(4)].repeat(rand(12))]);
    }
  }
  const initial = all.filter(() => rand(3) === 0);
  const later = all.filter(row => !initial.includes(row));
  scenarios.push({ name: `seeded_${caseNo}`, high: 1 + rand(150), nodes: initial, steps: [
    { end: rand(n + 1), next_message: !!rand(2) },
    { end: n, next_message: true },
    { end: n, add: later.filter((_, k) => k % 2 === 0) },
    { end: rand(n + 1), next_message: !!rand(2), add: later.filter((_, k) => k % 2 === 1) },
    { end: n, next_message: true },
  ] });
}

function snapshot(view) {
  return { parts: view.parts.map(p => [p.l, p.i]), size: view.size, batch: view.batch, high: view.high };
}
function lines(mem, end) {
  const rendered = Memory.view(mem, end).map(block => block.text).join("");
  const body = rendered.slice("<chat>\n".length, -"</chat>\n".length);
  return body === "" ? [] : body.slice(0, -1).split("\n");
}
let transitions = 0;
for (const scenario of scenarios) {
  const known = new Map(scenario.nodes.map(([l, i, text]) => [Memory.slot(l, i), text]));
  const view = { parts: (scenario.parts ?? []).map(([l, i]) => ({ l, i })), size: 0, batch: false, high: scenario.high };
  for (const step of scenario.steps) {
    for (const [l, i, text] of step.add ?? []) known.set(Memory.slot(l, i), text);
    const mem = { nodes: new Map(known), view, count: step.end,
      seen: { parts: [], size: 0, batch: false, high: 128000 },
      hold: [null, "paused"], ready: [[[0]], []], wait: new Set() };
    view.size = view.parts.reduce((n, part) => n + Memory.size(mem, part), 0);
    Memory.fit(mem, step.next_message ?? false);
    step.expected = snapshot(view);
    step.lines = lines(mem, step.end);
    step.render_ends = [...new Set([0, 1, Math.max(0, step.end - 1), step.end, step.end + 1])]
      .map(end => ({ end, lines: lines(mem, end) }));
    transitions++;
  }
}
const primitives = ["", "é😀中", " a \n \t b ", "a\r\nb", "a\rb", "a\u0085\n\u0085b", "a\uFEFF\n\uFEFFb", "a\u2028b", "a\ud800b", "a\ud83d\ude00b", "a\udc00\n b"]
  .map(text => ({ text, bytes: Memory.bytes(text), flat: Memory.flat(text) }));
const counts = [0, 1, 2, 3, 7, 8, 13, 31, 1000, 2 ** 30, 2 ** 31 - 1];
const rootCases = counts.map(count => ({ count, parts: Memory.roots(count).map(p => [p.l, p.i]) }));
const partCases = [[0, 0], [0, 8191], [1, 2], [7, 9], [30, 0]].map(([l, i]) => {
  const p = { l, i }; const [start, inclusiveStop] = Memory.span(p);
  return { part: [l, i], start, stop: inclusiveStop + 1, count: 2 ** l,
    children: Memory.halves(p).map(p => [p.l, p.i]) };
});
const result = { upstream: { repository: "https://github.com/VictorTaelin/UniiChat", sha, source: "Memory.ts", source_sha256: sourceSha256 },
  primitives, roots: rootCases, parts: partCases, scenarios };
writeFileSync(outputFile, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({ scenarios: scenarios.length, transitions, roots: rootCases.length, parts: partCases.length, primitives: primitives.length, output: outputFile }));
