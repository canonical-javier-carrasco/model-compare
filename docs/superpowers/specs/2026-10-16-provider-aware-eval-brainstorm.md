# Brainstorm capture: provider-aware evaluation (quantization baseline + per-task cost)

**Date:** 2026-10-16
**Status:** Brainstorm captured mid-flight; full design document **not yet written**.
This file is the resume point. Next step when picked up: write
`docs/superpowers/specs/YYYY-MM-DD-provider-aware-eval-design.md` from §3/§4,
resolve §5's open items, then builder check, then `superpowers:writing-plans`.
**Worktree:** `.worktrees/provider-aware-eval` (branch `provider-aware-eval`,
based on 0.3.3 / 8472c2e; baseline pytest 580 passed). This file is uncommitted.

## 1. Origin

Feedback received on the project (quoted):

> i couldn't find one information - are you controlling provider access to
> those that don't quantize model below model default?
>
> openrouter model score isn't being recalculated based on provider, it's one
> score for benched model under release circumstances. so you might run into
> situation where, for example, someone offers GLM-5.3-Flash at half the price
> of others, but they do it because they run it at FP4
>
> so pricing per token can sometimes be misleading
>
> or there are cases like deepseek-4.1-flash that ranks top at price-per-token,
> but because it's a sparse attention model with heavily compressed kvcache,
> natively trained at FP4, it uses sometimes upwards of 5x tokens of comparable
> models, so price-per-task suddenly isn't good anymore

Plain meaning: (a) does the tool steer/exclude providers serving below-default
quantized copies; (b) quality scores are per model, prices per provider — the
ranking mixes them; (c) price-per-token can mislead both via quantization and
via token-inefficient architectures — price-per-task is the honest metric.

Glossary (for the design doc's background section): provider = company hosting
a model on OpenRouter; endpoint = one (provider, variant, region, tier)
deployment with its own slug, price and one quantization label; quantization =
storing weights in fewer bits (FP4 = 4-bit float); model default = the
precision the maker shipped; KV cache = the model's running notes on prior
context; sparse attention = attending to a subset of prior tokens.

## 2. Verified facts (recon, 2026-10-16, live)

### 2.1 Endpoints API — the missing provider layer exists, public, no key

`GET https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints`
(verified keyless 200; docs: "List all endpoints for a model"). Per endpoint:

- `provider_name`, `tag` — tag looks like the routing slug
  (`open-inference/fp4`, `morph/fp8`, `relace`); slug shape is
  `provider[/variant][/region][/tier]`. Confirm against the model-page copy
  button during implementation.
- `quantization` — enum `int4 int8 fp4 mxfp4 nvfp4 fp6 fp8 mxfp8 fp16 bf16
  fp32 unknown`, or null. (Docs: `fp4` covers mxfp4/nvfp4; `fp8` covers mxfp8.)
- `pricing` — per endpoint, includes `prompt`, `completion`,
  `input_cache_read`, `discount`, and **`pricing.overrides`** (same
  conditional-tier structure the shipped v2 tiered-pricing code parses).
- `context_length`, `max_prompt_tokens`, `max_completion_tokens`,
  `service_tier` (flex/priority/ultrafast/null), `status`
  (EndpointStatus enum `0 -1 -2 -3 -5 -10`; 0 = live; the repo already treats
  non-zero as dead via the ZDR list), `supported_parameters`,
  `supports_tool_choice`, `uptime_last_5m/30m/1d`.
- `latency_last_30m`, `throughput_last_30m`, `perf_last_30m_by_workload` —
  null unauthenticated (key/cookie-gated).

### 2.2 Steering primitives (per-request `provider` object)

- `quantizations: ["fp8", ...]` — filter endpoints by quantization level
  before routing.
- `only` / `ignore` / `order` — endpoint slugs; base slug matches all
  variants/regions (`baseten` matches `baseten/fp4/us`), variant slug alone
  matches every region of that variant.
- `allow_fallbacks`, `require_parameters`, `data_collection`, `zdr`,
  `sort` (price/throughput/latency, optional `partition`),
  `preferred_min_throughput` / `preferred_max_latency` (percentile objects),
  `max_price`.

### 2.3 Default routing is quantization-blind and price-weighted

Documented default: load-balance across ALL endpoints, weighted by inverse
square of price (uptime-gated). Consequence: the cheapest endpoint —
disproportionately often the most quantized — receives most traffic when a
user types a bare `openrouter/<model>` id. This is the mechanism the feedback
warned about. `sort`/`order` disable load balancing.

### 2.4 Live evidence: z-ai/glm-5.3-flash

34 endpoints. Highlights (USD per 1M, in/out):

- Z.AI (the maker) — **fp8**, $0.15 / $0.50.
- A ~10-endpoint cluster at $0.15/$0.50 mixing fp8, nvfp4, unknown
  (BaseTen, CoreWeave, Modal, Crusoe, Together, Venice, Fireworks, ...).
- OpenInference — **fp4**, $0.0258 / $0.90 — cheapest routable input (~6x
  under the maker price).
- DeepInfra fp4 $0.075/$0.25; Reka unknown $0.60/$1.60 (262k ctx).
- 9 endpoints `unknown` (Relace, Reka, DigitalOcean, Makora, Fireworks x2,
  Friendli, Together, Venice).
- Status `-2` endpoints exist with attractive prices (StreamLake fp8
  $0.0435/$0.145) — endpoint status filtering is mandatory.

**Catalog price is not an executable price:** `/api/v1/models` top-level
pricing for this model is $0.0184/M input (below every routable endpoint;
cheapest routable input is $0.0258) and $0.90/M output (matches OpenInference
exactly). Semantics unresolved — follow-up needed (promo/discount artifact?).
Regardless, endpoint prices must become the ranking's ground truth.

### 2.5 Quantization spread across models (5-model sweep)

No provider served two distinct quantizations of the same model in the sample;
one provider can run several endpoints (regions/tiers) but they shared one
quantization in every observed case. Unknown prevalence:

- `anthropic/claude-haiku-5.5`: 10/10 endpoints `unknown` (closed model —
  quantization not disclosed/not applicable).
- `qwen/qwen3.7-flash`: 2/2 unknown.
- `deepseek/deepseek-v4-pro-0813`: 23 live — 8 fp8, 3 fp4, 1 mxfp4, 11 unknown.
- `meta-llama/llama-3.3-70b-instruct`: 11 live — 4 fp8, 1 bf16, 1 fp16,
  5 unknown.
- `deepseek/deepseek-r1`: 1 live endpoint, fp8.

### 2.6 Per-task cost data: OpenRouter session-cost dataset

`GET https://openrouter.ai/api/v1/datasets/session-cost`. Verified keyless
401 ("No cookie auth credentials found") — **any OpenRouter API key works**.
Weekly refresh, 30-day window, CC BY 4.0 (attribution required). Query params:
`app_slug` (published harness, e.g. `hermes-agent`), `model` (exact
permaslug), `turn_range` (`1-turn` | `2-9-turns` | `10-49-turns` |
`50-plus-turns`), `limit` (1-500), `offset` (0-5000). Cells:
`{app_name, app_slug, median_session_cost_usd, model_permaslug, turn_range}`;
`meta: {as_of, version, window_days, window_end_date}`. Example row:
hermes-agent x anthropic/claude-4.8-opus x 10-49-turns = $1.74.

Semantics: median USD spend of real sessions through published harnesses —
session-length mix, retries, failures included. NOT a controlled task. But it
has OpenRouter's real routing (fp4 traffic included) baked in, so over time it
answers "does cheap quantized routing actually come out cheaper per job?".
Coverage is partial (models used via published harnesses only). No comparable
alternative exists: AA publishes per-token prices, not per-task costs;
self-run harness measurement / plugin-accumulated local stats = roadmap only.

### 2.7 Repo state

- Tiered pricing (`pricing.overrides`) **already shipped**: catalog
  `schema_version` 2, effective-tier pricing, `schedules` index; release 0.3.3;
  site cadence is now **3h** (DESIGN.md still says 6h — docs lag, fix
  opportunistically).
- Provider-aware work therefore lands as catalog `schema_version` **3**.
- Endpoint pricing carries its own `pricing.overrides`, so the shipped
  `effective_pricing` machinery re-applies per endpoint almost unchanged.
- Benchmarks API (List Benchmarks) aggregates AA, Design Arena, OpenRouter
  tau-bench/GPQA/web-search — quality-side context for later; not needed for v1.

## 3. Decisions (made in brainstorm, 2026-10-16)

- **D1 — Steering is a user choice; both modes ship.** model-compare supports
  "know and show" (rank/warn) and steering (emit data the caller can use to
  control routing). `model-compare-opencode-plugin` vendors this script and
  must be able to steer. Ranking and publishing logic must treat providers as
  first-class (they currently don't at all).
- **D2 — Default precision definition.** Default = the quantization on the
  model maker's own endpoint (provider == model author), when known. Fallback
  (user amendment): **best available** — NOT most common — among live
  (status-0) endpoints, using a pinned precision ladder. Draft ladder:
  fp32 > bf16 ~= fp16 > fp8 ~= mxfp8 > fp6 > int8 > fp4 ~= mxfp4 ~= nvfp4 >
  int4; `unknown` never counts. Exact ordering/tie-breaks to pin in the design
  doc.
- **D3 — Baseline eligibility.** Known quantization >= default -> eligible.
  Maker/first-party endpoints with `unknown` -> eligible (otherwise closed
  models — claude-haiku-5.5 is 10/10 unknown — would vanish from the baseline).
  Third-party `unknown` -> NOT eligible (fail-closed: unverifiable, might be
  int4; the Relace case on glm-5.3-flash). First-party detection heuristic
  (author-name match plus an official-partner list such as Bedrock/Azure/
  Vertex for closed labs) to be specified in the design doc.
- **D4 — How quantization enters the score: option 1 (chosen).** Rank models
  at **default-precision (baseline) endpoint prices**; quantized and
  third-party-unknown endpoints become annotations: per-model price spread,
  cheapest-quantized price, risk flag; per-endpoint table on the site;
  CLI/plugin emit steering data. This alone fixes the ranking — a quantized
  cheap vendor can no longer drag a model's ranked price down.
  - Option 2 — routing-aware expected price (documented inverse-square
    weighting across endpoints) — **document in the roadmap, not v1**.
  - Option 3 — fixed quality haircut per quantization level — **rejected**
    for v1 (invented numbers); revisit only if real data justifies.
- **D5 — Per-task cost: separate published dimension, outside the weighted
  score.** Source: session-cost dataset (§2.6). Site column/section + additive
  catalog block carrying `as_of`/window metadata and CC BY 4.0 attribution.
  Rationale: partial coverage, weekly jitter, and the repo's byte-identical
  determinism guarantees cannot survive a noisy third-party median inside the
  score. Key handling: new optional `--openrouter-api-key` / `OPENROUTER_API_KEY`
  env (publish workflow: GitHub Actions secret); keyless -> graceful absence
  (AA precedent). Roadmap: optional `--priority value` (quality / median
  session cost) once the data proves stable over a few months.
- **D6 — Sequencing.** Builds on shipped v2; catalog schema v3; tiered
  pricing logic re-applies per endpoint.

## 4. Design shape (v1, implied by D1-D6)

- Fetch per-model endpoints for candidates; derive per-model: default
  precision (D2), baseline endpoint set (D3), baseline price = the price used
  by ranking (blended per existing `--input-share` convention, tiered per v2
  machinery applied to the endpoint's own `pricing.overrides`), plus
  annotations over non-baseline endpoints.
- Catalog v3: additive per-model provider/endpoint block (endpoint slug/tag,
  provider, quantization, prices, status, context, service tier), default
  precision + baseline set, quantization annotation fields; `session_cost`
  block (D5); schema_version 3 (meaning changes on existing scalar pricing ->
  bump required; coordinate with tokens.canonical.com).
- CLI: prices/ranking from baseline endpoints; new annotation column(s) or
  notes; steering data via `--json` (additive) and/or a flag.
- Site: per-endpoint table (expandable pattern exists), quantization badges,
  per-task cost section.
- Plugin: consumes steering output; applies `provider` object per request.

## 5. Open items (proposed but unconfirmed, or unresolved)

- **No-baseline-eligible endpoint** handling (e.g. an open model where the
  maker is absent and all third parties are `unknown`). Proposal: drop from
  ranking with an explicit drop reason (fail-closed, `schedule` precedent),
  listed in `filtered`. NEEDS USER CONFIRMATION.
- Catalog top-level price semantics (§2.4 mystery) — investigate; endpoint
  prices become ground truth regardless.
- Exact precision ladder ordering and tie-breaks (D2).
- First-party/partner detection list (D3).
- Quality stays per-model (AA index at release precision); no per-endpoint
  quality source exists — document the residual mismatch explicitly.
- Fetch strategy/cost: one request per candidate (~69 ZDR candidates) per
  refresh at 3h cadence; ~50KB/model; cache integration (CLI 6h TTL, publish
  primes caches); fail-soft vs fail-closed policy for the new fetch.
- ZDR x baseline interaction (model whose only ZDR endpoints are
  below-default quantized); `:free`/`:batch` endpoint variants; status
  filtering (status -2 traps, §2.4).
- Display specifics (CLI column layout, site sections), steering output JSON
  shape, plugin consumption contract.
- DESIGN.md 6h -> 3h cadence docs lag (unrelated, opportunistic fix).

## 6. Resume checklist

1. Read this file; re-verify §2 anchors if stale (endpoints API, session-cost).
2. Settle §5 items with the user (first: no-baseline handling).
3. Write the design doc per the brainstorming skill ("The Written Design"):
   `docs/superpowers/specs/YYYY-MM-DD-provider-aware-eval-design.md`.
4. Builder check via subagent (`builder-check-prompt.md`), one question round,
   update doc, hand over.
5. Then `superpowers:writing-plans` for the implementation plan.
