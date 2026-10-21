# Design

How the tool works under the hood. End-user docs live in the
[README](README.md), the site's docs in [web/README.md](web/README.md).

## Data sources

1. **OpenRouter** — `https://openrouter.ai/api/v1/models` (public, no key),
   the same endpoint used by
   [openrouterlist](https://github.com/jvrck/openrouterlist). Provides the
   catalog: per-token input/output prices, context window, listing date,
   supported parameters (used for the tool-calling check) and modality info.
2. **Artificial Analysis** — the [AA intelligence index](https://artificialanalysis.ai/models),
   a 0–100 composite of reasoning/coding/knowledge benchmarks. Obtained via:
   - **OpenRouter benchmarks (primary)**: the same undocumented frontend API
     behind the [`?discount=true`](https://openrouter.ai/models?discount=true)
     and ZDR filters also republishes the AA intelligence/coding/agentic
     indices per model (`benchmarks.aa`), keyed by exact OpenRouter id.
     Consumed first — no separate fetch, and exact keys make conflation
     impossible.
   - **AA API v2** (`artificialanalysis.ai/api/v2/language/models/free`) for
     models OpenRouter does not cover, when an API key is supplied through
     `--aa-api-key` or the `AA_API_KEY` env var
     ([free key](https://artificialanalysis.ai)); full model coverage. It
     matches to OpenRouter ids by exact slug/name only. The previous
     token-overlap fuzzy pass is gone: it paired `z-ai/glm-5.3-flash` with
     AA's `glm-5-3` entry and published the wrong model's index. Without a
     key there is no AA fallback: keyless runs get AA data only through the
     OpenRouter benchmarks, and do so silently (the catalog records the
     absence in `sources.aa.fallback`). Unmatched models simply score 0 on
     quality — the tool degrades gracefully rather than failing.

If no quality data is obtainable at all, the quality weight is dropped and the
remaining weights renormalize.

## Scoring

Each criterion is normalized to [0, 1] across the candidate pool and combined
with priority-dependent weights:

| priority  | quality | price | context | age |
|-----------|--------:|------:|--------:|----:|
| balanced  | 0.40    | 0.40  | 0.10    | 0.10 |
| price     | 0.20    | 0.60  | 0.10    | 0.10 |
| quality   | 0.60    | 0.20  | 0.10    | 0.10 |

- **price score** — prices are converted to USD per 1M tokens and blended:
  `blended = input_share × $in/M + (1 − input_share) × $out/M` (default
  `--input-share 0.75`, i.e. a 3:1 input:output mix, the convention used by
  Artificial Analysis and typical of coding-agent traffic). The score is the
  model's position on a **log-scaled** price axis within the current candidate
  pool: `1.0 − (log10(price+0.01) − log10(min+0.01)) / span`, clamped to
  [0, 1]; free listings score 1.0. Log-scaling keeps the pool's
  three-orders-of-magnitude price spread from crushing everything above the
  cheapest listing into one undifferentiated bucket.
- **quality score** — AA intelligence index ÷ `--quality-ref` (default 70),
  clamped to [0, 1].
- **context score** — logarithmic ramp from `--min-context` up to 4× that
  threshold: headroom helps, but with diminishing returns.
- **age score** — exponential decay with a `--recency-half-life` of 120 days:
  a listing half the age of another scores ~0.5× higher on this axis.
- **context filter** — hard minimum (`--min-context`, default 1M tokens),
  so a cheap small-window model can never win a big-context job.

## What gets filtered out

Below-minimum context, non-text outputs (image/audio), listings without
tool-calling support (`--no-require-tools` to relax), unparseable/negative
prices, expired listings, `:batch` variants (asynchronous completion — no
good for interactive agents; `--include-batch` to keep them), models without
a zero-data-retention (ZDR) endpoint by default (`--no-zdr` to consider
everything), and (with `--exclude-free`) rate-limited `:free` variants.

## Artificial Analysis data, in short

The intelligence index no longer needs a separate source: OpenRouter
republishes the AA indices alongside its own benchmark data, keyed by exact
OpenRouter id, and the script reads them from the frontend API it already
fetches. Only models missing there fall back to the AA API — the supported
V2 free-tier endpoint
[`/api/v2/language/models/free`](https://artificialanalysis.ai/data-api/docs)
(key required; any tier key works, free is enough; responses paginate at 200
models per page and are followed), matched by exact slug/name only. Without
a key there is no fallback and no warning: OpenRouter is the primary source.
With a key, an AA API failure or an empty result prints a warning on stderr.
Both paths are cached identically; models without a value rank on
price/context/age only. `--quality-ref` controls how
generous the quality normalization is.

## Caching

Responses are cached under `~/.cache/model-compare/` (`XDG_CACHE_HOME` is
honored) with a 6-hour TTL (`--cache-ttl`); `--no-cache` forces a refetch. The
cache keeps repeated invocations (e.g. in shell prompts or wrappers) fast and
polite.

## Discounts

The `DISC` column and the `--discount` filter use the same data as the
website's [`?discount=true` model filter](https://openrouter.ai/models?discount=true):
OpenRouter's frontend models API reports, per model variant, the fraction by
which the listed price is currently discounted (e.g. `75%`). The endpoint is
undocumented, so the tool degrades gracefully — if it breaks, every model
shows `--` and `--discount` returns nothing.

Ranking always uses the listed (undiscounted) prices; the discount is shown
as information. Variants are matched individually, so a `:batch` twin of a
discounted model only shows a discount when that variant itself is
discounted.

## Zero data retention

By default, only models with zero-data-retention (ZDR) endpoints are ranked —
providers that do not retain prompts or outputs. The ZDR set comes from
OpenRouter's public per-endpoint ZDR list
(`https://openrouter.ai/api/v1/endpoints/zdr`, public; its OpenAPI entry gives
the response shape but little semantics): one entry per ZDR endpoint, keyed by
the exact model id. A model id counts as ZDR if and only if it has at least
one entry in that list whose `status` is the integer `0`. OpenRouter's API
schema (`EndpointStatus` in `https://openrouter.ai/openapi.json`) lists the
values `0`, `-1`, `-2`, `-3`, `-5` and `-10` without describing them, so the
meaning of the non-zero values is unverified; any status other than `0`,
including a missing one, does not count (fail closed). The website's
`zdr=true` model filter matches this at the model level, but the list is per
variant, so non-ZDR variants of a ZDR model are excluded — NVIDIA's `:free`
endpoints, for example, train on prompts and are not listed, while their paid
twins are. The default filter therefore excludes `:free` variants without a
ZDR endpoint of their own, and `:batch` variants as well: the list currently
has no `:batch` ids, so `--include-batch` keeps nothing unless `--no-zdr` is
also given (by default `:batch` ids are dropped as `batch`; with
`--include-batch` they are dropped as `not ZDR` instead). The list and the
previous source (the `models/find?zdr=true` frontend filter) also differ in
other ids, in both directions; in a comparison on 2026-10-08, none of the ids
only the list had was a text-output model, and almost all of the text-output
ids it lacked were `:batch` or `:free` variants. If the list cannot be fetched
or yields no entries, the tool refuses to rank rather than silently
considering non-ZDR models; pass `--no-zdr` to explicitly consider everything.

## Catalog output contract

`--catalog` prints the full evaluation as one machine-readable JSON document:
every surviving candidate, ranked, plus every filtered-out model with its drop
reasons. The published site serves it as
[`catalog.json`](https://canonical.github.io/model-compare/catalog.json),
refreshed on the same 3-hour schedule as the picks.

```console
$ ./model_compare.py --catalog | python3 -m json.tool
```

The document is a **stable contract**. Downstream, `data.json` — not the
catalog — feeds internal Canonical tooling: tokens.canonical.com (the
`canonical/agent-provisioner` backend) fetches `data.json` daily and reads
only its `priorities` mapping (rows with a string `model` id, optional
`quality_index`, optional `discount`), validating nothing about
`schema_version` — a v2 `data.json` therefore needs no downstream work. The
catalog's strict consumer is the [model-compare OpenCode
plugin](https://www.npmjs.com/package/model-compare-opencode-plugin), which
rejects any catalog whose `schema_version` is not `2` and degrades gracefully
(cached older document, then no pick, with warnings); a v3-accepting plugin
release must land before or with the first v3 publish. Both facts were
investigated against the live consumer repos on 2026-10-20 — see
`docs/superpowers/notes/2026-10-20-downstream-investigations.md`. Same
inputs still produce the same output apart from `generated_at` and the age
score (`age_days`/`listed_at` use UTC date precision, but `scores.age` decays
continuously, so two runs on the same day differ slightly in `scores.age`).
`schema_version` is now `3` (v1 pinned the pricing scalars to the base tier;
v2 added the top-level `schedules` index): fields may be added without
notice, but renaming, removing, **or changing the meaning of** one bumps the
version. v3 changes the meaning of the per-model `pricing` block — see below.

Top level: `schema_version`, `tool`, `generated_at`, `parameters` (all knobs
plus the **effective** per-priority `weights` — reproducing `scores.overall`
needs nothing else), `sources` (`openrouter`, `aa` with `mode`
`openrouter`/`api`/`none`, `fallback` `api`/`none`, plus
the `matched` and `matched_openrouter` counts, `zdr` `ok`/`skipped`, `discounts`
`ok`/`unavailable` — where `unavailable` covers both a failed discount fetch
and a live pool with zero discounts, `endpoints` always `ok` — endpoint-stage
failures surface as pool drops, never as source state, `session_cost`
`ok`/`unavailable`/`skipped`), `pool` (`listed`, `candidates`,
`dropped`), `models`, `rankings`, `schedules`, `filtered`, plus an optional
`session_cost` block (present only when the per-task-cost fetch succeeded —
consumers branch on key presence, never on a null shape).

`session_cost` is the per-task cost dimension (OpenRouter's `session-cost`
dataset; a third-party median USD cost of real sessions through the pinned
published harness `app_slug = hermes-agent`). It carries `attribution`
(`"Data © OpenRouter, CC BY 4.0"` — the dataset's license, quoted verbatim),
`app_slug`, `as_of`, `window_days`, `window_end_date`, and `medians` — a
`model id → {turn range → median USD}` mapping; each model's cells are a
subset of the dataset's four turn ranges (`1-turn`, `2-9-turns`,
`10-49-turns`, `50-plus-turns`) — the ranges the dataset covers for it. The
block is
**excluded** from the weighted score: the score never reads it, the site
surfaces it as its own column, and a missing key renders as "not available",
never a zero.

`schedules` is a catalog-wide off-peak deals index built from the **raw**
OpenRouter model list, deliberately **not** filtered by the candidate pool
(ZDR, tool support, modality, expiry, context floor): today's live schedule
models mostly sit below the 1M context floor, and filtering would shrink the
deals table from three rows to two. Router aliases (`~` ids) and malformed
ids are excluded; `:batch` variants are included as distinct deals. Each
entry carries `id`, `name`, `context`, `quality` (the candidate's, else the
OpenRouter-published AA index, else `null`), `score` (the balanced overall
for candidates, else `null`), `max_discount` (4-decimal-rounded), `sched_note`,
`sched_detail`, `peak`/`offpeak` price blocks (per-1M; `offpeak` is the
cheapest window) and the full `schedule` window list. Downstream
time-weighted price for a window list: `Σ coverage × price` (per key).

`sources.aa.mode` is `openrouter` whenever at least one candidate's AA data
came from OpenRouter, and otherwise names the fallback. It says nothing about
whether the AA API worked. `sources.aa.fallback` records that on its own:
`api` when the AA API yielded entries, `none` when no key was set or the API
failed or returned nothing. It is an additive field, so it arrived without a
`schema_version` bump. Earlier releases could also report `scrape` (a JSON-LD
page-scrape fallback, now removed); that value can no longer be produced and
`web/build_site_data.py` rejects it, which needed no `schema_version` bump
because no emitted document changes. With `AA_API_KEY` set,
`web/publish.py` fails the run unless `fallback` is `api` — a rejected key
yields `none` — and
checks this right after the `--catalog` step, before any artifact is written.

`rankings` maps each priority (`balanced`, `price`, `quality`) to the full
ordered list of candidate ids for that priority (every `models` id exactly
once; rank = index + 1). The order is exactly what a `--priority P` run
prints: the unrounded weighted score, then quality descending, blended price
ascending, id. `rankings` is an additive field, so it arrived without a
`schema_version` bump. It adds no new source of
drift; the only intra-day variance remains the decaying age score.

Each `models` entry carries: `id` (bare `provider/model`), `name`,
`provider`, `family` (heuristic: leading token of the slug, e.g. `glm-5.3`
→ `glm`; `null` when there is none), `pricing` — **under v3 the
baseline endpoint's price structure in full** (see below) — `context`,
`listed_at`, `age_days`, `tool_calling`, `zdr`, `discount`, `expired`,
`quality` (AA intelligence index or `null`), `aa` (the OpenRouter-published
trio `intelligence_index`/`coding_index`/`agentic_index`, each possibly
`null`), `quality_match` (`openrouter`/`api`/`null`), `scores` —
the four component scores plus
`overall` for all three priorities, so downstream consumers never re-run
the scorer — and the v3 additive per-model blocks `endpoints`,
`default_precision`, `advertised_pricing` and `annotations`.

**The `pricing` block describes the baseline endpoint.** The baseline is the
cheapest live (status-0) baseline-eligible endpoint at the model's default
precision (ties break by provider name, then tag — deterministic; see the
precision ladder, eligibility and derivation rules in
`docs/superpowers/specs/2026-10-19-provider-aware-eval-design.md` §4.1–§4.5).
The block keeps its v2 shape — `input_per_1m`, `output_per_1m`,
`blended_per_1m` in USD per 1M tokens (the tier in effect at a prompt size of
`min_context` tokens), `base` (the baseline endpoint's base tier; for
time-windowed endpoints the deterministic peak window), `tiers` (ascending
cumulative long-context tiers) and `schedule` (all windows, peak included,
with per-window `coverage`) — but every value now comes from that endpoint's
own `pricing.overrides`, so the block stays self-consistent: the blended
price always matches the tiers and schedule displayed next to it. The models
API's top-level (advertised) scalars are **not** discarded: they move to the
per-model `advertised_pricing` block (`input_per_1m`, `output_per_1m`,
`blended_per_1m`, `source: "models-api"`), kept for comparison with
OpenRouter's website. That scalar is endpoint-derived (it exactly equals some
endpoint's base price — the default service tier where one exists, else the
cheapest listed endpoint, which may be non-routable), but OpenRouter
documents no general derivation rule, so it is **documentation-only**:
nothing may interpret it and no consumer reads it for ranking.

The other per-model additive blocks:

- `endpoints` — every endpoint from the OpenRouter endpoints API, each with
  `provider_name`, `tag`, `quantization` (a precision-ladder label,
  `"unknown"` or `null`), `pricing` (`prompt`, `completion` — USD per token —
  plus the raw `overrides` list), `context_length`, `status`
  (`0` = live; OpenRouter's `EndpointStatus` enum otherwise lists only
  negative values) and `service_tier`.
- `default_precision` — the precision the ranked price is quoted at (the
  maker's own best, else the best available in the derivation universe —
  live non-variant endpoints that pass the schedule gate); one of the
  precision-ladder labels, or `null` when no live endpoint names a known
  quantization (common for closed models — eligibility then relies solely on
  the first-party carve-out).
- `annotations` — quantization risk around the baseline, all four fields
  derived from one shared universe: the live non-variant endpoints that are
  **not** baseline-eligible (sub-default quantization, third-party
  unknown/null, schedule-gate-failed).
  - `price_spread`: cheapest non-eligible blended price ÷ cheapest eligible
    blended price (the `--input-share` blending convention on both sides);
    `null` when the universe is empty — generalized to `null` when either
    side has no priceable member — and `null` when either side's cheapest
    price is zero.
  - `cheapest_quantized`: `{provider, quantization, price}` of the cheapest
    member of the quantization-caused subset of the universe (schedule-gate
    failures are excluded even when cheaper); `null` when the subset has no
    priceable member.
  - `risk_flag`: boolean, true iff the sum of inverse-square blended-price
    weights (OpenRouter's default routing) over the universe **strictly
    exceeds** the sum over the baseline-eligible endpoints — default routing
    would send more than half its traffic to non-eligible endpoints (exact
    tie → `false`).
  - `baseline_context_below_catalog`: true iff the baseline endpoint's
    `context_length` is smaller than the catalog `context` (the models-API
    capability ceiling).
- `steering` — the OpenRouter `provider` routing object for callers that
  steer: `order` (baseline-eligible tags, cheapest first), `ignore` (every
  live non-eligible endpoint, variants included), `quantizations` (known
  quantizations ≥ default; omitted entirely when `default_precision` is
  `null` — never an empty list) and `allow_fallbacks`. The endpoint stage
  attaches it per candidate; neither the catalog document nor the `--json`
  rows publish it today — the plugin, which vendors this script, reaches it
  through its own integration (design §7). The plugin decides
  `require_parameters`/`zdr`/`data_collection` policy itself.

`filtered` entries are `{"id", "name", "reasons"}` plus optional additive
keys (`annotations` — a dropped model's stale cached annotations, marked
`"stale": true`, display-only); the reason keys are the same strings the
tool counts internally:

```
malformed id, context, pricing, schedule, free, batch, no discount, not ZDR,
modality, tool calling, expired, age, no_baseline_eligible_endpoint,
endpoint_fetch_failed
```

The `schedule` reason is fail-closed by design: a model whose
`pricing.overrides` time windows do not tile the week exactly (or whose
windows fail validation) has no deterministic price, so it is dropped
entirely rather than ranked at a fetch-time-dependent top level. The two new
endpoint-stage reasons are snake_case while the older ones are human phrases
— a deliberate, documented mixed convention: they are machine-readable keys
for the endpoint-stage drops. That registry sync is load-bearing:
`pool.dropped` is projected zero-filled from the `CATALOG_DROP_REASONS` tuple
in `model_compare.py`, which discards unknown reasons — so both the
pool-accounting invariant (`sum(dropped) + candidates == listed`, enforced by
`web/build_site_data.py`) and the publish drop-rate gate below, which read
`pool.dropped`, break if a reason is produced but not registered.

Version gates at the v3 cutover: catalog `schema_version` 3, `data.json` 2,
`history.json` 3 — each a strict identity check accepting exactly that value
and rejecting anything else. `web/publish.py` also fails the run when more
than 5% of the entered pool (`candidates` + the two endpoint-stage drops) is
dropped for an endpoint-derived reason (`no_baseline_eligible_endpoint` or
`endpoint_fetch_failed`; constant `MAX_ENDPOINT_DROP_RATE = 0.05`) — a
transient API outage or a first-party-detection miss must shrink the public
ranking loudly, never silently. The history bump resets ALL snapshot
history: prices, rank movement, the AA index and pool ids. Old snapshots
describe old price semantics (advertised models-API scalars), so mixing them
with baseline-endpoint prices would fabricate a fake price trend; no
continuity bridge is built. The site's 7-day movement arrows and the weekly
highlights diff degrade to their no-baseline state until fresh history
covers the full week: both look up the snapshot dated exactly seven days
before today, and history snapshots key on UTC dates, so that takes seven
days of fresh snapshots to accumulate.

`pool.dropped` lists all of the reason strings above zero-filled. `--top`
and `--priority` are ignored with `--catalog` (the document always covers the
full pool, sorted by the balanced overall score); `--catalog` cannot be
combined with `--best` or `--json`.

On the published site the catalog is the ranking and timestamping authority.
It carries the ranking itself in `rankings`, and everything else projects it
or is checked against it: the `data.json` table rows are reordered to it (and
the build fails if the `--json` runs ranked different models), the
`history.json` tabs and the weekly highlights diff slice it, and
`web/publish.py` fails the run unless `best.txt` is `rankings.balanced[0]`.
The CLI and the catalog share one score function and one sort key in
`model_compare.py` rather than copies of either. `web/publish.py` runs
`--catalog` first to prime the shared caches, which makes the later
invocations very likely to rank the same inputs. It is not a guarantee:
degraded fetches (an empty discount map, no AA benchmark entries) are
deliberately never cached, so a later run may fetch fresh, different data.
The publish contract is fail-loud: such a mismatch makes `order_rows` or
`web/publish.py` fail the run, and the next run recovers on its own.
`data.json` is stamped with the catalog's `generated_at` (as is the
newest `history.json` snapshot), and `web/publish.py` fails the run when the
deployed artifacts' stamps disagree.

## Tests

A `pytest` suite covers the ranking logic (`test_model_compare.py` — input
coercion, candidate filtering, scoring math, discount parsing, Artificial
Analysis matching) and the site tooling (`web/` — the data builder, the
highlights generator, the preview script and the publish orchestrator), with
no network access (external calls are stubbed). Run it from the repo root:

```console
$ pytest
```

## Release checklist

1. Bump `VERSION` in `model_compare.py`, `web/publish.py` and
   `web/generate_highlights.py` so `--version` and the user agents match,
   then update the pinned literal in
   `test_model_compare.py::test_version_flag` (it exists precisely so a
   forgotten bump fails the suite).
2. Add a `CHANGELOG.md` entry dated today.
3. Run `pytest` (collects the root and `web/` suites).
4. `git tag -a vX.Y.Z -m "model-compare X.Y.Z"`, push main + tag.
5. `gh release create vX.Y.Z` with notes linking the pinned standalone
   download:
   `raw.githubusercontent.com/canonical/model-compare/vX.Y.Z/model_compare.py`.
6. Never rename the catalog `tool` literal or remove schema fields without
   bumping `schema_version`.
