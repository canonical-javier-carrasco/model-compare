# Provider-Aware Evaluation: Quantization Baseline + Per-Task Cost — Design

**Date:** 2026-10-19
**Status:** Approved design (settled from the 2026-10-16 brainstorm §5 open items).
**Predecessor:** `docs/superpowers/specs/2026-10-16-provider-aware-eval-brainstorm.md` (D1–D6, verified API facts §2).
**Baseline:** branch `provider-aware-eval` at 0.3.3 / 8472c2e (580 pytest passing per brainstorm record; local env lacked pytest — re-verify on pickup).

## 1. Intent & Why

OpenRouter's default routing load-balances across ALL endpoints, weighted by the inverse square of price. Consequence: the cheapest endpoint — disproportionately often the most heavily quantized (FP4/INT4) — receives most traffic when someone types a bare `openrouter/<model>` id. Meanwhile:

- Catalog/model scores are **per model** (release precision); prices are **per provider/endpoint**. The shipped ranking mixes a model-level quality score with a provider-level price — misleading.
- Price-per-token is misleading both via quantization (FP4 copies at half price) and via token-inefficient architectures (e.g. sparse attention / compressed KV cache models consuming 5x the tokens — price-per-task is the honest metric).

This design makes model-compare **provider-aware**: rank models at **default-precision (baseline) endpoint prices**, surface quantization as annotations, emit steering data the caller can use to control routing, and publish per-task cost as a separate, explicitly-sourced dimension.

## 2. Goals, Non-Goals, Anti-Goals

### Goals

1. **G1 — Correct ranking ground truth.** Ranking prices come from **endpoint** prices (not catalog top-level pricing), filtered to **baseline-eligible** endpoints only (see §4 D3/D4).
2. **G2 — Baseline precision definition.** "Default" = quantization on the model **maker's own** endpoint (provider == model author) when known. Fallback: **best available** precision among live (status-0) endpoints using a pinned precision ladder. `unknown` never counts.
3. **G3 — Fail-closed eligibility.** Known quantization >= default → eligible. Maker/first-party `unknown` → eligible (so closed models don't vanish). Third-party `unknown` → **NOT** eligible.
4. **G4 — No-baseline handling.** A model with no baseline-eligible endpoint is **dropped from the ranking with an explicit drop reason**, listed under `filtered` (fail-closed, `schedule` precedent). *Settled with user 2026-10-19.*
5. **G5 — Baseline price rule.** When several eligible default-precision endpoints exist, the **cheapest** live eligible one sets the model's ranked price (tiered per its own `pricing.overrides`). *Settled with user 2026-10-19.*
6. **G6 — Annotations.** Non-baseline endpoints become per-model annotations: price spread, cheapest-quantized price, risk flag; full per-endpoint table on the site; steering data via CLI `--json`.
7. **G7 — Steering.** Both "know and show" (rank/warn) and steering (emit OpenRouter `provider` routing object: `order`/`ignore`/`quantizations` etc.). `model-compare-opencode-plugin` vendors this script and must be able to steer.
8. **G8 — Per-task cost dimension.** Source: OpenRouter `session-cost` dataset. **Outside** the weighted score; separate site column/section + additive catalog block carrying `as_of`/window metadata and CC BY 4.0 attribution.
9. **G9 — Schema v3.** Catalog `schema_version` bumps to **3** (semantics of the scalar pricing change). Coordinate with tokens.canonical.com.

### Non-Goals

- **Routing-aware expected price** (documented inverse-square weighting across endpoints) — documented as roadmap, not v1 (brainstorm D4 option 2).
- **Invented quality haircut** per quantization level (brainstorm D4 option 3) — rejected for v1.
- Per-endpoint **quality** scores — no such source exists; quality stays per-model (AA index at release precision). Residual mismatch documented explicitly (§6).
- Replacing or re-scoring the AA benchmark itself.

### Anti-Goals (what would make this a failure)

- Letting a quantized cheap vendor drag a model's ranked price down (the original complaint).
- Publishing catalog data whose meaning silently changed without a `schema_version` bump.
- Mixing a noisy third-party median (session cost) into the weighted score.
- Silently ranking a model whose precision can't be verified.

## 3. Constraints

- **Determinism:** the repo's byte-identical publishing guarantees hold. Session-cost data (weekly refresh, third-party) must never enter the weighted score.
- **Existing machinery reuse:** tiered pricing (`pricing.overrides`) logic re-applies **per endpoint** nearly unchanged (shipped in v2 / 0.3.3).
- **Key handling:** new optional `--openrouter-api-key` / `OPENROUTER_API_KEY` env (publish workflow: GitHub Actions secret). Keyless → graceful absence (AA precedent).
- **Fetch cadence:** site refresh is 3h; one request per candidate model (~69 ZDR candidates) per refresh, ~50KB/model. Cache integration: CLI 6h TTL, publish primes caches.
- **Fail policy (settled):** **hybrid** — fail-closed for ranking-critical baseline prices (model without a fetched baseline can't be ranked; drop with reason); fail-soft for annotations (stale data kept with a degraded marker). *Settled with user 2026-10-19.*

## 4. The How (decided)

### 4.1 Precision ladder (D2, pinned here — builder note: exact ordering/tie-breaks are now fixed)

```
fp32 > bf16 ~= fp16 > fp8 ~= mxfp8 > fp6 > int8 > fp4 ~= mxfp4 ~= nvfp4 > int4
```

- Ranking: `fp32` (4) > `bf16`/`fp16` (3) > `fp8`/`mxfp8` (2) > `fp6` (1) > `int8` (0) > `fp4`/`mxfp4`/`nvfp4` (-1) > `int4` (-2). *(Builder: implement as an ordered dict; equal-rank entries inside a tier share the same numeric rank. Correction, designer call 2026-10-19, second builder check: the ladder string is authoritative — `fp6 > int8` is strict, so fp6 outranks int8 numerically; the earlier draft that gave both rank 1 was a transcription error.)*
- `unknown` **never** participates in the ladder; it only routes through the maker/first-party eligibility carve-out (D3).
- **Default precision derivation:** pick the **highest-ranked** quantization among live (status-0) **non-variant** endpoints (§4.3 variant ruling) of the **maker** (matched per §4.2). If the maker has no live non-variant endpoint with a known quantization, the default is the **highest-ranked** quantization among all live non-variant endpoints (best-available fallback). If **no** live endpoint has a known quantization (common for closed models), `default_precision` is `null` and eligibility relies solely on the first-party carve-out (§4.3); a model with no first-party live endpoint is then dropped per §4.4. *(All-unknown rule: designer call, 2026-10-19. Variant exclusion in the derivation universe: designer call, 2026-10-19, second builder check — a `:free` variant must never pin the default that then governs paid endpoints.)*

### 4.2 First-party/partner detection (D3, resolved)

A provider is treated as "first-party" for eligibility purposes when **any** of:

1. `provider_name` **normalized** equals the model's author slug normalized. Normalization = casefold + strip all non-alphanumeric characters, so the live-recorded maker name `Z.AI` matches the author slug `z-ai` (both normalize to `zai`). *(Normalization pinned as designer call, 2026-10-19 — plain case-insensitive equality fails against the recorded evidence in brainstorm §2.4.)*
2. `provider_name` is in a pinned partner list of official hosting platforms for closed labs: `bedrock`, `azure`, `vertex`, `azure-ai`, `google-vertex`, `aws-bedrock`. *(Builder: maintain as a module-level constant `FIRST_PARTY_HOSTS`; extend only with evidence, e.g. the endpoint's `tag` matching `<lab>-official`. If normalization still misses a maker during implementation, add an explicit `FIRST_PARTY_ALIASES` mapping — verified against live data first.)*

Everything else is third-party.

### 4.3 Eligibility rule (D3, fail-closed)

Precedence (ZDR clarification, designer call 2026-10-19): the existing **model-level ZDR gate is unchanged** — it runs first and determines which models can be ranked at all, exactly as shipped today. Endpoint eligibility operates only within that ZDR-gated pool. Per-endpoint ZDR data is not fetched (the endpoints API does not expose it in the verified field set); steering `zdr`/`data_collection` remains the plugin caller's choice per §4.7/§7.

An endpoint is **baseline-eligible** iff:

- `status == 0` (live), AND
- it is not a variant endpoint (see rulings below), AND
- its effective price survives the schedule gate (see rulings below), AND
- quantization is known AND `rank(quantization) >= rank(default_precision)`, OR
- quantization is `unknown` **or `null`** AND the endpoint is first-party (D3 carve-out).

Rulings (designer calls, 2026-10-19):

- **`quantization: null` vs `"unknown"`:** treated identically — both are unverifiable, never enter the ladder, and route only through the first-party carve-out. Third-party null/unknown → not eligible.
- **Variant endpoints:** endpoints belonging to `:free` / `:batch` (or any zero-priced variant) are **excluded** from baseline eligibility and from the `cheapest_quantized` annotation; they appear in the per-endpoint table flagged as variants. Rationale: a rate-limited free tier is not an executable price for the harness workloads the ranking serves — letting it win "cheapest eligible" would reintroduce the misleading-price failure this design exists to fix.
- **Schedule gate per endpoint:** if the cheapest eligible endpoint's own `pricing.overrides` contains non-tiling schedule windows (the v2 `schedule` precedent), that **endpoint** is excluded from eligibility and the next-cheapest eligible endpoint is used. A model is dropped only when **no** eligible endpoint survives the schedule gate (drop reason `no_baseline_eligible_endpoint`).

Everything else (quantized below default, third-party unknown/null, variants, non-live status) is **not** eligible and becomes an annotation.

### 4.4 No-baseline drop (G4, fail-closed)

If no endpoint qualifies per §4.3:

- Model is **dropped from the ranking**.
- Added to the `filtered` list with an explicit machine-readable drop reason, e.g. `{"slug": ..., "reason": "no_baseline_eligible_endpoint"}` (same pattern as the existing `schedule`-based `filtered` precedent).
- Site/CLI still surface the model in the filtered/dropped section, not the ranking.

### 4.5 Baseline price (G5)

Among eligible endpoints: **cheapest** blended per-token price wins (existing `--input-share` blending convention; tiered pricing applied per-endpoint from that endpoint's own `pricing.overrides`). Ties broken by (1) lower provider name, then (2) endpoint slug — deterministic.

**Pricing block coherence under v3 (designer call, 2026-10-19, second builder check):** the published per-model `pricing` block (`input`, `output`, `blended`, `base`, `tiers`, `schedule`) describes the **baseline endpoint's** price structure in full — scalars and tiers/schedule from that endpoint's own `pricing.overrides` — so the block stays self-consistent (the blended price always matches the tiers displayed next to it). The model's top-level (advertised) price from the models API is **not** discarded: it is retained as an additive annotation field `advertised_pricing` (scalars only, with a `source: "models-api"` note) alongside the per-endpoint block, for comparison with OpenRouter's website. No consumer reads the advertised price for ranking.

### 4.6 Annotations (G6)

Per model, additive catalog fields (v3):

- `price_spread`: ratio between cheapest eligible and cheapest non-eligible (quantized/third-party-unknown) live endpoint.
- `cheapest_quantized`: `{provider, quantization, price}` of the cheapest live endpoint that is NOT baseline-eligible due to quantization.
- `risk_flag`: boolean, true when the default routing (inverse-square price weighting) would route predominantly to a sub-default-quantization endpoint. Universe (designer call, 2026-10-19, second builder check): computed over **live non-variant** endpoints only — variant endpoints never trigger the flag (§4.3 ruling); schedule-gate-failed endpoints **do** (they are live, executable in principle, and not baseline-eligible — a real routing risk).
- Full endpoint table (per-endpoint slug/tag, provider, quantization, prices, status, context, service tier) published to the site under each model's expandable section.
- **Context caveat (designer call, 2026-10-19):** the model-level `context_length` in the catalog keeps coming from the models API (the model's capability ceiling). The baseline endpoint's own `context_length` is published alongside the baseline price; when the baseline endpoint's context is **smaller** than the catalog context, an explicit caveat field (`baseline_context_below_catalog: true`) is emitted so the site never publishes "full context at an endpoint price" silently.

### 4.7 Steering output (G7)

`model_compare.py --json` gains an additive `steering` block per model:

```json
{
  "model": "z-ai/glm-5.3-flash",
  "provider": {
    "order": ["Z.AI/fp8"],
    "ignore": ["open-inference/fp4", "deepinfra/fp4"],
    "quantizations": ["fp8"],
    "allow_fallbacks": true
  }
}
```

Rules: `order` lists baseline-eligible endpoint tags (cheapest first); `ignore` lists **every live endpoint that is not baseline-eligible** — sub-default quantization *and* third-party unknown (designer call, 2026-10-19: limiting `ignore` to sub-default quantization would leave unverifiable cheap endpoints unblocked, and inverse-square default routing favors exactly those); `quantizations` lists quantizations >= default. `require_parameters`/`zdr`/`data_collection` are left to the caller (plugin contract — see §7).

### 4.8 Per-task cost (G8/D5)

- Source: `GET https://openrouter.ai/api/v1/datasets/session-cost` — requires an OpenRouter API key (401 otherwise). Query params: `app_slug` (publish uses a fixed published harness, e.g. `hermes-agent` — builder: pick the harness with the widest model coverage at implementation time and pin it in a constant), `model` (exact permaslug), `turn_range`, `limit`, `offset`.
- Cells: `{app_name, app_slug, median_session_cost_usd, model_permaslug, turn_range}`; meta: `{as_of, version, window_days, window_end_date}`.
- Catalog block `session_cost` carries the median value(s), `as_of`, window metadata, harness slug, and the attribution string "Data © OpenRouter, CC BY 4.0".
- **Excluded** from the weighted score. Published as a separate site column/section.
- Keyless (no `OPENROUTER_API_KEY`) → block omitted entirely; site renders an explicit "not available" state (AA precedent), never a zero.
- **Turn-range selection (designer call, 2026-10-19):** the catalog block stores the medians for **all four** turn ranges plus meta; the site column displays the pinned `10-49-turns` median (constant `SESSION_COST_DISPLAY_TURN_RANGE`) as representative of harness/plugin multi-turn workloads.
- **Failure with key present (designer call, 2026-10-19):** CI always sets `OPENROUTER_API_KEY` (already used by `generate_highlights.py`), so the keyless branch is unreachable in CI. With a key set and the fetch failing after retries (§4.9): graceful degradation — the block is omitted and the site renders the explicit "not available this refresh" state. No publish gate: session cost is a secondary, noisy third-party dimension (D5 rationale); failing the whole publish over it would be disproportionate. The state is loud, never silent, never zero.

### 4.9 Fetch strategy & cache

- One `endpoints` request per candidate model per refresh; session-cost one request per publish (fetch all needed models via pagination or per-model query — builder decides based on measured payload; cache regardless).
- Cache integration: reuse the existing CLI 6h TTL cache pattern; publish primes caches before building.
- **Fetch concurrency (designer call, 2026-10-19):** endpoint fetches for all candidates run concurrently through a bounded worker pool (e.g. `ThreadPoolExecutor`, 8 workers), cache-first. A two-phase "coarse rank then fetch top-K" path was considered and rejected: a top-K cut on catalog price is unsound precisely because catalog prices are the misleading quantities this design corrects.
- **Retries (designer call, 2026-10-19):** endpoint and session-cost fetches retry 3 times with backoff before being treated as failed.
- **Fail policy (hybrid):** endpoint fetch failure → that model cannot be ranked this refresh → dropped with reason `endpoint_fetch_failed` (fail-closed for ranking). **Stale-annotation referent (designer call, 2026-10-19, second builder check):** a dropped model's `filtered` entry additionally carries the model's **last cached** endpoint annotations (from the expired cache, if any) under an `annotations` key marked `"stale": true`, for display in the filtered section only. Ranking on stale prices remains forbidden (the fail-closed settlement stands); if no cache exists the entry carries the drop reason alone.
- **Publish drop-rate gate (designer call, 2026-10-19, scope widened per second builder check):** if more than **5%** of ZDR candidates are dropped in a publish run for **any endpoint-derived reason** — `endpoint_fetch_failed` **or** `no_baseline_eligible_endpoint` — the publish **fails** (constant `MAX_ENDPOINT_DROP_RATE = 0.05`) — fail-loud, consistent with the repo's existing publish gates (`check_aa_fallback`, strict schema gates). A transient 5xx must never silently shrink the public ranking; neither may a first-party detection miss (e.g. a closed lab whose provider names don't match §4.2 yet) or an API shape regression that nulls `quantization` fleet-wide.

## 5. What's Left to the Builder

- Exact constant values: precision-ladder dict, `FIRST_PARTY_HOSTS`, harness slug for session-cost.
- Payload-shape parsing details from a live endpoints response during implementation (tag vs slug splitting for `order`/`ignore` lists).
- CLI column layout for new annotation columns/notes; exact `--json` key names (must stay additive).
- Site section placement for per-endpoint table, quantization badges, per-task cost column.
- Whether session-cost fetch is per-model or paginated-bulk (decide on payload size).
- Test strategy mirroring existing suites (pytest for CLI/catalog, site JS harness parity tests for new columns).

## 6. Residual Mismatch (documented, not fixed)

Quality (AA index) is measured at the model maker's release precision. Endpoint prices are now filtered to that same precision band (>= default), so the mismatch is *narrower* than before, but a per-endpoint quality signal still does not exist. The design accepts this and labels the ranking "price at default precision, quality at release precision".

## 7. Plugin Contract (vendor + steering)

`model-compare-opencode-plugin` vendors `model_compare.py`. On selection it reads the `steering` block and applies the OpenRouter `provider` object to its own request routing. The plugin decides fallback policy (`allow_fallbacks`) and whether to also pass `require_parameters`/`zdr`. model-compare only guarantees that `order`/`ignore`/`quantizations` are present and correct per §4.7.

## 8. Schema Version & Artifacts

- Catalog `schema_version` → **3**. The publish gate (shipped in 0.3.3 for v1) must be extended to accept exactly `3` and reject anything else (fail-closed identity check, same pattern as the existing strict gate).
- **Site `data.json` bumps to schema_version 2** (designer call, 2026-10-19): the new site columns/sections change the site artifact's shape, so `publish.py`'s strict `data.json` version gate is extended the same way — accept exactly `2`, reject anything else. Both gates are strict identity checks.
- **Per-endpoint block lives in catalog.json v3** (per brainstorm D1 — "additive per-model provider/endpoint block"); the site reads it from there. Catalog churn from endpoint prices is acceptable: the catalog already churns per publish with prices.
- **`tokens.canonical.com` coordination (settled with the human partner, 2026-10-19): investigate first, then decide.** Before the first v3 publish, inspect what tokens.canonical.com consumes and whether it validates `schema_version` strictly; block the first v3 publish only if it would break. If downstream turns out strict, coordinate the downstream update and land it before shipping; if tolerant, ship and note the dependency. This investigation is a plan task, not a docs-only note.
- **history.json at the v3 cutover (designer call, 2026-10-19, second builder check):** `HISTORY_SCHEMA_VERSION` bumps 2 → 3. `merge_history` already starts fresh on unknown versions, so the price history effectively **resets at the cutover** — the honest behavior: old snapshots describe old price semantics (catalog top-level prices), and mixing them with baseline-endpoint prices would fabricate a fake ~8x "price increase" trend (e.g. glm-5.3-flash: $0.0184 → $0.15 per 1M input). No continuity bridge is built; the design explicitly accepts the history reset.

## 9. Docs Lag (opportunistic, unrelated)

`DESIGN.md` still documents a 6h site cadence; actual is 3h (e1e79b5). Fix opportunistically in the same PR.
