# Provider-Aware Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank models at default-precision (baseline) OpenRouter endpoint prices, annotate quantization risk, emit steering data, and publish per-task cost as a separate dimension (catalog v3 / data.json v2 / history v3).

**Architecture:** A new endpoint stage sits between `build_candidates` and `compute_scores` in `model_compare.py`: for each surviving candidate it fetches the model's OpenRouter endpoints (cached, concurrent, retried), derives the default precision from the maker's endpoint, picks the cheapest eligible baseline endpoint, overwrites the candidate's pricing fields with that endpoint's effective pricing, and attaches annotations + steering. Failures drop the model with new machine-readable reasons that join `CATALOG_DROP_REASONS`. The site gains a catalog.json fetch and per-model endpoint disclosure; publish gains a drop-rate gate.

**Tech Stack:** Python 3 stdlib (urllib, ThreadPoolExecutor, argparse, json), pytest, node-backed JS harness for site helpers.

**Spec:** `docs/superpowers/specs/2026-10-19-provider-aware-eval-design.md` (read together with this plan; the spec's §4 rulings and §5 builder items are normative here).

**Out of scope:** the `model-compare-opencode-plugin` repo (separate repository; it consumes the `steering` block this plan emits).

## Global Constraints

- Byte-identical determinism: outputs must be reproducible; every sort/tie-break is pinned; concurrency must not affect output (results applied per candidate id, never in completion order).
- Strict identity version gates: catalog `schema_version` 3 (both `model_compare.py` and `web/build_site_data.py` constants), site `data.json` 2 (`build_site_data.py` + `publish.py` pin), `history.json` 3.
- Fail-closed ranking: endpoint fetch failure or no eligible endpoint ⇒ drop with reason; never rank on stale prices.
- Session cost never enters the weighted score; keyless or failed fetch ⇒ block omitted, never zero.
- CLI changes are additive only; existing flags, `--best`/`--json`/`--catalog` semantics preserved.
- New drop reasons MUST be added to `CATALOG_DROP_REASONS` (pool-accounting invariant `sum(dropped) + candidates == listed` is load-bearing; enforced by `web/build_site_data.py` `validate_catalog`).
- Tests run as in CI: `python3 -m pytest -q` from repo root (CI: Python 3.12, `pip install pytest`). Local sandbox may lack pip/pytest — use `uv run --with pytest pytest -q` as the local fallback (verified working during review).

## Review Focus

1. **All-unknown closed models** (e.g. `anthropic/claude-haiku-5.5`, 10/10 endpoints `unknown`): must NOT vanish — first-party carve-out keeps them ranked; `default_precision` is null; steering omits `quantizations`. → Task 4 (`test_all_unknown_model_keeps_first_party_ranked`), Task 5 (`test_steering_omits_quantizations_when_default_null`).
2. **Zero-priced / `:free` / `:batch` variants**: must never set the ranked price nor pin the default. → Task 3 (`test_variant_detection_*`), Task 4 (`test_variant_never_pins_default`).
3. **Schedule-broken endpoints** (`schedule_error` in full): must not pin the default nor be ranked. → Task 4 (`test_schedule_broken_never_pins_default`).
4. **Transient endpoint-fetch failure**: model drops with `endpoint_fetch_failed`, and >5% of such drops fails the publish. → Task 6 (`test_fetch_failure_drops_model`), Task 10 (`test_endpoint_drop_rate_gate`).
5. **Dated permaslugs** (`model_permaslug` with `-YYYYMMDD`): must map to the bare catalog id, latest date wins. → Task 7 (`test_session_cost_merges_dated_permaslugs`).

---

### Task 1: Downstream investigations (tokens.canonical.com + advertised-price semantics)

**Files:**
- Create: `docs/superpowers/notes/2026-10-20-downstream-investigations.md`

**Interfaces:**
- Produces: recorded answers used by Task 12 (DESIGN.md contract rewrite) and the go/no-go note for the first v3 publish (design §8).

- [ ] **Step 1: Investigate tokens.canonical.com.** Fetch the site, identify which artifact it consumes (`catalog.json` and/or `data.json`) and whether it validates `schema_version` strictly. Record the URL(s) inspected and the evidence.
- [ ] **Step 2: Investigate advertised-price semantics.** Using the OpenRouter docs and a live `/api/v1/models` response for `z-ai/glm-5.3-flash`, determine why top-level pricing ($0.0184/M input) sits below every routable endpoint ($0.0258 cheapest). Record the finding or mark it "unresolved — documentation-only annotation".
- [ ] **Step 3: Write the notes file** with both outcomes and the publish go/no-go recommendation.
- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/notes/2026-10-20-downstream-investigations.md
git commit -m "docs: record tokens.canonical.com + advertised-price investigations"
```

### Task 2: Precision ladder + provider normalization

**Files:**
- Modify: `model_compare.py` (constants near `CATALOG_SCHEMA_VERSION`, ~line 89; new functions near `norm_key`, ~line 117)
- Test: `test_model_compare.py` (new test class near the existing catalog tests)

**Interfaces:**
- Produces: `PRECISION_LADDER: dict[str, int]` (fp32:4, bf16:3, fp16:3, fp8:2, mxfp8:2, fp6:1, int8:0, fp4:-1, mxfp4:-1, nvfp4:-1, int4:-2); `QUANTIZATION_ENUM: frozenset` (the 10 known values + "unknown"); `FIRST_PARTY_HOSTS: frozenset` ("bedrock", "azure", "vertex", "azure-ai", "google-vertex", "aws-bedrock"); `FIRST_PARTY_ALIASES: dict[str, str]` (empty initially); `normalize_provider(name: str) -> str` (casefold + strip all non-alphanumeric); `is_first_party(provider_name: str, author: str) -> bool` (normalized author equality OR normalized name in normalized FIRST_PARTY_HOSTS/ALIASES).
- Consumed by: Tasks 3, 4, 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_precision_ladder_fp6_outranks_int8():
    assert PRECISION_LADDER["fp6"] > PRECISION_LADDER["int8"]
    assert PRECISION_LADDER["fp32"] == max(PRECISION_LADDER.values())
    assert PRECISION_LADDER["int4"] == min(PRECISION_LADDER.values())
    assert "unknown" not in PRECISION_LADDER

def test_normalize_provider_strips_punctuation():
    assert normalize_provider("Z.AI") == normalize_provider("z-ai") == "zai"
    assert normalize_provider("AWS Bedrock") == normalize_provider("aws-bedrock")

def test_first_party_author_and_partner():
    assert is_first_party("Z.AI", "z-ai")
    assert is_first_party("AWS Bedrock", "anthropic")
    assert not is_first_party("OpenInference", "z-ai")
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest test_model_compare.py -q -k "precision_ladder or normalize_provider or first_party"` → FAIL (NameError).
- [ ] **Step 3: Implement** the constants and the two functions per the signatures above.
- [ ] **Step 4: Run to verify they pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add model_compare.py test_model_compare.py && git commit -m "feat: precision ladder + first-party provider normalization"`

### Task 3: Endpoint fetch, parsing, variant + schedule classification

**Files:**
- Modify: `model_compare.py` (new "Endpoints" section after the cache helpers, ~line 226)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: `OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{id}/endpoints"`; `fetch_model_endpoints(model_id: str, args) -> list[dict]` (cache name `endpoints:<id with "/"→"_">`, TTL `args.cache_ttl`, `fetch_json(..., retries=3)`; non-200/4xx ⇒ raise); `parse_endpoints(payload) -> list[dict]` keeping `provider_name`, `tag`, `quantization`, `pricing` (prompt/completion/`overrides`), `context_length`, `status`, `service_tier` with prices through `parse_price`; `is_variant_endpoint(endpoint) -> bool` (slug/tag carries `:free`/`:batch` marker OR input and output prices both zero); `endpoint_schedule_ok(endpoint, args) -> bool` (`effective_pricing(endpoint pricing, args.min_context, args.input_share)`, `schedule_error` ⇒ False); `load_cache_expired(name: str) -> dict | None` (returns payload even past TTL, None when absent/corrupt).
- Consumes: Task 2; existing `fetch_json`, `parse_price`, `effective_pricing`, `load_cache`, `save_cache`.
- Consumed by: Tasks 4, 5, 6.

- [ ] **Step 1: Write the failing tests** (fixture mirrors the live shape recorded in brainstorm §2.1/§2.4 — same style as `zdr_endpoint()` at test_model_compare.py:1303)

```python
def _endpoint(**over):
    base = {"provider_name": "Z.AI", "tag": "z-ai/fp8", "quantization": "fp8",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000005"},
            "context_length": 262144, "status": 0, "service_tier": None}
    base.update(over); return base

def test_parse_endpoints_keeps_verified_fields():
    eps = parse_endpoints({"data": [_endpoint()]})
    assert eps[0]["provider_name"] == "Z.AI" and eps[0]["status"] == 0
    assert eps[0]["pricing"]["prompt"] == 0.00000015

def test_variant_detection_marker_and_zero_price():
    assert is_variant_endpoint(_endpoint(tag="z-ai:free"))
    assert is_variant_endpoint(_endpoint(tag="z-ai:batch"))          # marker, not price
    assert is_variant_endpoint(_endpoint(pricing={"prompt": "0", "completion": "0"}))
    assert not is_variant_endpoint(_endpoint())

def test_endpoint_schedule_gate_uses_full_schedule_error():
    assert endpoint_schedule_ok(_endpoint(), _args())
    broken = _endpoint(pricing={"prompt": "0.1", "completion": "0.1",
                                "overrides": {"schedule": [{"utc_days": ["monday"], "utc_start": "25:99", "utc_end": "01:00"}]}})
    assert not endpoint_schedule_ok(broken, _args())

def test_load_cache_expired_returns_stale_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    save_cache("k", {"v": 1})
    assert load_cache_expired("k") == {"v": 1}      # even though fresh-load would also pass, expire it:
```

(exact expiry simulation: write a blob with `fetched_at` in the past, then assert `load_cache` returns None while `load_cache_expired` returns the payload)

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest test_model_compare.py -q -k "parse_endpoints or variant_detection or endpoint_schedule or load_cache_expired"` → FAIL.
- [ ] **Step 3: Implement** the functions and constant per the signatures.
- [ ] **Step 4: Run to verify they pass** — same command → PASS; then full `python3 -m pytest -q` → 580+ pass (no regressions).
- [ ] **Step 5: Commit** — `git commit -m "feat: OpenRouter endpoints fetch, parsing, variant + schedule classification"`

### Task 4: Default precision, eligibility, baseline selection

**Files:**
- Modify: `model_compare.py` (Endpoints section)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: `derive_default_precision(endpoints: list[dict], author: str) -> str | None` (universe = live status-0, non-variant, schedule-passing; maker = normalized `provider_name == author` ONLY — partners never pin; maker-best, else best-available; None when no known quantization); `baseline_eligible(endpoint, default_precision, author, args) -> bool` (status 0, not variant, schedule passes, then: known quant ≥ rank(default) OR unknown/null AND first-party); `pick_baseline(endpoints, eligible_flags, args) -> dict` (cheapest blended from `effective_pricing`, ties by `(normalize_provider(provider_name), tag)`).
- Consumes: Tasks 2, 3. Consumed by: Tasks 5, 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_maker_quantization_pins_default():
    eps = [_endpoint(provider_name="Z.AI", quantization="fp8"),
           _endpoint(provider_name="DeepInfra", quantization="fp4")]
    assert derive_default_precision(eps, "z-ai") == "fp8"

def test_variant_and_schedule_broken_never_pin_default():
    eps = [_endpoint(provider_name="Z.AI", quantization="fp8", tag="z-ai:free"),
           _endpoint(provider_name="Z.AI", quantization="fp8", schedule_broken=True),
           _endpoint(provider_name="DeepInfra", quantization="fp4")]
    assert derive_default_precision(eps, "z-ai") == "fp4"   # best-available fallback

def test_all_unknown_model_default_is_null():
    eps = [_endpoint(provider_name="Anthropic", quantization="unknown")]
    assert derive_default_precision(eps, "anthropic") is None

def test_partner_never_pins_maker_default():
    eps = [_endpoint(provider_name="AWS Bedrock", quantization="fp8")]
    assert derive_default_precision(eps, "anthropic") is None   # partner-only, unknown-free but not maker

def test_first_party_unknown_eligible_third_party_not():
    args = _args()
    d = None
    assert baseline_eligible(_endpoint(provider_name="Anthropic", quantization="unknown"), d, "anthropic", args)
    assert not baseline_eligible(_endpoint(provider_name="Relace", quantization="unknown"), d, "anthropic", args)

def test_baseline_cheapest_with_pinned_tiebreak():
    # two eligible fp8 endpoints, same blended price -> normalized provider name, then tag
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest test_model_compare.py -q -k "derive_default or baseline_eligible or baseline_cheapest"` → FAIL.
- [ ] **Step 3: Implement** the three functions.
- [ ] **Step 4: Run to verify they pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: default-precision derivation, baseline eligibility + selection"`

### Task 5: Annotations + steering builder

**Files:**
- Modify: `model_compare.py` (Endpoints section)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: `NON_ELIGIBLE_UNIVERSE(endpoints, eligible_flags) -> list` (live, non-variant, not eligible: sub-default quantization, third-party unknown/null, schedule-gate-failed); `build_annotations(endpoints, eligible_flags, baseline, catalog_context, args) -> dict` with keys `price_spread` (cheapest non-eligible blended ÷ cheapest eligible blended; None when the universe is empty), `cheapest_quantized` (quantization-caused subset only: sub-default known quantization or third-party unknown/null — `{provider, quantization, price}` or None), `risk_flag` (True iff Σ 1/blended² over the non-eligible universe > Σ over eligible — strict, exact tie False), `baseline_context_below_catalog` (bool); `build_steering(model_id, endpoints, eligible_flags, default_precision) -> dict` = `{"model": model_id, "provider": {"order": [eligible tags cheapest-first], "ignore": [tags of every live non-eligible endpoint, variants and schedule-gated included], "quantizations": [known quantizations ≥ default among live non-variant, sorted] (OMITTED entirely when default_precision is None), "allow_fallbacks": True}}`.
- Consumes: Tasks 2–4. Consumed by: Task 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_price_spread_direction_and_null():
    # eligible fp8 at 0.15, non-eligible fp4 at 0.075 -> spread = 0.075/0.15 = 0.5
def test_price_spread_null_when_no_non_eligible(): ...
def test_cheapest_quantized_excludes_schedule_gated(): ...
def test_risk_flag_weight_formula():
    # eligible fp8 @0.15 vs non-eligible fp4 @0.0258 -> 1/0.0258² >> 1/0.15² -> True
    # eligible-only model -> False; exact-tie construction -> False
def test_baseline_context_below_catalog_flag(): ...
def test_steering_order_ignores_and_quantizations(): ...
def test_steering_omits_quantizations_when_default_null(): ...
```

- [ ] **Step 2: Run to verify they fail** — `-k "price_spread or cheapest_quantized or risk_flag or steering or context_below"`.
- [ ] **Step 3: Implement** the two builders.
- [ ] **Step 4: Run to verify they pass**.
- [ ] **Step 5: Commit** — `git commit -m "feat: endpoint annotations + steering builder"`

### Task 6: CLI wiring — endpoint stage, drops, `--json` extension

**Files:**
- Modify: `model_compare.py` (`CATALOG_DROP_REASONS` :94; new `apply_endpoint_pricing` in the Endpoints section; `run()` :1852 wiring; `print_json` :1487; `parse_args` :1721)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: `CATALOG_DROP_REASONS` extended with `("no_baseline_eligible_endpoint", "endpoint_fetch_failed")` (snake_case — deliberate mixed convention, documented); `apply_endpoint_pricing(candidates: list[dict], args, filtered: list) -> tuple[list[dict], dict[str, int]]` — concurrent (`ThreadPoolExecutor(max_workers=8)`, results applied per candidate id), per-model cache; on fetch failure after retries: drop `endpoint_fetch_failed`, filtered entry gains `annotations` from `load_cache_expired` marked `stale: true` (absent cache ⇒ reasons only); on no eligible endpoint: drop `no_baseline_eligible_endpoint` (same stale-annotations rule); otherwise mutates the candidate in place: `price_in/price_out/blended/base_price_in/base_price_out/tier_prompt_tokens/tiers/schedule/max_discount/tier_note/sched_note/sched_detail` ← baseline endpoint's `effective_pricing` outputs (per-1M scaling as in `build_candidates` :1253-1274), plus new keys `endpoints`, `default_precision`, `annotations`, `advertised_pricing` (models-API scalars + `"source": "models-api"`), `steering`. `parse_args` gains `--openrouter-api-key` (env `OPENROUTER_API_KEY` fallback). `print_json` rows gain additive keys exactly matching Task 9's ROW_KEYS extension: `default_precision`, `risk_flag`, `price_spread`, `session_cost_usd` (None until Task 7).
- Consumes: Tasks 2–5. Consumed by: Tasks 7–9.

- [ ] **Step 1: Write the failing tests**

```python
def test_pool_accounting_balances_with_endpoint_drops(...):
    # monkeypatch fetch_model_endpoints; one model with no eligible endpoint
    # assert len(models) == len(candidates) + sum(dropped.values())
    # assert dropped["no_baseline_eligible_endpoint"] == 1
    # assert filtered entry == {"id": ..., "name": ..., "reasons": ["no_baseline_eligible_endpoint"], ...}

def test_candidate_pricing_replaced_by_baseline(...):
    # catalog pricing block == baseline endpoint prices, not models-API prices
    # advertised_pricing carries the models-API scalars

def test_fetch_failure_drops_model_with_stale_annotations(...):
    # fetch raises; expired cache exists -> filtered entry annotations.stale is True
    # assert dropped["endpoint_fetch_failed"] == 1

def test_json_rows_carry_new_additive_keys(): ...
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the wiring (drop-reason constants, `apply_endpoint_pricing`, `run()` insertion point: after `build_candidates(...)` line 1864, before `resolve_quality` line 1872 — merge `extra_dropped` into `dropped`, reassign `candidates`; print_json keys; CLI flag).
- [ ] **Step 4: Run to verify they pass** + full suite `python3 -m pytest -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat: provider-aware endpoint stage with fail-closed drops + steering in --json"`

### Task 6b: CLI/site surface — PREC column, risk marker, quant_note

**Files:**
- Modify: `model_compare.py` (`print_table` ~:2160, `print_json` ~:2300, new `fmt_quant_note` near the other `fmt_*` helpers :2035)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: `fmt_quant_note(annotations: dict) -> str | None` — None when no non-eligible live endpoint exists; else a terse house-style string like `cheapest non-baseline: fp4 @ open-inference, $0.026/M in (0.17x baseline)` built from `cheapest_quantized` + `price_spread` (the exact format is pinned by the test). `print_table`: new `PREC` header after `DISC`; cell = `default_precision` or `n/a` when None, suffixed `!` when `annotations.risk_flag`; footer legend gains the PREC explanation line (pinned in the test). `print_json`: additive row key `quant_note` = `fmt_quant_note(annotations)` (build_site_data's ROW_KEYS does not require it until Task 9 — emit-first ordering preserves the missing-key check invariant).
- Consumes: Task 6's candidate keys (`default_precision`, `annotations`). Consumed by: Task 9 (ROW_KEYS gains `quant_note`), Task 11 (site tooltip shows the string verbatim).

- [ ] **Step 1: Write the failing tests** — `test_fmt_quant_note_formats_cheapest_non_baseline` (asserts the exact string from cheapest_quantized {provider, quantization, price} + spread), `test_fmt_quant_note_none_without_non_eligible`, `test_cli_table_shows_prec_column_and_risk_marker` (capsys on a print_table run: PREC column present, `fp8!` on a risk row, `n/a` for null default, footer legend line present), `test_json_rows_carry_quant_note`.
- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest test_model_compare.py -q -k "quant_note or prec_column"`.
- [ ] **Step 3: Implement** the formatter, the column, the footer line, the JSON key.
- [ ] **Step 4: Run to verify they pass** + full suite (`uv run --with pytest pytest -q`).
- [ ] **Step 5: Commit** — `git commit -m "feat: PREC column + risk marker + quant_note in CLI/JSON output"`

### Task 7: Session-cost dataset fetch

**Files:**
- Modify: `model_compare.py` (new "Session cost" section)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: `SESSION_COST_URL = "https://openrouter.ai/api/v1/datasets/session-cost"`; `SESSION_COST_HARNESS_SLUG = "hermes-agent"`; `SESSION_COST_DISPLAY_TURN_RANGE = "10-49-turns"`; `SESSION_COST_ATTRIBUTION = "Data © OpenRouter, CC BY 4.0"`; `fetch_session_cost(args) -> dict | None` — None when no key (`args.openrouter_api_key` / `OPENROUTER_API_KEY`) or after retries fail; bulk query `?app_slug=<slug>&limit=500&offset=k` paginated until a short page or offset > 5000; `strip_permaslug_date(permaslug) -> str` (trailing `-YYYYMMDD` removed; collision ⇒ latest date wins — same rule as `build_aa_benchmarks` :977); returns `{"attribution", "app_slug", "as_of", "window_days", "window_end_date", "medians": {catalog_id: {turn_range: usd}}}`.
- Consumes: Task 6 (key handling). Consumed by: Task 8.

- [ ] **Step 1: Write the failing tests** (fixture rows with dated permaslugs; assert merge, keyless None, failure None, pagination stop on short page).
- [ ] **Step 2: Run to verify they fail** — `-k "session_cost or permaslug"`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: OpenRouter session-cost dataset (per-task cost, outside the score)"`

### Task 8: Catalog v3 — schema bump + additive blocks

**Files:**
- Modify: `model_compare.py` (`CATALOG_SCHEMA_VERSION` :89 → 3; `build_catalog` :1527)
- Test: `test_model_compare.py`

**Interfaces:**
- Produces: catalog entries gain `endpoints`, `default_precision`, `advertised_pricing`, `annotations` (from candidate keys — always present, possibly null); top-level `session_cost` block (from Task 7, omitted when unavailable); `sources` gains `"endpoints"` (`"ok"`/`"unavailable"`) and `"session_cost"` (`"ok"`/`"unavailable"`/`"skipped"`). The `pricing` block is already baseline-endpoint-shaped via Task 6's mutation — assert it.
- Consumes: Tasks 6, 7. Consumed by: Task 9's validator.

- [ ] **Step 1: Write the failing tests** — `test_catalog_schema_version_3`, `test_catalog_entry_carries_endpoint_blocks`, `test_catalog_session_cost_block_omitted_keyless`, `test_catalog_pricing_block_is_baseline_endpoint` (run `--catalog` against monkeypatched fetches).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the schema bump and additive blocks.
- [ ] **Step 4: Run to verify they pass** + full suite.
- [ ] **Step 5: Commit** — `git commit -m "feat: catalog schema v3 — provider/endpoint blocks + session cost"`

### Task 9: build_site_data.py — validate v3, data.json v2, history v3

**Files:**
- Modify: `web/build_site_data.py` (`CATALOG_SCHEMA_VERSION` :42 → 3; `validate_catalog` :206; `DATA_SCHEMA_VERSION` :717 → 2; `ROW_KEYS` :21; `HISTORY_SCHEMA_VERSION` :462 → 3)
- Test: `web/test_build_site_data.py`

**Interfaces:**
- Produces: `validate_catalog` additionally validates (additive-tolerant style, fail-closed on shape); `ROW_KEYS` += `("default_precision", "risk_flag", "price_spread", "session_cost_usd", "quant_note")` — five keys, matching Tasks 6/6b's `print_json` exactly: `endpoints` list entries carry string `provider_name`/`tag`, `quantization` in `QUANTIZATION_ENUM ∪ {None}` (import or duplicate the enum — publish is subprocess-isolated but build_site_data already duplicates catalog constants; duplicate `PRECISION_LADDER`/`QUANTIZATION_ENUM` here), numeric-or-null prices, int `status`; `default_precision` is None or a ladder key; `annotations` keys numeric-or-null / bool; `advertised_pricing` numeric-or-null; top-level `session_cost` (when present) carries string `attribution`/`as_of`/`app_slug`, int `window_days`, `window_end_date`, `medians` mapping ids to 4-turn-range numeric dicts.
- Consumes: Tasks 6–8 outputs.

- [ ] **Step 1: Write the failing tests** — `test_validate_catalog_accepts_v3_blocks`, `test_validate_catalog_rejects_v2_document`, `test_validate_catalog_rejects_bad_endpoint_entry`, `test_rows_carry_new_keys_ordered_by_ranking`, `test_merge_history_resets_on_v2_history` (prev `schema_version: 2` ⇒ fresh).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the three version bumps + validators + ROW_KEYS.
- [ ] **Step 4: Run to verify they pass** — `python3 -m pytest web/ -q` then full suite.
- [ ] **Step 5: Commit** — `git commit -m "feat: site validators for catalog v3, data.json v2, history v3"`

### Task 10: publish.py — data.json pin 2 + endpoint drop-rate gate

**Files:**
- Modify: `web/publish.py` (`check_stamps` :97-104 version pin 1 → 2; new `check_endpoint_drops` near `check_aa_fallback` :150; wire into `build_site` after `check_aa_fallback` :193)
- Test: `web/test_publish.py`

**Interfaces:**
- Produces: `MAX_ENDPOINT_DROP_RATE = 0.05`; `check_endpoint_drops(catalog_path: Path) -> None` — reads `pool.dropped["no_baseline_eligible_endpoint"] + ["endpoint_fetch_failed"]`; denominator = `pool["candidates"] + those two counts` (the models that entered the endpoint stage); raise `RuntimeError` when rate > 0.05; zero denominator cannot occur (check_best fails on empty rankings first) but the check must not divide by zero if reached.
- Consumes: Task 8's catalog. Consumed by: publish pipeline.

- [ ] **Step 1: Write the failing tests** — `test_data_json_version_pin_is_2` (update existing `test_build_site_requires_data_json_schema_version`), `test_endpoint_drop_rate_gate_passes_under_threshold`, `test_endpoint_drop_rate_gate_fails_over_threshold`, `test_endpoint_drop_rate_gate_zero_denominator_safe`.
- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest web/test_publish.py -q`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: publish gates — data.json v2 pin + 5% endpoint drop-rate gate"`

### Task 11: Site UI — catalog fetch, endpoint disclosure, badges, session cost

**Files:**
- Modify: `web/site/index.html` (fetch block ~:892; render section; new per-model disclosure UI)
- Test: `web/test_site_js.py` (+ `web/js_stubs.js` if needed)

**Interfaces:**
- Consumes: data.json v2 rows (`default_precision`, `risk_flag`, `price_spread`, `session_cost_usd`) and a new `catalog.json` fetch for per-endpoint tables (catalog.json is already a deployed artifact).
- Produces: quantization badge column (PREC: `default_precision` or `n/a`, `!` suffix when `risk_flag`; header tooltip defines both) whose cell tooltip shows `quant_note` verbatim (the CLI-printed string, Task 6b); risk-flag marker is the `!`; session-cost column TASK$ (`session_cost_usd` formatted `$X.XX`, explicit `—` "not available this refresh" state — never zero, never blank-silent; header tooltip cites the session-cost dataset + CC BY 4.0); per-model expandable endpoint table (NEW UI pattern — the site has only table-level "Show more" today) rendering provider, quantization badge, in/out prices, status, context, service tier, ✓ on the baseline row, from a new catalog.json fetch; CTX-cell caveat tooltip when `baseline_context_below_catalog` ("ranked price from a <N>-token endpoint; model supports <M>").

- [ ] **Step 1: Write failing JS-harness tests** for the new pure helpers (badge text from `default_precision`, session-cost formatting + not-available state, spread formatting).
- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest web/test_site_js.py -q` (skipped cleanly when node absent).
- [ ] **Step 3: Implement** the fetch + render changes in index.html.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(site): endpoint disclosure, quantization badges, per-task cost column"`

### Task 12: Docs — DESIGN.md contract rewrite + cadence fixes

**Files:**
- Modify: `DESIGN.md` ("Catalog output contract" section ~:151-237; cadence line ~:145), `README.md` (:56-57), `web/README.md` (6h claim)

**Interfaces:**
- Consumes: Task 1's investigation notes; the final implemented shapes from Tasks 6–10.

- [ ] **Step 1: Rewrite DESIGN.md's Catalog output contract for v3**: baseline-endpoint pricing semantics (scalars + base/tiers/schedule describe the baseline endpoint), `advertised_pricing` (documentation-only; semantics per Task 1), annotations + their shared universe, `endpoints` block, `session_cost` block + attribution, the two new drop reasons (mixed naming convention documented), schema_version 3 / data.json 2 / history 3, the drop-rate gate.
- [ ] **Step 2: Fix cadence claims** 6h → 3h in DESIGN.md, README.md, web/README.md.
- [ ] **Step 3: Verify** — `grep -rn "6 hours\|6h\|starts at 2" README.md web/README.md DESIGN.md` returns no stale contract claims (cadence) and the contract section describes v3.
- [ ] **Step 4: Commit** — `git commit -m "docs: DESIGN.md catalog contract v3 + cadence corrections"`

### Task 13: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite** — `python3 -m pytest -q` → all pass (baseline was 580).
- [ ] **Step 2: Live smoke test** — `python3 model_compare.py --catalog > /tmp/catalog.json && python3 -c "import json; d=json.load(open('/tmp/catalog.json')); assert d['schema_version']==3; print(len(d['models']), 'models')"`; then `python3 web/build_site_data.py --help` sanity.
- [ ] **Step 3: Publish dry run** — `python3 web/publish.py --help` and (if network allows) a full `python3 web/publish.py` into a temp output dir; confirm gates pass.
- [ ] **Step 4: Report** the counts and any live-data anomalies in the session summary (no commit).
