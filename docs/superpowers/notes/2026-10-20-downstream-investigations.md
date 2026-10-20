# Downstream investigations — tokens.canonical.com + advertised-price semantics

*Task 1 of the provider-aware-eval plan (2026-10-20). Investigated live from the
sandbox; every claim below carries the URL or quote it rests on. Investigated by
a plan subagent; reviewed artifacts saved under `/tmp/opencode/` during the
session (raw HTML/JSON/api clones), not in the repo.*

---

## Question 1 — What does tokens.canonical.com consume, and does it validate `schema_version`?

### Evidence

**E1.1 — tokens.canonical.com is a private GitHub Pages site; the live JS cannot
be fetched anonymously.**
`curl -sIL https://tokens.canonical.com` (2026-10-20 18:36 UTC) returned:

> ```
> HTTP/2 302
> server: GitHub.com
> x-pages-private: 1
> location: https://github.com/pages/auth?nonce=79898eb2-...&page_id=80090374&path=Lw
> ```

followed by a second 302 to `https://github.com/login?return_to=...`; the
anonymous body is GitHub's "Sign in to GitHub · GitHub" page. DNS confirms the
host is a Pages custom domain: `tokens.canonical.com` → CNAME
`canonical.github.io` (IPv4 185.199.108–111.153, AAAA `2606:50c0:8000–8003::153`,
reverse-resolving to `canonical.github.io`). Consequence: the live HTML/JS could
not be inspected from the sandbox; the ground truth inspected instead is the
**source repository** of the app behind the site (E1.3), which is better
evidence than minified JS. *Uncertain marker:* which commit the deployed
instance runs cannot be verified from the sandbox; the inspected repo HEAD is
`6eea511` (2026-10-19 17:17 UTC, Merge PR #605).

**E1.2 — The consumer is `canonical/agent-provisioner` (private repo).**
GitHub code search, authenticated as `rkratky` (PAT with `repo` scope):
- `q="tokens.canonical.com" org:canonical` → 63 results in exactly three repos:
  `canonical/agent-provisioner` (60 files), `canonical/model-compare` (1 file:
  DESIGN.md), `canonical/ai-design-ops` (2 files: end-user docs).
- `q="canonical.github.io/model-compare"` (global) → 16 results in exactly three
  repos: `canonical/agent-provisioner`, `canonical/model-compare` (own docs),
  `canonical/model-compare-opencode-plugin` (see E1.6).

`canonical/agent-provisioner` self-identifies as the site's app:
- `src/web/settings.py:60`: `frontend_origins: str = "https://tokens.canonical.com,http://localhost:5173"`
- `README.md:196`: "`docs/` is published at <https://tokens.canonical.com/docs/>"
- `charm/charmcraft.yaml`: `default: "https://tokens.canonical.com"` for
  `post-login-redirect`; `charm/README.md:110`:
  `juju config tokens-backend post-login-redirect=https://tokens.canonical.com`
  (the deployed charm is named **tokens-backend**).
- `.github/workflows/deploy-dashboard.yml:3`: "Publish tokens.canonical.com to
  GitHub Pages: the spend-dashboard frontend" — the Pages site serves the
  frontend; the backend (the Python web app with the ingests below) is the
  juju-deployed charm.

**E1.3 — What it fetches: model-compare's `data.json`, NOT catalog.json, NOT
history.json.**
`src/web/settings.py:172–186` (verbatim):

> ```python
> # Model-comparison snapshots (021): one dated snapshot per fetch of the
> # OpenRouter catalog + model-compare's published data.json + the ZDR set.
> model_snapshot_ingest_hour: int = Field(default=6, ge=0, le=23)
> model_snapshot_catalog_url: str = "https://openrouter.ai/api/v1/models"
> model_snapshot_artifact_url: str = "https://canonical.github.io/model-compare/data.json"
> model_snapshot_zdr_url: str = (
>     "https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text&zdr=true"
> )
> ```

`src/web/ingestion/sources/model_comparison.py:41`:
`DEFAULT_ARTIFACT_URL = "https://canonical.github.io/model-compare/data.json"`.
The adapter's module docstring (`:4`): "model-compare's published ``data.json``
supplies the quality index and discount". A grep over `src/`, `frontend/` and
`frontend-react/` for `catalog.json` / `history.json` / `github.io` finds no
model-compare catalog or history fetch anywhere; `frontend/models.html:155`
merely links to `https://github.com/canonical/model-compare#scoring` for the
score definition. `CONFIGURATION.md:257–260` documents the same three URLs.

**E1.4 — No `schema_version` validation of data.json; required-field checks
only.**
The data.json reader is `_quality_by_id(artifact)` in
`src/web/ingestion/sources/model_comparison.py:289–311`. Its entire contract for
the artifact (verbatim):

> ```python
> priorities = _require(artifact, "priorities", "artifact")
> if not isinstance(priorities, dict):
>     raise ModelComparisonSchemaError("artifact: expected a 'priorities' mapping")
> ...
> for profile, rows in priorities.items():
>     ...
>     for row in rows:
>         if not isinstance(row, dict) or not isinstance(row.get("model"), str):
>             raise ModelComparisonSchemaError(f"artifact[{profile}]: row lacks a 'model' id")
>         quality = row.get("quality_index")
>         if quality is not None and row["model"] not in matched:
>             discount = row.get("discount")
> ```

That is: the top-level `priorities` mapping is **required**; each row must be a
dict with a string `model` id; `quality_index` and `discount` are read
**optionally** (`row.get(...)`). The artifact's `schema_version`,
`generated_at`, `best` and `schedules` keys are never read. There is no strict
or loose `schema_version` check on data.json anywhere in the repo. The
`SCHEMA_VERSION = "model-comparison/2"` constant (`src/access/model_comparison.py:38`)
is the **snapshot payload stamp** agent-provisioner writes into its own store —
unrelated to model-compare's data.json `schema_version`.
(`tests/contracts/test_model_comparison_source.py:116` asserts that stamp.)

**E1.5 — Failure mode is loud and non-destructive.**
Adapter docstring (`:20–22`): "Validation is loud (FR-003): a payload that does
not parse or is missing a required field raises :class:`ModelComparisonSchemaError`
and the job fails leaving the previous snapshot intact." The ingest runs daily
at a fixed UTC hour (`model_snapshot_ingest_hour: 6`, `CONFIGURATION.md:257`:
"UTC hour of the daily run, 0-23 (a cron hour, not an interval)").

**E1.6 — The strict catalog.json consumer is a different repo: the OpenCode
plugin.**
`canonical/model-compare-opencode-plugin` (cloned at tag `v0.1.1`) fetches
`https://canonical.github.io/model-compare/catalog.json`
(`src/config.ts:32`: `DEFAULT_CATALOG_URL = "https://canonical.github.io/model-compare/catalog.json"`)
and validates it strictly — `src/core/catalog.ts:20–27` (verbatim):

> ```ts
> // Supported schema: v2 only — upstream's current contract (0.3.0: pricing
> // scalars are effective-tier values; pricing.base/tiers/schedule and a
> // top-level schedules index were added — ignored here, see the roadmap
> // for consumer ideas). The plugin has never been released, so no stored
> // v1 documents exist anywhere; the published catalog has served v2 since
> // 2026-10-10.
> if (raw.schema_version !== 2) {
>   return { ok: false, reason: `schema_version must be 2, got ${JSON.stringify(raw.schema_version)}` }
> }
> ```

(The in-repo comment "has never been released" is stale: the plugin **is
published** — npm `model-compare-opencode-plugin`, dist-tags `latest: 0.1.1`,
created 2026-10-13, GitHub release `v0.1.1` published 2026-10-14 — and the
published 0.1.1 tarball carries the same strict check in
`package/src/core/catalog.ts:26`.) The vendored copy pins
`CATALOG_SCHEMA_VERSION = 2` too (`package/vendor/model_compare.py:89`), so both
plugin sources (remote fetch and vendored script) would reject v3. Its failure
behavior is graceful degradation, per `src/sources/remote.ts` ("A fetch or
validation failure returns the cached record whatever its age; with no usable
record the reason is rejected") and `src/opencode/refresh.ts:5–8` ("on failure
keep going WITHOUT a catalog, so §9's chain (stored pin if runnable → fallback
if runnable → none) is re-evaluated"), plus a one-per-refresh console warning
(`warnDisabledAliases`).

**E1.7 — The published artifacts are public and today's shapes confirm the
contract.**
`https://canonical.github.io/model-compare/catalog.json` and
`.../data.json` both return HTTP 200 anonymously (`access-control-allow-origin: *`,
last-modified 2026-10-20 18:34 UTC). Live `data.json` top-level keys:
`schema_version` (=1 today), `generated_at`, `best`, `priorities`,
`schedules`; `priorities` has profiles `balanced`/`price`/`quality`, and rows
carry `model`, `quality_index`, `discount` (plus many display fields the
consumer ignores).

### Finding

- **tokens.canonical.com (= canonical/agent-provisioner) consumes only
  model-compare's `data.json`, via its backend's daily model-comparison ingest.
  It does not consume `catalog.json` or `history.json` at all.**
- **It does not validate data.json's `schema_version`** — strict or otherwise.
  Its artifact contract is: top-level `priorities` mapping → per-profile lists
  of rows with a string `model` id, optional `quality_index`, optional
  `discount`. Everything else in the artifact is ignored.
- The **strict** `schema_version === 2` consumer of catalog.json is the
  **OpenCode plugin** (`canonical/model-compare-opencode-plugin` 0.1.1, on npm
  since 2026-10-13/14) — not tokens.canonical.com. DESIGN.md's statement that
  the catalog contract is "consumed by internal Canonical tooling
  (`tokens.canonical.com`)" is **contradicted by this evidence** and must be
  corrected in Task 12's "Catalog output contract" rewrite; the catalog's real
  downstream consumer is the plugin, and data.json's consumer is
  agent-provisioner.
- *Limitations:* the live site JS was not inspectable (private Pages, auth
  redirect; E1.1); the conclusion rests on the consumer repo's source at HEAD
  `6eea511` (2026-10-19), which self-identifies as the tokens.canonical.com
  app. GitHub code search covers the default branch of repos the PAT can reach
  (it did return the private repos searched); indexing lag is possible but the
  two consumer repos were also found by direct cloning, so no consumer is known
  to be missing.

### Impact on the plan

- **Catalog `schema_version` 2→3: no impact on tokens.canonical.com** (it never
  fetches catalog.json). **Site `data.json` 1→2: no impact** as long as the v2
  data.json keeps top-level `priorities` (profile → list of rows with string
  `model`, optional `quality_index`, optional `discount`). The design's data.json
  v2 plan (new site columns/sections; per-endpoint blocks stay in catalog.json,
  design §4.5/§8) does not touch `priorities` rows, so the contract survives.
  Even if it ever broke, the failure is loud (daily ingest job fails, previous
  snapshot keeps serving), not silent corruption (E1.5).
- **Go/no-go: GO — the tokens.canonical.com check does not block the first v3
  publish.** This resolves design §8's "block the first v3 publish only if it
  would break" question in favor of shipping, with the dependency noted.
- **New coordination item for the plan (not a blocker, but real):** the OpenCode
  plugin strictly rejects anything but catalog `schema_version: 2` (E1.6). At
  the first v3 publish, its remote source (and its vendored script, which would
  also emit v3) fails validation; users degrade to a cached v2 document served
  "whatever its age", then to stored-pin → fallback-alias → no pick, with
  warnings. A plugin release accepting v3 (validator + vendored
  `CATALOG_SCHEMA_VERSION`) should be coordinated with — or immediately follow —
  the first v3 publish. This is a plan-level note for Task 12 / the publish
  checklist, not a reason to hold the publish.

---

## Question 2 — Why is the models-API advertised price below every routable endpoint?

*Recorded numbers (design brainstorm §2.4, 2026-10-19): advertised
`z-ai/glm-5.3-flash` $0.0184/M input vs cheapest routable endpoint $0.0258/M.
Prices are live data and have moved since — today's figures below differ, but
the *shape* (advertised < every routable endpoint) is unchanged.*

### Evidence

**E2.1 — Live models-API entry (fetched 2026-10-20 ~18:36 UTC,
https://openrouter.ai/api/v1/models, 468 models):**
`z-ai/glm-5.3-flash` (canonical_slug `z-ai/glm-5.3-flash-20260826`):

> ```json
> "pricing": { "prompt": "0.000000048", "completion": "0.00000012", "input_cache_read": "0.000000006" }
> ```

i.e. **$0.048/M input, $0.12/M output, $0.006/M cache-read**. The entry's full
key list is `architecture, benchmarks, canonical_slug, context_length, created,
default_parameters, description, expiration_date, hugging_face_id, id,
knowledge_cutoff, links, name, per_request_limits, pricing, reasoning,
supported_parameters, supported_voices, top_provider` — **there is no
model-level discount or promo field**; the `pricing` object has exactly
`prompt`, `completion`, `input_cache_read`.

**E2.2 — Live endpoints API (https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash-20260826/endpoints,
34 endpoints):**
the advertised $0.048/M **exactly equals** the cheapest *listed* endpoint's base
price — `OpenInference`, tag `open-inference/fp4`,
`"pricing": { "prompt": "0.000000048", "completion": "0.00000012",
"input_cache_read": "0.000000006", "discount": 0 }` — and that endpoint has
`"status": -2`. Every **routable** endpoint (status 0) costs more; cheapest
routable today: `Wafer`, tag `wafer/nvfp4`, prompt `0.000000059` ($0.059/M,
discount 0). Most providers list $0.15/M base ($0.15 prompt / $0.50 completion,
e.g. `Z.AI`'s own endpoint `z-ai/fp8`), some with endpoint-level discounts:
StreamLake `discount: 0.5077`, GMICloud `0.5`, DeepInfra `0.5`, Novita `0.44`,
Phala `0.15`.

**E2.3 — The published catalog corroborates the live figure.**
`https://canonical.github.io/model-compare/catalog.json` (generated
`2026-10-20T18:33:56+00:00`) records for `z-ai/glm-5.3-flash`:
`"input_per_1m": 0.048, "output_per_1m": 0.12, "blended_per_1m": 0.066` — the
advertised scalar, as the models API reports it, flowed into the catalog
unchanged. So the advertised figure moved **$0.0184 → $0.048 within ~a day**,
tracking endpoint churn; it is not a static official price.

**E2.4 — OpenRouter's documented promo machinery is endpoint-level, not
model-level.**
From the OpenAPI spec served by the docs site (source: `OpenRouterTeam/docs`
repo, `openapi/openapi.yaml`, PublicEndpoint schema; rendered at
https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model):

> `discount`: "Fractional discount applied to this endpoint's pricing; the price
> is multiplied by (1 - discount) (0 = no discount, 1 = free)"

> `overrides`: "Conditional overrides of the base pricing (e.g. long-context or
> time-based pricing). An entry applies when all of its condition fields (e.g.
> min_prompt_tokens, or the utc_start/utc_end time window) match the request;
> among applicable entries, later entries win per key; price keys absent from an
> entry inherit the base price. The top-level pricing keys always reflect the
> price that applies under default conditions."

The models-list response's per-model `pricing` is documented only as
`PublicPricing` — "Pricing information for the model" — with **no documented
derivation rule** and no discount/promo keys
(https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties;
no FAQ or guide passage states how the per-model figure is aggregated from
endpoints — searched `OpenRouterTeam/docs` for
"cheapest provider/lowest price/average price/price shown" without a hit).

**E2.5 — Endpoint `status` semantics.**
Public spec `EndpointStatus` is an integer enum `0, -1, -2, -3, -5, -10` with
**no per-value descriptions**. The private-deployment analogue
(`PrivateEndpointStatus`, strings draft/active/disabled) documents:
"`draft` endpoints are not routable until validated and activated; `disabled`
endpoints are activated but temporarily not routable." Reading: `0` = active,
negative = not currently routable. OpenInference's `-2` therefore marks it
non-routable today.

**E2.6 — Cross-model checks rule out one tidy rule but confirm exact-equality
with a specific endpoint's base price.**
Scripted comparison of models-API pricing vs the endpoints API across a sample:

| Model | Advertised prompt | Cheapest listed endpoint (any status) | Cheapest status-0 endpoint | Advertised equals |
|---|---|---|---|---|
| z-ai/glm-5.3-flash | $0.048/M | $0.048/M — OpenInference `open-inference/fp4`, **status −2** | $0.059/M (Wafer) | cheapest listed (non-routable) |
| google/gemini-2.5-flash | $0.30/M | $0.15/M — Google AI Studio `google-ai-studio/flex`, status 0 | $0.15/M | the **default-tier** endpoint `google-ai-studio` ($0.30, status 0) |
| openai/gpt-6-sol-pro | $2.00/M | $1.00/M — OpenAI `openai/flex`, status 0 | $1.00/M | the **default-tier** endpoint `openai` ($2.00, status 0); its models-API pricing even carries `overrides` |
| openai/gpt-5.4-image-2, openai/o3-mini, tencent/hy-mt2-1.8b, z-ai/glm-4.5v, bytedance-seed/seed-2.0-lite | — | — | — | cheapest (single-provider, no service tiers) |

So the advertised scalar always **equals some endpoint's base `pricing.prompt`
exactly**, but *which* endpoint differs: the default-tier endpoint for
models that have service-tier variants (flex/priority/fast), the cheapest listed
endpoint for glm-5.3-flash (whose cheapest is a sub-default-quantization fp4
endpoint, currently non-routable). No single documented rule reproduces all
cases, and the spec does not state one.

### Finding

- **The advertised scalar is not a promo artifact of OpenRouter's** in the sense
  of a model-level discount: the models API has no model-level
  discount/promo field at all (E2.1, E2.4), and endpoint-level `discount`
  multiplies an endpoint's own base price at billing time without feeding the
  models-list scalar (E2.4).
- **Best supported explanation (evidence-backed, mechanism undocumented):** the
  per-model advertised pricing is derived from **an endpoint's base pricing**
  — exact equality in every case examined — with the chosen endpoint being
  the default service tier where one exists, and (for glm-5.3-flash) the
  cheapest *listed* endpoint, which may be **non-routable** (`status: -2`).
  That non-routability is why the advertised figure can sit below every
  *routable* endpoint, both yesterday ($0.0184 vs $0.0258) and today
  ($0.048 vs $0.059). The figure moves with endpoint churn (E2.3).
- **The precise general derivation rule remains undocumented and could not be
  pinned from outside** (E2.6 counterexamples against "cheapest listed" and
  "cheapest routable" as universal rules; no docs passage states the rule).
  Verdict for the plan: **unresolved at the level of a general rule — treat as
  documentation-only annotation**, exactly as design §4.5 already decided:
  retain the scalar as `advertised_pricing` (scalars only, `source:
  "models-api"`), "nothing may interpret it", "no consumer reads the advertised
  price for ranking".

### Impact on the plan

- No change to the design: the `advertised_pricing` annotation field (§4.5,
  "Pricing block coherence under v3") is the right treatment; the investigation
  adds the evidence quotes above rather than new behavior. Any wording that
  hints the advertised scalar is "a promo/discount artifact" should be softened
  to "endpoint-derived; exact rule undocumented" (Task 12 wording input).
- The design's §8 history-reset rationale already anticipated the semantics
  change (old snapshots = advertised scalars, new = baseline-endpoint prices,
  e.g. "$0.0184 → $0.15 per 1M input"); today's data confirms $0.15/M is the
  Z.AI baseline endpoint's base input price, consistent with that example.

---

## Summary of the go/no-go recommendation

**GO for the first v3 publish.** tokens.canonical.com consumes only
`data.json`, validates nothing about `schema_version`, and tolerates the planned
v2 data.json shape (its `priorities` contract is untouched by the design);
failure, if the contract ever broke, would be loud and non-destructive. The one
real downstream risk is the **OpenCode plugin** (`schema_version === 2` strict
check in released npm 0.1.1): it degrades gracefully (cached v2 doc → stored pin
→ fallback alias → none, with warnings) but should receive a v3-accepting
release in coordination with the first v3 publish.
