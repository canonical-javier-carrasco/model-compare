# model-compare

A single-file Python CLI that picks the best value-for-money LLM on
[OpenRouter](https://openrouter.ai/) before you start a coding agent. It queries
the OpenRouter catalog, scores every candidate on quality, price, context
window and freshness, and prints the winners — either as a ranked table or as a
single parseable `provider/model` id.

Built by an AI coding agent (opencode, powered mostly by GLM 5.3 Flash).

## Quick start

```console
$ ./model_compare.py                          # top 5, balanced
$ ./model_compare.py --best                   # only "#1 model" as an opencode id
$ MODEL=$(./model_compare.py --best)          # shell integration
$ opencode --model $MODEL
$ ./model_compare.py --priority price --top 3
```

`--best` prints the pick ready to use: provider-qualified as opencode
expects, e.g. `openrouter/z-ai/glm-5.3-flash` (`openrouter/` comes from a
single constant in the script, so other providers can be supported later).
The `--json` output carries both forms: `model` (catalog id) and
`opencode_model` (qualified id), plus `discount_pct`, the `DISC` column's
display string (e.g. `25%` or `--`).

No dependencies beyond Python 3.10+ (stdlib only). Exit codes: `0` success,
`1` fetch failure, `2` no candidates survive the filters.

## Options

| flag | default | meaning |
|------|---------|---------|
| `--priority` | `balanced` | `price`, `quality` or `balanced` |
| `--top N` | 5 | how many models to list |
| `--best` | off | print only the #1 model id (for scripting) |
| `--json` | off | machine-readable output |
| `--catalog` | off | print the full model catalog (ranked candidates + filtered, with reasons, plus the per-priority model ranking) as one JSON document |
| `--min-context N` | 1000000 | hard context-window floor (tokens) |
| `--input-share F` | 0.75 | input share of the blended price (0–1) |
| `--recency-half-life D` | 120 | age decay half-life (days) |
| `--max-age-days D` | off | drop models listed more than N days ago |
| `--quality-ref N` | 70 | index counting as full quality score |
| `--aa-api-key KEY` | `$AA_API_KEY` | Artificial Analysis API key |
| `--exclude-free` | off | drop `:free` variants |
| `--include-batch` | off | keep `:batch` (async completion) variants; no effect without `--no-zdr`, as the ZDR list has no `:batch` ids today |
| `--discount` | off | only models with an active discount |
| `--no-zdr` | off | rank all models, not just zero-data-retention ones |
| `--no-require-tools` | off | allow models without tool calling |
| `--no-cache` / `--cache-ttl S` | 6h | cache control |
| `--version` | off | print `model-compare <VERSION>` and exit |

## Published picks

No need to run anything: a [GitHub Actions workflow](web/README.md) refreshes
[the picks](https://canonical.github.io/model-compare/) every 3 hours, and
`best.txt` always serves the current *balanced* #1 as a plain model id:

```console
$ opencode --model "$(curl -fsSL https://canonical.github.io/model-compare/best.txt)"
$ alias oc-best='opencode --model "$(curl -fsSL https://canonical.github.io/model-compare/best.txt)"'
```

The site also serves the machine-readable `catalog.json` (`--catalog` output)
plus weekly `history.json` and `highlights.json`. How that pipeline works:
[web/README.md](web/README.md).

## How it works

OpenRouter's public catalog provides prices, context windows and listing
dates; model quality comes from the Artificial Analysis intelligence index,
resolved through a three-source hierarchy with visible provenance (see
[Model quality](#model-quality) below). Each candidate is scored on quality,
blended price, context headroom and listing age with weights depending on
`--priority`, after hard filters (context floor, text-only, tool-calling,
ZDR by default). The full scoring formulas, filter list, the `--catalog`
JSON contract and the test setup are in [DESIGN.md](DESIGN.md).

## Model quality

Quality is the [Artificial Analysis](https://artificialanalysis.ai/models)
intelligence index (0–100). Each model's value is resolved through three
sources in a strict order — rank, not freshness: a higher-ranked source
always wins, even when a lower-ranked one is newer.

1. **Live AA API** — with `--aa-api-key` (or `AA_API_KEY`) set, the AA v2
   API is consulted first; this is the live index.
2. **Our published catalog** — without a key, quality hydrates from
   [model-compare.dev/catalog.json](https://model-compare.dev/catalog.json),
   the artifact this repo's keyed CI runs publish, joined by exact model id.
3. **OpenRouter snapshot** — the AA indices OpenRouter republishes per
   listing, used only for models neither AA path covers.

Every quality value carries its provenance end to end: `--json` rows carry
`quality_source` and `quality_as_of`, `catalog.json` entries and the site's
QUAL tooltips show them, and the CLI's footer note lists how many models
came from each source and when each was checked.

**Why our QUAL can differ from openrouter.ai:** OpenRouter republishes a
**frozen copy** of the intelligence index per dated listing, while the AA
API is live — a model whose index has moved since OpenRouter's snapshot
will legitimately show different numbers here. When the winning value and
OpenRouter's snapshot for the same model differ by 3 or more index points,
the row is flagged as drift (`quality_drift`, with `quality_or_index` and
`quality_or_as_of` naming the snapshot it was compared against) — visible as
a tooltip note on the site and counted in the CLI footer. Prices can differ
from the website too, for a different reason: this tool ranks at the
cheapest eligible endpoint, not the default one (see
[Tiered pricing](#tiered-pricing)). Provenance — the source plus its as-of —
makes every such discrepancy explainable instead of mysterious; the
background is in issues
[#17](https://github.com/canonical/model-compare/issues/17) and
[#18](https://github.com/canonical/model-compare/issues/18).

## Tiered pricing

OpenRouter publishes conditional price overrides per model
(`pricing.overrides`): long-context tiers that charge more above a prompt
threshold, and off-peak schedules that charge less during certain UTC
windows. This tool ranks every model at **the tier in effect at a prompt
size of `min_context` tokens** — with the default 1M context that is the
worst case, which is what a long coding session actually pays. Time-windowed
models are ranked at their peak window (deterministic) and annotated; models
whose windows do not tile the week exactly are dropped (`schedule` drop
reason). The table's TIER column marks a token threshold (`>100k`), an
off-peak discount (`-38%`), or a sub-1% schedule (`SCHED`);
`--json` carries `pricing_tier_prompt_tokens`, `base_input_usd_per_m`,
`base_output_usd_per_m`, `tier_note`, `sched_note`, `sched_detail` and
`time_schedule`. `catalog.json` publishes the full picture: `pricing.base`,
`pricing.tiers`, `pricing.schedule` (with per-window `coverage` fractions
for time-weighted pricing) and the top-level `schedules` deals index.

## License

GPL-3.0 — see [LICENSE](LICENSE).
