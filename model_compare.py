#!/usr/bin/env python3
"""model-compare: pick the best value-for-money LLM on OpenRouter.

Run it before launching a coding agent to auto-select the model that gives
the most capability per dollar today. Data sources:

  * OpenRouter public API (/api/v1/models) - catalog, pricing (input and
    output), context window, release date and capabilities.
  * Artificial Analysis intelligence index - model quality. Primary
    source: OpenRouter's republished AA benchmarks (data.benchmarks in
    the frontend models API, exact per-model keys). When a model is not
    covered there: the AA API v2 (--aa-api-key or the AA_API_KEY
    environment variable; free key at artificialanalysis.ai), matched by
    exact slug/name only. Without a key, AA data comes only from
    OpenRouter. Unmatched models rank on price/context/age.

Ranking: every criterion is normalized to [0, 1] and combined with
priority-dependent weights:

    balanced:  quality 0.40 / price 0.40 / context 0.10 / age 0.10
    price:     quality 0.20 / price 0.60 / context 0.10 / age 0.10
    quality:   quality 0.60 / price 0.20 / context 0.10 / age 0.10

  price_score    log-scaled blended price (USD per 1M tokens,
                 --input-share input / rest output) within the candidate
                 pool; free listings score 1.0.
  quality_score  AA intelligence index divided by --quality-ref (default
                 70), clamped to [0, 1]; unmatched models score 0.
  context_score  logarithmic bonus from --min-context up to 4x that
                 threshold (extra headroom helps, but with diminishing
                 returns).
  age_score      exponential decay with --recency-half-life (default 120
                 days): the fresher the listing, the better.

Examples:
  model_compare.py                          top 5, balanced
  model_compare.py --best                   print only "#1 opencode id" (openrouter/<model>)
  model_compare.py --priority price --top 3 three cheapest-sensible picks
  model_compare.py --discount               only currently-discounted models
  model_compare.py --catalog                full catalog (ranked + filtered) as one JSON document
  MODEL=$(model_compare.py --best)          shell integration (opencode --model $MODEL)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_DISCOUNTS_URL = (
    "https://openrouter.ai/api/frontend/v1/models/find?output_modalities=text"
)
# Public per-endpoint list of zero-data-retention endpoints; DESIGN.md
# ("Zero data retention") describes how its entries are interpreted.
OPENROUTER_ZDR_URL = "https://openrouter.ai/api/v1/endpoints/zdr"
# Supported V2 free-tier replacement for the retired /api/v2/data/llms/models
# (legacy endpoints 410 after 2026-11-04; see
# https://artificialanalysis.ai/data-api/migrate-v2-data).
AA_API_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
VERSION = "0.4.0"
USER_AGENT = f"model-compare/{VERSION} (https://github.com/canonical/model-compare)"

# opencode expects provider-qualified model ids: openrouter/<provider/model>.
# Single source of truth -- other provider namespaces can be supported later
# without touching consumers of --best / --json.
OPENCODE_PROVIDER = "openrouter"


def opencode_model_id(model_id: str) -> str:
    return f"{OPENCODE_PROVIDER}/{model_id}"


PRIORITY_WEIGHTS = {
    "balanced": {"quality": 0.40, "price": 0.40, "context": 0.10, "age": 0.10},
    "price": {"quality": 0.20, "price": 0.60, "context": 0.10, "age": 0.10},
    "quality": {"quality": 0.60, "price": 0.20, "context": 0.10, "age": 0.10},
}

CATALOG_SCHEMA_VERSION = 3

# The drop-reason keys build_candidates counts, verbatim -- keep in sync with
# its drop() call sites. pool.dropped lists all of them zero-filled so the
# contract is self-describing. The last two come from the endpoint stage
# (apply_endpoint_pricing's drop sites), not build_candidates: snake_case
# while the older reasons are human phrases. The mixed naming convention is
# a deliberate, documented extension -- machine-readable reasons for the
# endpoint-stage drops the publish drop-rate gate keys on.
CATALOG_DROP_REASONS = (
    "malformed id",
    "context",
    "pricing",
    "schedule",
    "free",
    "batch",
    "no discount",
    "not ZDR",
    "modality",
    "tool calling",
    "expired",
    "age",
    "no_baseline_eligible_endpoint",
    "endpoint_fetch_failed",
)

# The only reasons the endpoint stage itself produces (the last two
# CATALOG_DROP_REASONS entries) -- the empty-pool warning keys on them;
# web/publish.py keeps its own copy of this tuple by design.
ENDPOINT_DROP_REASONS = (
    "no_baseline_eligible_endpoint",
    "endpoint_fetch_failed",
)

PAREN_RE = re.compile(r"\([^)]*\)")
PROVIDER_PREFIX_RE = re.compile(r"^[^:\s]{1,30}:\s*")

# Quantization precision ladder: label -> rank, highest fidelity first. The
# ladder ordering in the design doc is authoritative: fp6 > int8 is strict,
# and formats within one fidelity tier share the same rank. "unknown" never
# enters the ladder -- it only routes through the first-party eligibility
# carve-out, never through rank comparison.
PRECISION_LADDER: dict[str, int] = {
    "fp32": 4,
    "bf16": 3,
    "fp16": 3,
    "fp8": 2,
    "mxfp8": 2,
    "fp6": 1,
    "int8": 0,
    "fp4": -1,
    "mxfp4": -1,
    "nvfp4": -1,
    "int4": -2,
}

# Every quantization value an endpoint may report: the ladder keys plus the
# "unknown" sentinel. Whether a recorded value is recognized at all is
# membership here, not ladder rank.
QUANTIZATION_ENUM: frozenset[str] = frozenset(PRECISION_LADDER) | {"unknown"}

# Official partner hosts that serve closed labs' models: an endpoint on one
# of these providers counts as first-party for the eligibility carve-out
# even though it is not the maker itself. Entries are compared after
# normalize_provider (casefold + strip non-alphanumerics), so the slugs here
# are display forms -- "aws-bedrock" and "AWS Bedrock" both hit. Extend only
# with live-data evidence (e.g. an endpoint tag matching <lab>-official).
FIRST_PARTY_HOSTS = frozenset(
    {"bedrock", "azure", "vertex", "azure-ai", "google-vertex", "aws-bedrock"}
)

# Explicit first-party alias pairs that normalization alone cannot bridge
# (design §4.2): makers that self-name differently than their OpenRouter
# author slug. Key = the exact live provider_name string, value = the
# author slug the pair is first-party FOR; both sides are compared
# normalized (case/punctuation variants match), and the alias only fires
# when the model's author equals the value. Verified against live data
# 2026-10-20 (task-13-fix-report.md section 8b; endpoints fetched fresh,
# every endpoint quantization "unknown", status 0):
#   ("Amazon Bedrock", "amazon")   -- amazon/nova-2-lite-v1, 2 endpoints
#   ("Google AI Studio", "google") -- google/gemini-2.5-pro-preview, 3 endpoints
#   ("Mistral", "mistralai")       -- mistralai/mistral-large-4-0, 3 endpoints
# Add entries only after the same live verification -- never pin an
# unverified string.
FIRST_PARTY_ALIASES: dict[str, str] = {
    "Amazon Bedrock": "amazon",
    "Google AI Studio": "google",
    "Mistral": "mistralai",
}


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ---------------------------------------------------------------------------
# Provider normalization + first-party detection
# ---------------------------------------------------------------------------


def normalize_provider(name: str) -> str:
    """Casefold a provider name and strip ALL non-alphanumeric characters.

    "Z.AI", "z-ai" and "z ai" all normalize to "zai", so a live-recorded
    maker name matches its author slug regardless of punctuation or case.
    None or empty input normalizes to "".
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def is_first_party(provider_name: str, author: str) -> bool:
    """True iff the provider is the model's own maker or a partner host.

    Criterion 1: the normalized provider name equals the normalized author
    slug. Criterion 2: it matches a normalized FIRST_PARTY_HOSTS entry, or
    a FIRST_PARTY_ALIASES key whose value equals the normalized author slug
    (the §4.2 alias pairs: a maker self-naming differently than its author
    slug is first-party only for that author). Everything else is
    third-party.
    """
    provider = normalize_provider(provider_name)
    if provider == normalize_provider(author):
        return True
    if provider in {normalize_provider(host) for host in FIRST_PARTY_HOSTS}:
        return True
    author_slug = normalize_provider(author)
    return any(
        provider == normalize_provider(alias_provider)
        and author_slug == normalize_provider(alias_author)
        for alias_provider, alias_author in FIRST_PARTY_ALIASES.items()
    )


# ---------------------------------------------------------------------------
# HTTP + cache
# ---------------------------------------------------------------------------


class StripApiKeyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop the AA API key when a redirect leaves the original origin.

    urllib copies custom request headers onto the redirected request even
    across hosts. The key is kept only when scheme, host and port all match,
    so a cross-host redirect or an https->http downgrade never carries it.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old, dest = (
                urllib.parse.urlsplit(req.full_url),
                urllib.parse.urlsplit(new.full_url),
            )
            same_origin = (old.scheme, old.hostname, old.port) == (
                dest.scheme,
                dest.hostname,
                dest.port,
            )
            if not same_origin:
                # Request stores header names capitalize()d: "X-api-key".
                new.remove_header("X-api-key")
        return new


def http_get(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip",
            **(headers or {}),
        },
    )
    opener = urllib.request.build_opener(StripApiKeyRedirectHandler)
    with opener.open(req, timeout=timeout) as resp:
        data = resp.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def fetch_json(
    url: str, headers: dict | None = None, timeout: int = 30, retries: int = 1
):
    last_exc: Exception = RuntimeError("request failed")
    for attempt in range(retries + 1):
        try:
            return json.loads(http_get(url, headers, timeout))
        except urllib.error.HTTPError as exc:
            # HTTPError is a URLError, so this clause must come first. A
            # 4xx (bad key, forbidden, rate limited) will not change on an
            # immediate retry; raise it at once. 5xx falls through to retry.
            if 400 <= exc.code < 500:
                raise
            last_exc = exc
            if attempt < retries:
                time.sleep(1)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1)
    raise last_exc


def cache_path(name: str):
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return base and os.path.join(base, "model-compare", f"{name}.json")


def load_cache(name: str, ttl_seconds: float):
    path = cache_path(name)
    if not path:
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        if time.time() - blob["fetched_at"] <= ttl_seconds:
            return blob["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def save_cache(name: str, payload) -> None:
    path = cache_path(name)
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as fh:
            json.dump({"fetched_at": time.time(), "payload": payload}, fh)
        os.replace(tmp_path, path)
    except OSError as exc:
        warn(f"could not write cache: {exc}")


# ---------------------------------------------------------------------------
# OpenRouter endpoints (per-model provider list)
# ---------------------------------------------------------------------------


OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{id}/endpoints"


def load_cache_expired(name: str) -> dict | None:
    """Return the cached payload regardless of age; None when absent/corrupt.

    load_cache drops payloads past their TTL; this variant serves them anyway
    so a dropped model's last known endpoint annotations can still be shown
    (marked stale by the caller). Never used for ranking.
    """
    path = cache_path(name)
    if not path:
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        return blob["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_endpoints(payload) -> list[dict]:
    """Keep the verified per-endpoint fields of an endpoints response.

    The live GET /api/v1/models/{id}/endpoints response nests the list one
    level below the usual envelope: `data` is the model object (id, name,
    architecture, ...) and the per-provider entries live in
    `data["endpoints"]`. `pricing` keeps prompt/completion (parsed through
    parse_price) and the raw `overrides` list -- the tiered-pricing
    machinery parses the entries itself. Fields beyond the verified
    per-entry shape are dropped, and the parse keeps `status` as-is (the
    caller filters).
    """
    # Task 13 lesson: the first unit fixtures fed the models/find shape
    # ({"data": [entry, ...]}) -- a list directly under "data" -- so the
    # suite stayed green while every live parse returned [] and the whole
    # pool died as no_baseline_eligible_endpoint. Pin the live envelope:
    # the list is read ONLY from data["endpoints"], and every other shape
    # parses to [] (no compat branch for the wrong fixture shape).
    data = payload.get("data") if isinstance(payload, dict) else None
    entries = data.get("endpoints") if isinstance(data, dict) else None
    endpoints = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("pricing")
        raw_pricing = raw if isinstance(raw, dict) else {}
        endpoints.append(
            {
                "provider_name": entry.get("provider_name"),
                "tag": entry.get("tag"),
                "quantization": entry.get("quantization"),
                "pricing": {
                    "prompt": parse_price(raw_pricing.get("prompt")),
                    "completion": parse_price(raw_pricing.get("completion")),
                    "overrides": raw_pricing.get("overrides"),
                },
                "context_length": entry.get("context_length"),
                "status": entry.get("status"),
                "service_tier": entry.get("service_tier"),
            }
        )
    return endpoints


def fetch_model_endpoints(model_id: str, args) -> list[dict]:
    """Fetch + parse one model's endpoint list, cached per model.

    The cache name flattens the model id ("/" -> "_") to keep the cache
    files flat. Fetches retry 3 times (4 attempts total) via fetch_json,
    which raises at once on 4xx and after the retries on 5xx/network
    errors -- the caller owns the fail-closed policy.
    """
    cache_name = f"endpoints:{model_id.replace('/', '_')}"
    if not args.no_cache:
        cached = load_cache(cache_name, args.cache_ttl)
        if cached:
            return cached
    payload = fetch_json(
        OPENROUTER_ENDPOINTS_URL.format(id=model_id), timeout=30, retries=3
    )
    endpoints = parse_endpoints(payload)
    save_cache(cache_name, endpoints)
    return endpoints


def is_variant_endpoint(endpoint) -> bool:
    """True iff the endpoint is a :free/:batch variant or entirely free.

    Variant iff its slug or tag carries a `:free`/`:batch` marker OR both
    input and output prices are zero. `:batch` is cheaper-but-restricted,
    not zero-priced -- only the marker test catches it. Missing or
    unparseable prices are not zero.
    """
    for field in ("slug", "tag"):
        marker = endpoint.get(field)
        if isinstance(marker, str) and (":free" in marker or ":batch" in marker):
            return True
    pricing = endpoint.get("pricing")
    prices = pricing if isinstance(pricing, dict) else {}
    return (
        parse_price(prices.get("prompt")) == 0.0
        and parse_price(prices.get("completion")) == 0.0
    )


def endpoint_schedule_ok(endpoint, args) -> bool:
    """False iff the endpoint's pricing.overrides fails the v2 schedule rule.

    Reuses effective_pricing in full: the gate fires on malformed windows,
    exactly-one-of utc_start/utc_end pairs, bad utc_days, combined
    token+time entries and missing inheritance -- not just non-tiling
    windows. An endpoint without overrides has no schedule to fail.
    """
    result = effective_pricing(
        endpoint.get("pricing"), args.min_context, args.input_share
    )
    return not result["schedule_error"]


# Schedule gate for args-less callers: effective_pricing's schedule_error
# depends only on the SHAPE of pricing.overrides (malformed windows, bad
# utc_days, ...), never on min_context/input_share -- those only affect the
# resolved prices. derive_default_precision has no args parameter (pinned
# interface), so it gates through this fixed stand-in; any args would give
# the same schedule_error verdict.
_SCHEDULE_GATE_ARGS = argparse.Namespace(min_context=0, input_share=0.75)


def _baseline_universe(endpoints: list[dict], args=None) -> list[dict]:
    """The shared derivation universe (design section 4.1).

    Live (status == 0), not a variant endpoint (:free/:batch marker or
    zero-priced), and passing the schedule gate. Schedule-broken endpoints
    must not pin the default that governs healthy endpoints, same rationale
    as the variant exclusion. args defaults to the structural stand-in
    (see _SCHEDULE_GATE_ARGS); callers with real args pass them.
    """
    if args is None:
        args = _SCHEDULE_GATE_ARGS
    return [
        endpoint
        for endpoint in endpoints
        if endpoint.get("status") == 0
        and not is_variant_endpoint(endpoint)
        and endpoint_schedule_ok(endpoint, args)
    ]


def derive_default_precision(endpoints: list[dict], author: str) -> str | None:
    """Derive a model's default precision from its endpoint list (section 4.1).

    Universe: live non-variant schedule-passing endpoints (_baseline_universe).
    Primary rule: the highest-ranked KNOWN quantization among the maker's
    endpoints -- matched by normalized author-slug equality only, so partner
    hosts never pin the maker's default (section 4.2 criterion 2 does not
    apply here). Fallback: when the maker has no known quantization, the
    highest-ranked known quantization across the FULL universe (partners
    included -- best-available). None when no endpoint in the universe has a
    known quantization (the all-unknown closed-model case): eligibility then
    relies solely on the first-party carve-out.

    Known means a PRECISION_LADDER key: "unknown" and unparseable/absent
    values never enter the ladder. An endpoint whose provider_name is
    absent/empty (or normalizes to empty) is skipped for maker matching
    (never first-party -- fail-closed), but still feeds the fallback
    universe.
    """
    universe = _baseline_universe(endpoints, _SCHEDULE_GATE_ARGS)
    maker = normalize_provider(author)
    maker_best = None
    fallback_best = None
    for endpoint in universe:
        quantization = endpoint.get("quantization")
        if not isinstance(quantization, str) or quantization not in PRECISION_LADDER:
            continue  # "unknown", null, unrecognized: no rank, no pinning
        rank = PRECISION_LADDER[quantization]
        if fallback_best is None or rank > fallback_best[0]:
            fallback_best = (rank, quantization)
        # Maker matching only: a provider_name that normalizes to nothing
        # must never match an (equally empty) author slug.
        provider_slug = (
            normalize_provider(endpoint.get("provider_name"))
            if isinstance(endpoint.get("provider_name"), str)
            else ""
        )
        if (
            provider_slug
            and provider_slug == maker
            and (maker_best is None or rank > maker_best[0])
        ):
            maker_best = (rank, quantization)
    return (maker_best or fallback_best or (None, None))[1]


def baseline_eligible(endpoint, default_precision, author, args) -> bool:
    """Baseline-eligibility rule (design section 4.3, fail-closed).

    True iff the endpoint is live (status == 0), not a variant, passes the
    schedule gate, and then either:

    * its quantization is a known ladder label with rank >=
      rank(default_precision) -- provider-blind (default_routing considers
      every endpoint OpenRouter may send traffic to), or
    * its quantization is "unknown" or null AND the endpoint is first-party
      (the D3 carve-out: the maker or a partner host).

    When default_precision is None (all-unknown model) the rank comparison is
    meaningless -- only the carve-out applies. A provider_name that is
    absent/empty (or normalizes to empty) is never first-party, so such an
    endpoint can only qualify through the rank branch. An unrecognized
    quantization string (neither a ladder label nor "unknown"/null) is a
    schema anomaly: no rank comparison and no carve-out -- fail-closed
    (fleet-wide new-label drift is what the publish drop-rate gate exists
    to catch).
    """
    if (
        endpoint.get("status") != 0
        or is_variant_endpoint(endpoint)
        or not endpoint_schedule_ok(endpoint, args)
    ):
        return False
    quantization = endpoint.get("quantization")
    if isinstance(quantization, str) and quantization in PRECISION_LADDER:
        # Known quantization: the rank branch, provider-blind.
        if default_precision is None:
            return False
        return PRECISION_LADDER[quantization] >= PRECISION_LADDER[default_precision]
    # Unverifiable: the carve-out applies to "unknown" and null only --
    # an unrecognized string fails closed. A provider_name that normalizes
    # to nothing is never first-party (fail-closed; the literal
    # is_first_party("", "") iff is guarded out here).
    provider_name = endpoint.get("provider_name")
    provider_slug = (
        normalize_provider(provider_name) if isinstance(provider_name, str) else ""
    )
    return (
        (quantization is None or quantization == "unknown")
        and bool(provider_slug)
        and is_first_party(provider_name, author)
    )


def pick_baseline(
    endpoints: list[dict], eligible_flags: list[bool], args
) -> dict | None:
    """Pick the baseline endpoint among the eligible ones (design section 4.5).

    Cheapest blended per-token price wins, computed from effective_pricing
    with the args' min_context and input_share (the shared --input-share
    convention). Ties broken by (normalized provider name, tag), then --
    still tied -- first in list order (deterministic; Python's sort is
    stable and the endpoints list order is the API's). Returns None when no
    eligible endpoint has a usable price (the caller owns the drop).
    """
    candidates = []
    for endpoint, eligible in zip(endpoints, eligible_flags):
        if not eligible:
            continue
        eff = effective_pricing(
            endpoint.get("pricing"), args.min_context, args.input_share
        )
        price_in, price_out = eff["price_in"], eff["price_out"]
        if (
            price_in is None
            or price_out is None
            or not math.isfinite(price_in)
            or not math.isfinite(price_out)
        ):
            continue  # unpriceable: no executable blended price
        blended = args.input_share * price_in + (1 - args.input_share) * price_out
        provider_name = endpoint.get("provider_name")
        candidates.append(
            (
                blended,
                normalize_provider(provider_name)
                if isinstance(provider_name, str)
                else "",
                endpoint.get("tag") or "",
                endpoint,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[:3])[3]


def _blended_price(endpoint, args) -> float | None:
    """Blended per-token price of one endpoint (pick_baseline's price rule).

    effective_pricing with the args' min_context and input_share, then
    input_share * price_in + (1 - input_share) * price_out. None when either
    resolved price is missing or non-finite -- no executable blended price,
    the same skip pick_baseline applies. A zero result is returned as-is:
    zero-priced endpoints are variants (already excluded upstream), so the
    callers apply their own zero guards defensively.
    """
    eff = effective_pricing(endpoint.get("pricing"), args.min_context, args.input_share)
    price_in, price_out = eff["price_in"], eff["price_out"]
    if (
        price_in is None
        or price_out is None
        or not math.isfinite(price_in)
        or not math.isfinite(price_out)
    ):
        return None
    blended = args.input_share * price_in + (1 - args.input_share) * price_out
    return blended if math.isfinite(blended) else None


def _flag_list(endpoints: list[dict], eligible_flags: list[bool]):
    """Zip endpoints with their parallel eligibility flags, hardened.

    eligible_flags is a parallel list to endpoints (the baseline_eligible
    verdict per endpoint, same order -- Task 4 contract). A length mismatch
    must raise, never truncate silently (Task 4 review ruling).
    """
    if len(endpoints) != len(eligible_flags):
        raise ValueError(
            "eligible_flags must be a parallel list to endpoints: "
            f"{len(endpoints)} endpoints, {len(eligible_flags)} flags"
        )
    return zip(endpoints, eligible_flags, strict=True)


def NON_ELIGIBLE_UNIVERSE(endpoints: list[dict], eligible_flags: list[bool]) -> list:
    """The shared annotations universe (design section 4.6).

    Live (status == 0), non-variant endpoints that are NOT baseline-eligible:
    sub-default known quantization, third-party unknown/null quantization, or
    schedule-gate-failed. All annotations derive from this one universe.
    Eligibility comes from the parallel eligible_flags list (the
    baseline_eligible verdicts); variants and non-live endpoints are excluded
    even though they too are not eligible -- unlike the steering `ignore`
    list, which includes live variants (section 4.7).

    Raises ValueError when eligible_flags does not match endpoints in length.
    """
    return [
        endpoint
        for endpoint, eligible in _flag_list(endpoints, eligible_flags)
        if endpoint.get("status") == 0
        and not is_variant_endpoint(endpoint)
        and not eligible
    ]


def build_annotations(
    endpoints: list[dict],
    eligible_flags: list[bool],
    baseline,
    catalog_context,
    args,
) -> dict:
    """Per-model endpoint annotations (design section 4.6).

    All four annotations derive from one shared universe -- the live
    non-variant endpoints that are not baseline-eligible
    (NON_ELIGIBLE_UNIVERSE):

    * price_spread: cheapest non-eligible blended price divided by the
      cheapest eligible blended price, both blended with the args'
      min_context and input_share (the shared --input-share convention).
      None when the universe is empty -- generalized to None when either
      side has no priceable member at all -- and None when either cheapest
      price is 0 (zero-priced endpoints are variants by definition, so the
      zero half of the guard is defensive only). Unpriceable endpoints are
      skipped when looking for a cheapest price.
    * cheapest_quantized: {provider, quantization, price} of the cheapest
      member of the quantization-caused subset of the universe -- endpoints
      non-eligible due to quantization only (sub-default known quantization
      or third-party unknown/null), NOT schedule-gate-failed ones, which are
      excluded even when cheaper. Unpriceable subset members are skipped
      (defensively also non-positive ones, matching the repo-wide treatment
      of negative prices as malformed); None when the subset has no
      priceable member. Price ties keep the first in list order
      (deterministic).
    * risk_flag: True iff the sum of 1/blended**2 over the universe strictly
      exceeds the sum over the baseline-eligible endpoints (exact tie ->
      False). Any endpoint whose blended price is 0/None (defensively also
      negative) is skipped from BOTH sums -- variants are already excluded,
      so this only guards malformed data, and the flag stays deterministic.
    * baseline_context_below_catalog: True iff the baseline endpoint's
      context_length is smaller than the catalog context (the models-API
      value). A None baseline -- or missing context on either side -- yields
      False: the caveat marks a positive finding, never missing data.

    Raises ValueError when eligible_flags does not match endpoints in length.
    """
    universe = NON_ELIGIBLE_UNIVERSE(endpoints, eligible_flags)

    universe_prices = []
    for endpoint in universe:
        price = _blended_price(endpoint, args)
        if price is not None:
            universe_prices.append(price)
    eligible_prices = []
    for endpoint, eligible in _flag_list(endpoints, eligible_flags):
        if not eligible:
            continue
        price = _blended_price(endpoint, args)
        if price is not None:
            eligible_prices.append(price)
    cheapest_non_eligible = min(universe_prices) if universe_prices else None
    cheapest_eligible = min(eligible_prices) if eligible_prices else None
    price_spread = (
        cheapest_non_eligible / cheapest_eligible
        if cheapest_non_eligible is not None
        and cheapest_eligible is not None
        and cheapest_non_eligible > 0
        and cheapest_eligible > 0
        else None
    )

    quantized_best = None
    for endpoint in universe:
        if not endpoint_schedule_ok(endpoint, args):
            continue  # schedule-gate-failed: not a quantization-caused exclusion
        price = _blended_price(endpoint, args)
        if price is None or price <= 0:
            continue  # unpriceable/non-positive: no comparable price
        if quantized_best is None or price < quantized_best[0]:
            quantized_best = (price, endpoint)  # ties keep the first in list order
    if quantized_best is None:
        cheapest_quantized = None
    else:
        price, endpoint = quantized_best
        cheapest_quantized = {
            "provider": endpoint.get("provider_name"),
            "quantization": endpoint.get("quantization"),
            "price": price,
        }

    non_eligible_weight = 0.0
    for endpoint in universe:
        price = _blended_price(endpoint, args)
        if price is None or price <= 0:
            continue  # skipped from BOTH sums, deterministically
        non_eligible_weight += 1.0 / (price * price)
    eligible_weight = 0.0
    for endpoint, eligible in _flag_list(endpoints, eligible_flags):
        if not eligible:
            continue
        price = _blended_price(endpoint, args)
        if price is None or price <= 0:
            continue
        eligible_weight += 1.0 / (price * price)
    risk_flag = non_eligible_weight > eligible_weight

    baseline_context = baseline.get("context_length") if baseline is not None else None
    baseline_context_below_catalog = (
        baseline_context is not None
        and catalog_context is not None
        and baseline_context < catalog_context
    )

    return {
        "price_spread": price_spread,
        "cheapest_quantized": cheapest_quantized,
        "risk_flag": risk_flag,
        "baseline_context_below_catalog": baseline_context_below_catalog,
    }


def build_steering(
    model_id: str,
    endpoints: list[dict],
    eligible_flags: list[bool],
    default_precision: str | None,
    args,
) -> dict:
    """OpenRouter provider steering object (design section 4.7).

    {"model": model_id, "provider": {"order": [...], "ignore": [...],
    "quantizations": [...], "allow_fallbacks": True}}.

    * order: tags of the baseline-eligible endpoints, cheapest blended first
      (blended price asc, ties broken by (normalized provider name, tag) --
      the same comparator as pick_baseline, stable so the final tie is list
      order). Eligible endpoints whose blended price is unpriceable cannot
      be price-ranked; they are appended after the ranked ones (still
      (provider, tag)-ordered) so every eligible endpoint keeps a place --
      ranked last, never silently left to default priority.
    * ignore: tags of EVERY live endpoint that is not baseline-eligible --
      sub-default quantization, third-party unknown/null, schedule-gate-
      failed AND variants (variants are live but not eligible; the universal
      quantifier includes them). Non-live (status != 0) endpoints are never
      listed. List order (the API's).
    * quantizations: sorted KNOWN quantizations (PRECISION_LADDER labels)
      with rank >= rank(default_precision) among live non-variant endpoints,
      deduplicated -- no schedule or eligibility filter. OMITTED from the
      provider dict entirely when default_precision is None -- never an
      empty list (OpenRouter semantics would filter out every endpoint).
    * allow_fallbacks: True.

    Signature deviation from the plan (controller ruling at Task 5 spawn):
    args is a required parameter -- the order ranking is real pricing math
    (effective_pricing via min_context/input_share), and reusing the
    shape-only _SCHEDULE_GATE_ARGS stand-in for pricing was explicitly
    ruled out in the Task 4 review.

    Raises ValueError when eligible_flags does not match endpoints in length.
    """
    ignore = []
    ranked = []  # (blended, normalized provider name, tag) -- pick_baseline order
    unranked = []  # (normalized provider name, tag) -- eligible but unpriceable
    for endpoint, eligible in _flag_list(endpoints, eligible_flags):
        if endpoint.get("status") == 0 and not eligible:
            ignore.append(endpoint.get("tag") or "")
        if not eligible:
            continue
        provider_name = endpoint.get("provider_name")
        provider_key = (
            normalize_provider(provider_name) if isinstance(provider_name, str) else ""
        )
        tag = endpoint.get("tag") or ""
        price = _blended_price(endpoint, args)
        if price is None:
            unranked.append((provider_key, tag))
        else:
            ranked.append((price, provider_key, tag))
    ranked.sort()  # stable: the final tie is list order, as in pick_baseline
    order = [tag for _, _, tag in ranked] + [tag for _, tag in sorted(unranked)]

    provider = {"order": order, "ignore": ignore}
    if default_precision is not None:
        default_rank = PRECISION_LADDER[default_precision]
        quants = set()
        for endpoint in endpoints:
            if endpoint.get("status") != 0 or is_variant_endpoint(endpoint):
                continue
            quantization = endpoint.get("quantization")
            if (
                not isinstance(quantization, str)
                or quantization not in PRECISION_LADDER
            ):
                continue  # "unknown"/null/unrecognized: not a known quantization
            if PRECISION_LADDER[quantization] >= default_rank:
                quants.add(quantization)
        provider["quantizations"] = sorted(quants)
    provider["allow_fallbacks"] = True
    return {"model": model_id, "provider": provider}


def _stale_filtered_annotations(model_id, catalog_context, args):
    """Annotations for a dropped model's filtered entry, or None.

    Built from the model's last cached endpoint list (load_cache_expired --
    TTL ignored), never from a fresh fetch: the model is out of the ranking,
    so the block is marked "stale": True for display in the filtered section
    only (design section 4.9). Same builders as the live stage: eligibility
    and baseline are re-derived from the cached list, so the shape matches a
    ranked candidate's annotations. A model whose cache holds no endpoint
    list at all yields None (the entry carries the drop reason alone).
    """
    cached = load_cache_expired(f"endpoints:{model_id.replace('/', '_')}")
    if not isinstance(cached, list):
        return None
    author = model_id.partition("/")[0]
    try:
        default_precision = derive_default_precision(cached, author)
        eligible_flags = [
            baseline_eligible(endpoint, default_precision, author, args)
            for endpoint in cached
        ]
        baseline = pick_baseline(cached, eligible_flags, args)
        annotations = build_annotations(
            cached, eligible_flags, baseline, catalog_context, args
        )
    except Exception:
        # Display-only block for an already-dropped model: never fail the run.
        return None
    annotations["stale"] = True
    return annotations


def _endpoint_stage_drop(extra_dropped, filtered, cand, reason, args):
    """Count one endpoint-stage drop and append its filtered entry.

    Same validated entry shape as build_candidates' drop() plus the optional
    stale annotations block (absent cache -> reasons only).
    """
    extra_dropped[reason] = extra_dropped.get(reason, 0) + 1
    entry = {"id": cand["id"], "name": cand["name"] or cand["id"], "reasons": [reason]}
    annotations = _stale_filtered_annotations(cand["id"], cand.get("context"), args)
    if annotations is not None:
        entry["annotations"] = annotations
    filtered.append(entry)


def apply_endpoint_pricing(candidates, args, filtered):
    """Reprice every candidate at its baseline endpoint (design section 4.9).

    The CLI stage between build_candidates and resolve_quality. Endpoint
    fetches run concurrently (ThreadPoolExecutor, 8 workers) through
    fetch_model_endpoints' per-model cache, but the results are applied PER
    CANDIDATE ID in the original candidates order -- never in completion
    order -- so the stage is deterministic.

    Per candidate: endpoints = fetch. On fetch failure after fetch_json's
    internal retries the model is dropped "endpoint_fetch_failed" and removed
    from the returned list. With endpoints, default_precision is derived,
    every endpoint scored for baseline eligibility and the baseline picked;
    no usable baseline drops the model "no_baseline_eligible_endpoint"
    (the stale-annotations filtered-entry rule covers both drop reasons).
    Otherwise the candidate is mutated in place: the pricing block
    (price_in/price_out/blended/base_*/tier/schedule/notes) is replaced with
    the baseline endpoint's effective_pricing outputs scaled per-1M exactly
    like build_candidates, and the additive keys "endpoints", "default_precision",
    "annotations" (catalog context = the candidate's existing "context"),
    "advertised_pricing" (the models-API scalars the candidate carried before
    the mutation, source-tagged) and "steering" are attached.

    A baseline whose prices are missing/non-finite counts as unusable
    (pick_baseline's blended-price skip is authoritative for the top-level
    effective prices). The same bar applies to the base prices the mutation
    would publish: parse_price returns float | None, so a None base price
    (tier-over-None-base endpoint) is reachable -- the isinstance(..., float)
    half of the guard is load-bearing for None, not a redundant type check.
    Returns (surviving candidates in the original order, dropped counts for
    the two new reasons).
    """
    extra_dropped: dict[str, int] = {}
    if not candidates:
        return candidates, extra_dropped
    endpoints_by_id: dict[str, list] = {}
    failed_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for cand in candidates:
            # Duplicate ids fetch once and share the result.
            futures.setdefault(
                cand["id"], pool.submit(fetch_model_endpoints, cand["id"], args)
            )
        for cand_id, future in futures.items():
            try:
                endpoints_by_id[cand_id] = future.result()
            except Exception:
                # fetch_json already retried 3 times; the caller owns the
                # fail-closed policy.
                failed_ids.add(cand_id)
    survivors = []
    for cand in candidates:
        cand_id = cand["id"]
        if cand_id in failed_ids:
            _endpoint_stage_drop(
                extra_dropped, filtered, cand, "endpoint_fetch_failed", args
            )
            continue
        endpoints = endpoints_by_id[cand_id]
        author = cand_id.partition("/")[0]
        default_precision = derive_default_precision(endpoints, author)
        eligible_flags = [
            baseline_eligible(endpoint, default_precision, author, args)
            for endpoint in endpoints
        ]
        baseline = pick_baseline(endpoints, eligible_flags, args)
        eff = (
            effective_pricing(
                baseline.get("pricing"), args.min_context, args.input_share
            )
            if baseline is not None
            else None
        )
        if baseline is None or not all(
            isinstance(eff[key], float) and math.isfinite(eff[key])
            for key in ("price_in", "price_out", "base_price_in", "base_price_out")
        ):
            _endpoint_stage_drop(
                extra_dropped, filtered, cand, "no_baseline_eligible_endpoint", args
            )
            continue
        # The advertised models-API scalars, captured before the mutation.
        advertised_pricing = {
            "input_per_1m": cand["price_in"],
            "output_per_1m": cand["price_out"],
            "blended_per_1m": cand["blended"],
            "source": "models-api",
        }
        # Replace the pricing block with the baseline endpoint's outputs,
        # scaled per-1M exactly like build_candidates.
        cand["price_in"] = eff["price_in"] * 1_000_000.0
        cand["price_out"] = eff["price_out"] * 1_000_000.0
        cand["blended"] = (
            args.input_share * cand["price_in"]
            + (1.0 - args.input_share) * cand["price_out"]
        )
        cand["base_price_in"] = eff["base_price_in"] * 1_000_000.0
        cand["base_price_out"] = eff["base_price_out"] * 1_000_000.0
        cand["tier_prompt_tokens"] = eff["tier_prompt_tokens"]
        cand["tiers"] = eff["tiers"]
        cand["schedule"] = eff["schedule"]
        cand["max_discount"] = eff["max_discount"]
        cand["tier_note"] = fmt_tier_note(eff["tier_prompt_tokens"])
        cand["sched_note"] = fmt_sched_note(eff["max_discount"])
        cand["sched_detail"] = fmt_sched_detail(eff["schedule"], eff["peak_blended"])
        cand["endpoints"] = endpoints
        cand["default_precision"] = default_precision
        cand["annotations"] = build_annotations(
            endpoints, eligible_flags, baseline, cand["context"], args
        )
        cand["advertised_pricing"] = advertised_pricing
        cand["steering"] = build_steering(
            cand_id, endpoints, eligible_flags, default_precision, args
        )
        survivors.append(cand)
    return survivors, extra_dropped


# ---------------------------------------------------------------------------
# OpenRouter session-cost dataset (per-task cost, outside the score)
# ---------------------------------------------------------------------------


SESSION_COST_URL = "https://openrouter.ai/api/v1/datasets/session-cost"
SESSION_COST_HARNESS_SLUG = "hermes-agent"
# The turn range the catalog/CLI surfaces (Task 8).
SESSION_COST_DISPLAY_TURN_RANGE = "10-49-turns"
# The dataset's four turn ranges (brainstorm section 2.6) -- the catalog
# contract's closed set (web/build_site_data.py pins the same tuple).
# Cells naming anything else are dropped: an unknown range is dataset
# drift, and the block must never fail the publish (design section 4.8).
SESSION_COST_TURN_RANGES = ("1-turn", "2-9-turns", "10-49-turns", "50-plus-turns")
SESSION_COST_ATTRIBUTION = "Data © OpenRouter, CC BY 4.0"


def strip_permaslug_date(permaslug: str) -> str:
    """Permaslug without its trailing -YYYYMMDD (undated ids map to themselves)."""
    match = re.match(r"^(.*)-(\d{8})$", permaslug)
    return match.group(1) if match else permaslug


def fetch_session_cost(args) -> dict | None:
    """Per-task session-cost medians from OpenRouter's dataset, or None.

    A separate published dimension (design section 4.8): the median USD
    cost of real sessions through the pinned published harness -- never
    part of the weighted score. One bulk paginated fetch per invocation
    (?app_slug=<harness>&limit=500&offset=k) regardless of the model list;
    pagination stops on a page shorter than the limit or once the next
    offset would exceed the API's 5000 cap. Cells key on dated permaslugs
    (z-ai/glm-5.3-flash-20260826), stripped to the bare catalog id with
    strip_permaslug_date; when several strip to the same id, the latest
    date wins each turn-range cell (same rule as build_aa_benchmarks).

    Returns None (graceful degradation, design section 4.9) when no key is
    configured (the --openrouter-api-key flag, else OPENROUTER_API_KEY),
    when the fetch fails after fetch_json's retries, or when the response
    shape is unexpected -- a malformed payload must never crash the run.
    The merged result is cached under "session-cost" with the standard TTL.
    """
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    if not args.no_cache:
        cached = load_cache("session-cost", args.cache_ttl)
        if isinstance(cached, dict) and isinstance(cached.get("medians"), dict):
            return cached
    cells = []
    # The API caps limit at 500 and offset at 5000 (11 full pages max).
    offset = 0
    while offset <= 5000:
        try:
            payload = fetch_json(
                f"{SESSION_COST_URL}?app_slug={SESSION_COST_HARNESS_SLUG}"
                f"&limit=500&offset={offset}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
                retries=3,
            )
        except Exception as exc:
            warn(f"session-cost dataset request failed ({exc})")
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            warn("session-cost dataset returned an unexpected response shape")
            return None
        cells.extend(cell for cell in data if isinstance(cell, dict))
        if len(data) < 500:
            break
        offset += 500
    medians: dict[str, dict[str, float]] = {}
    seen_date: dict[tuple[str, str], str] = {}

    def accept(bare, date, turn_range, median):
        key = (bare, turn_range)
        if key in seen_date and seen_date[key] >= date:
            return
        seen_date[key] = date
        medians.setdefault(bare, {})[turn_range] = median

    for cell in cells:
        permaslug = cell.get("model_permaslug")
        turn_range = cell.get("turn_range")
        raw = cell.get("median_session_cost_usd")
        if not (
            isinstance(permaslug, str)
            and permaslug
            and isinstance(turn_range, str)
            and turn_range
        ):
            continue
        # Closed contract set: a cell naming a new turn range is dropped,
        # never published (the catalog validator rejects unknown ranges,
        # and session cost must never fail the publish).
        if turn_range not in SESSION_COST_TURN_RANGES:
            continue
        # Non-finite medians would serialize as NaN and poison the catalog.
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        try:
            median = float(raw)  # huge ints raise OverflowError
        except OverflowError:
            continue
        if not math.isfinite(median):
            continue
        match = re.match(r"^(.*)-(\d{8})$", permaslug)
        accept(
            strip_permaslug_date(permaslug),
            match.group(2) if match else "",
            turn_range,
            median,
        )
    if not medians:
        warn("session-cost dataset returned no usable cells")
        return None
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        warn("session-cost dataset returned an unexpected response shape")
        return None
    as_of = meta.get("as_of")
    window_days = meta.get("window_days")
    window_end_date = meta.get("window_end_date")
    # The catalog validator demands non-empty as_of/window_end_date strings
    # and a positive integer window_days; anything else is dataset drift.
    # Degrade to None (block omitted, "not available this refresh") rather
    # than emit a block the validator would reject -- session cost must
    # never fail the publish (design section 4.8).
    if not (
        isinstance(as_of, str)
        and as_of
        and isinstance(window_end_date, str)
        and window_end_date
        and isinstance(window_days, int)
        and not isinstance(window_days, bool)
        and window_days > 0
    ):
        warn("session-cost dataset returned malformed meta; omitting the block")
        return None
    result = {
        "attribution": SESSION_COST_ATTRIBUTION,
        "app_slug": SESSION_COST_HARNESS_SLUG,
        "as_of": as_of,
        "window_days": window_days,
        "window_end_date": window_end_date,
        "medians": medians,
    }
    save_cache("session-cost", result)
    return result


# ---------------------------------------------------------------------------
# OpenRouter catalog
# ---------------------------------------------------------------------------


def fetch_openrouter_models(args):
    if not args.no_cache:
        cached = load_cache("openrouter-models", args.cache_ttl)
        if cached:
            return cached, True
    payload = fetch_json(OPENROUTER_MODELS_URL, timeout=30)
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("OpenRouter API returned no models")
    save_cache("openrouter-models", models)
    return models, False


FRONTEND_DISCOUNTS_CACHE = "openrouter-frontend-discounts"
FRONTEND_AA_CACHE = "openrouter-frontend-aa"


def fetch_openrouter_frontend(args):
    """Derive discounts, ZDR ids and OR-published AA benchmarks.

    Two loads: the base frontend URL serves discounts and benchmarks in one
    response; the per-endpoint ZDR list serves ZDR and is skipped entirely
    under --no-zdr. Each payload caches independently under its own key and is
    cached only when non-empty, so a degraded payload self-heals on the
    next run. A base-URL outage never blocks the ZDR fetch, and vice
    versa. Returns (discounts, zdr_ids, aa_by_id, cache_hits) where
    cache_hits names the payloads served from cache ("discounts", "zdr",
    "aa").
    """
    discounts = {}
    aa_by_id = {}
    zdr_ids = set()
    cache_hits = set()

    aa_from_cache = False
    if not args.no_cache:
        cached = load_cache(FRONTEND_DISCOUNTS_CACHE, args.cache_ttl)
        if cached:
            discounts = cached
            cache_hits.add("discounts")
        cached = load_cache(FRONTEND_AA_CACHE, args.cache_ttl)
        if cached:
            aa_by_id = cached
            aa_from_cache = True
            cache_hits.add("aa")

    if not discounts or not aa_from_cache:
        try:
            payload = fetch_json(OPENROUTER_DISCOUNTS_URL, timeout=30)
        except Exception as exc:
            if not discounts:
                warn(f"could not fetch discount data: {exc}")
            if not aa_from_cache:
                warn(f"could not fetch AA benchmark data: {exc}")
        else:
            data = payload.get("data") if isinstance(payload, dict) else None
            entries = data.get("models") if isinstance(data, dict) else None
            fresh_discounts = {}
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                slug = entry.get("slug") or ""
                if not slug or slug.startswith("~"):
                    continue
                endpoint = entry.get("endpoint") or {}
                raw = (endpoint.get("pricing") or {}).get("discount")
                if not isinstance(raw, (int, float)):
                    continue
                variant = endpoint.get("variant") or ""
                key = slug if variant in ("", "standard") else f"{slug}:{variant}"
                fresh_discounts.setdefault(key, float(raw))
            fresh_aa = build_aa_benchmarks(
                data.get("benchmarks") if isinstance(data, dict) else None
            )
            if fresh_discounts:
                discounts = fresh_discounts
                save_cache(FRONTEND_DISCOUNTS_CACHE, discounts)
                cache_hits.discard("discounts")
            elif not discounts:
                # An intact endpoint always yields hundreds of entries
                # (discount: 0 is still an entry); an empty map means the
                # response shape changed -- never cache that, so the next
                # run recovers on its own.
                warn("no discount entries found; treating discounts as unavailable")
            if not aa_from_cache:
                if fresh_aa:
                    aa_by_id = fresh_aa
                    save_cache(FRONTEND_AA_CACHE, aa_by_id)
                else:
                    warn(
                        "no AA benchmark entries found; treating OpenRouter "
                        "benchmarks as unavailable"
                    )

    if args.no_zdr:
        return discounts, zdr_ids, aa_by_id, cache_hits
    if not args.no_cache:
        # v3: the id set now comes from the per-endpoint ZDR list; v1/v2
        # sets were derived from the models/find response and must never
        # be read.
        cached = load_cache("openrouter-zdr-v3", args.cache_ttl)
        if cached:
            return discounts, set(cached), aa_by_id, cache_hits | {"zdr"}
    try:
        payload = fetch_json(OPENROUTER_ZDR_URL, timeout=30)
    except Exception as exc:
        warn(f"could not fetch ZDR data: {exc}")
        return discounts, zdr_ids, aa_by_id, cache_hits
    # One entry per ZDR endpoint, keyed by the exact model id (variants
    # such as :free are listed individually). An id is ZDR iff it has at
    # least one ZDR endpoint. The models/find ?zdr=true filter cannot be
    # used: it is model-level, so it lists non-ZDR variants (NVIDIA's
    # :free endpoints) too. No "~" router aliases occur in this list, so
    # unlike the find parsing there is no "~" skip; any stray id is
    # harmless because build_candidates only matches real model ids.
    # An endpoint counts only when its status is exactly the integer 0.
    # OpenRouter's OpenAPI schema (EndpointStatus) lists 0, -1, -2, -3, -5
    # and -10 without describing them, so the meaning of the non-zero values
    # is unverified; treat anything other than 0, including a missing
    # status, as not serving (fail closed).
    entries = payload.get("data") if isinstance(payload, dict) else None
    ids = set()
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if type(status) is not int or status != 0:
            continue
        model_id = entry.get("model_id")
        if isinstance(model_id, str) and model_id:
            ids.add(model_id)
    if not ids:
        warn("no ZDR entries found; treating ZDR data as unavailable")
        return discounts, zdr_ids, aa_by_id, cache_hits
    save_cache("openrouter-zdr-v3", sorted(ids))
    return discounts, ids, aa_by_id, cache_hits


def parse_price(value) -> float | None:
    # bools are not prices; a huge JSON int (json.loads yields an exact
    # Python int) overflows float() -- both fail soft to None.
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# OpenRouter tiered pricing (pricing.overrides)
# ---------------------------------------------------------------------------

# Keys of a pricing.overrides entry (OpenRouter docs, verified 2026-10-09).
# An entry carrying any other key is skipped whole (docs forward-compat rule).
TIER_CONDITION_KEYS = (
    "min_prompt_tokens",
    "utc_start",
    "utc_end",
    "utc_days",
)
TIER_PRICE_KEYS = (
    "prompt",
    "completion",
    "request",
    "image",
    "image_output",
    "audio",
    "audio_output",
    "web_search",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
    "input_cache_write_1h",
    "input_audio_cache",
)
WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_WEEK_MINUTES = 7 * 24 * 60


def _integral(value) -> int | None:
    """Finite, non-bool, integral number -> int; else None (fail-soft)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # Magnitude guard first: math.isfinite raises OverflowError on huge ints
    # (json.loads of a 400-digit literal); no real token count or clock
    # comes anywhere near 1e15.
    if abs(value) > 10**15:
        return None
    if not math.isfinite(value) or value != int(value):
        return None
    return int(value)


def _hhmm(value) -> int | None:
    """Integer HHMM clock in 0..2359 with minutes < 60; else None."""
    clock = _integral(value)
    if clock is None or not 0 <= clock <= 2359 or clock % 100 > 59:
        return None
    return clock


def _hhmm_minutes(clock: int) -> int:
    return (clock // 100) * 60 + clock % 100


def _window_minutes(start: int, end: int) -> int:
    """Minutes a HHMM window covers per day (wrap-aware; whole day when equal)."""
    span = _hhmm_minutes(end) - _hhmm_minutes(start)
    if span == 0:
        return 24 * 60
    return span % (24 * 60)


def _window_span(start: int, end: int):
    """The per-day minute values a window matches (wrap-aware)."""
    start_min = _hhmm_minutes(start)
    end_min = _hhmm_minutes(end)
    if start_min == end_min:
        return range(24 * 60)
    if start_min < end_min:
        return range(start_min, end_min)
    return list(range(start_min, 24 * 60)) + list(range(0, end_min))


def _coverage(days_count: int, minutes: int) -> float:
    """Fraction of the 7-day week a window covers, 4 decimals."""
    return round(days_count * minutes / _WEEK_MINUTES, 4)


def _entry_prices(entry):
    """Parse an override entry's prompt/completion prices (spec rule 5).

    Returns {"prompt": float|None, "completion": float|None} where None means
    the key is absent (inheritance), or None when a present price is
    malformed, negative or non-finite (the entry is skipped whole).
    """
    values = {}
    for key in ("prompt", "completion"):
        if key not in entry:
            values[key] = None
            continue
        value = parse_price(entry[key])
        if value is None or not math.isfinite(value) or value < 0:
            return None
        values[key] = value
    return values


def _schedule_windows(window_entries):
    """Normalize validated time-window entries (spec rules 6, 9, 10).

    Returns the window list (API order, days-only windows normalized to 0/0,
    utc_days deduplicated in week order, per-window coverage), or None when
    the windows do not tile the 7-day week exactly (gaps or overlaps), or a
    window is missing prompt/completion (rule 7: no top-level inheritance --
    the top level is fetch-time dependent).
    """
    windows = []
    covered = set()
    matched = 0
    for entry in window_entries:
        start, end = entry["start"], entry["end"]
        days = entry["days"]  # None, or a list of weekday indices
        if start is None:  # days-only window: the whole UTC day
            start, end = 0, 0
        price_in, price_out = entry["price_in"], entry["price_out"]
        if price_in is None or price_out is None:
            return None
        day_indices = list(range(len(WEEKDAYS))) if days is None else sorted(set(days))
        span = _window_span(start, end)
        for day in day_indices:
            for minute in span:
                covered.add((day, minute))
            matched += len(span)
        windows.append(
            {
                "utc_days": None
                if days is None
                else [WEEKDAYS[i] for i in day_indices],
                "utc_start": start,
                "utc_end": end,
                "coverage": _coverage(len(day_indices), len(span)),
                "price_in": price_in,
                "price_out": price_out,
            }
        )
    if matched != _WEEK_MINUTES or len(covered) != _WEEK_MINUTES:
        return None  # gaps or overlaps: the schedule does not tile the week
    return windows


def effective_pricing(pricing, prompt_tokens, input_share):
    """Resolve pricing.overrides into one effective price (spec section 5.1).

    Prices are USD per token. Returns a dict with the effective and base
    prices (per token), the applied tier threshold, the cumulative tier list,
    the normalized schedule (or None), whether the schedule is invalid, the
    rounded maximum off-peak discount, and the peak window's blended price.
    """
    source = pricing if isinstance(pricing, dict) else {}
    top_in = parse_price(source.get("prompt"))
    top_out = parse_price(source.get("completion"))
    base_in, base_out = top_in, top_out
    overrides = source.get("overrides")

    tier_entries = []  # (threshold, price_in|None, price_out|None), API order
    window_entries = []  # validated time-window pre-windows, API order
    time_touched = False
    schedule_broken = False

    if isinstance(overrides, list):
        for entry in overrides:
            if not isinstance(entry, dict):
                continue
            time_keys = [
                key for key in ("utc_start", "utc_end", "utc_days") if key in entry
            ]
            if any(
                key not in TIER_CONDITION_KEYS and key not in TIER_PRICE_KEYS
                for key in entry
            ):
                if time_keys:
                    schedule_broken = True  # skipped time window: tiling gap
                continue
            has_threshold = "min_prompt_tokens" in entry
            threshold = _integral(entry["min_prompt_tokens"]) if has_threshold else None
            if has_threshold and (threshold is None or threshold < 0):
                if time_keys:
                    schedule_broken = True
                continue
            if ("utc_start" in entry) != ("utc_end" in entry):
                schedule_broken = True  # exactly one of the pair
                continue
            start = end = None
            if "utc_start" in entry:
                start = _hhmm(entry["utc_start"])
                end = _hhmm(entry["utc_end"])
                if start is None or end is None:
                    schedule_broken = True
                    continue
            days = None
            if "utc_days" in entry:
                names = entry["utc_days"]
                if (
                    not isinstance(names, list)
                    or not names
                    or any(day not in WEEKDAYS for day in names)
                ):
                    schedule_broken = True
                    continue
                days = sorted({WEEKDAYS.index(day) for day in names})
            prices = _entry_prices(entry)
            if prices is None:
                if time_keys:
                    schedule_broken = True
                continue
            if time_keys:
                time_touched = True
                if has_threshold:
                    # combined token+time entry: invalid time window (rule 9)
                    schedule_broken = True
                    continue
                window_entries.append(
                    {
                        "start": start,
                        "end": end,
                        "days": days,
                        "price_in": prices["prompt"],
                        "price_out": prices["completion"],
                    }
                )
            elif has_threshold:
                tier_entries.append((threshold, prices["prompt"], prices["completion"]))
            # else: conditionless entry (price keys only) -- ignored (plan
            # Review Focus; none live, and honoring an unconditional price
            # swap would be dangerous)

    schedule = None
    peak_blended = None
    max_discount = None
    schedule_error = False

    if time_touched or schedule_broken:
        if schedule_broken:
            schedule_error = True
        else:
            schedule = _schedule_windows(window_entries)
            if schedule is None:
                schedule_error = True
                schedule = None
    if schedule is not None and not schedule_error:
        for window in schedule:
            window["blended"] = (
                input_share * window["price_in"]
                + (1 - input_share) * window["price_out"]
            )
        peak = max(schedule, key=lambda window: window["blended"])
        peak_blended = peak["blended"]  # ties: first in API order
        # Deterministic base (rule 7): freeze at the peak window's prices.
        base_in, base_out = peak["price_in"], peak["price_out"]
        if peak_blended == 0:
            max_discount = 0.0
        else:
            max_discount = round(
                max(1 - window["blended"] / peak_blended for window in schedule), 4
            )

    # Token tiers (rule 8) over the (possibly frozen) base: later entries win
    # per key in array order; a threshold equal to prompt_tokens never applies.
    applied = None
    current_in, current_out = base_in, base_out
    by_threshold = {}
    for threshold, tier_in, tier_out in tier_entries:
        if prompt_tokens > threshold:
            if tier_in is not None:
                current_in = tier_in
            if tier_out is not None:
                current_out = tier_out
            applied = threshold
        slot = by_threshold.setdefault(threshold, {"price_in": None, "price_out": None})
        if tier_in is not None:
            slot["price_in"] = tier_in
        if tier_out is not None:
            slot["price_out"] = tier_out
    tiers = []
    running_in, running_out = base_in, base_out
    for threshold in sorted(by_threshold):
        slot = by_threshold[threshold]
        if slot["price_in"] is not None:
            running_in = slot["price_in"]
        if slot["price_out"] is not None:
            running_out = slot["price_out"]
        tiers.append(
            {
                "min_prompt_tokens": threshold,
                "price_in": running_in,
                "price_out": running_out,
            }
        )

    return {
        "price_in": current_in,
        "price_out": current_out,
        # top_*: the RAW OpenRouter top-level prices (spec rule 1) -- the
        # validity gate checks these even for windowed models, whose base_*
        # fields are the frozen peak window (rule 7).
        "top_price_in": top_in,
        "top_price_out": top_out,
        "base_price_in": base_in,
        "base_price_out": base_out,
        "tier_prompt_tokens": applied,
        "tiers": tiers,
        "schedule": schedule,
        "schedule_error": schedule_error,
        "max_discount": max_discount,
        "peak_blended": peak_blended,
    }


def fmt_tier_note(threshold):
    """>100k-style marker for the applied token tier; None passes through."""
    if threshold is None:
        return None
    value = int(threshold)
    if value >= 1000:
        return f">{value // 1000}k"
    return f">{value}"


def fmt_sched_note(max_discount):
    """-38%-style marker for a schedule's max off-peak discount."""
    if max_discount is None:
        return None
    formatted = f"{max_discount:.0%}"
    if formatted == "0%":
        return "SCHED"
    return f"-{formatted}"


def fmt_sched_detail(schedule, peak_blended):
    """Human-readable off-peak schedule summary (spec section 5.6 grammar).

    Off-peak windows are those whose blended price is strictly below the
    peak's; schedule windows carry their parser-computed "blended" price.
    """
    if not schedule or peak_blended is None:
        return None
    offpeak = [window for window in schedule if window["blended"] < peak_blended]
    if not offpeak:
        return None

    def _clock(clock):
        return f"{clock // 100:02d}:{clock % 100:02d}"

    groups = {}
    for window in offpeak:
        days = window["utc_days"]
        # All seven weekdays listed explicitly is the same day set as
        # utc_days absent: same group, same "daily" label (spec 5.6).
        if days is None or set(days) == set(WEEKDAYS):
            key = (0, -1, ())
            label = "daily"
        else:
            indices = [WEEKDAYS.index(day) for day in days]
            key = (1, min(indices), tuple(days))
            exact = {WEEKDAYS[i] for i in range(5)}
            if set(days) == exact:
                label = "weekdays"
            elif set(days) == {"saturday", "sunday"}:
                label = "weekends"
            else:
                label = ",".join(day[:3] for day in days)  # week order
        group = groups.setdefault(key, {"label": label, "windows": []})
        group["windows"].append(window)
    parts = []
    for key in sorted(groups):
        group = groups[key]
        bits = []
        for window in sorted(group["windows"], key=lambda w: w["utc_start"]):
            if window["utc_start"] == window["utc_end"]:
                bits.append("all day")
            else:
                bits.append(
                    f"{_clock(window['utc_start'])}-{_clock(window['utc_end'])}"
                )
        parts.append(f"{group['label']} {', '.join(bits)}")
    return "; ".join(parts) + " UTC"


def coerce_int(value, default: int = 0) -> int:
    """Best-effort int coercion for external fields that may be str/float/None."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# Artificial Analysis intelligence index
# ---------------------------------------------------------------------------


def aa_api_entries(api_key: str) -> list:
    """Intelligence entries from AA's V2 language models endpoint.

    Documented envelope: {tier, intelligence_index_version, pagination,
    data[]} with snake_case fields; the index lives at
    data[].evaluations.artificial_analysis_intelligence_index. Items
    without a usable slug/name or without a finite numeric index are
    skipped (nulls mean "not measured"). Pagination follows the documented
    has_more/total_pages contract with a 25-page hard cap; a failure past
    page 1 is best-effort (warn, keep earlier pages), a page-1 failure
    propagates.
    """
    entries = {}
    page = 1
    while page <= 25:  # hard cap: halt after processing page 25
        try:
            payload = fetch_json(
                f"{AA_API_URL}?page={page}",
                headers={"x-api-key": api_key},
                timeout=30,
            )
        except Exception as exc:
            if page == 1:
                raise
            warn(f"AA API page {page} failed ({exc}); using entries collected so far")
            break
        data = payload.get("data") if isinstance(payload, dict) else None
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            name = item.get("name")
            key = (
                slug
                if isinstance(slug, str) and slug
                else (name if isinstance(name, str) and name else None)
            )
            if not key:
                continue
            evals = item.get("evaluations")
            raw = (
                evals.get("artificial_analysis_intelligence_index")
                if isinstance(evals, dict)
                else None
            )
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                continue
            try:
                index = float(raw)  # huge ints raise OverflowError
            except OverflowError:
                continue
            if not math.isfinite(index):
                continue
            entries.setdefault(
                key,
                {
                    "key": key,
                    "name": name if isinstance(name, str) and name else key,
                    "index": index,
                },
            )
        pag = payload.get("pagination") if isinstance(payload, dict) else None
        if not isinstance(pag, dict):
            break  # undocumented shape: single-page degrade
        if pag.get("has_more") is not True:
            break
        total_pages = pag.get("total_pages")
        if isinstance(total_pages, int) and page >= total_pages:
            break
        page += 1
    return list(entries.values())


def fetch_aa_entries(args):
    """AA intelligence entries from the keyed AA API v2, or nothing.

    Returns (entries, source, cached). With a key, an API failure or an
    empty result warns and yields ([], None, False). Without a key this is
    silent and yields ([], None, False): OpenRouter's benchmarks are the
    primary source and the catalog records the absence (sources.aa).
    """
    if not args.no_cache:
        # v2: entries cached under the old key predate the V2 endpoint
        # migration and must never be read. Only AA API entries are read;
        # build_catalog rejects any other source as unknown.
        cached = load_cache("aa-intelligence-v2", args.cache_ttl)
        if isinstance(cached, dict) and cached.get("source") == "AA API v2":
            return cached.get("entries", []), cached.get("source"), True
    api_key = args.aa_api_key or os.environ.get("AA_API_KEY")
    if not api_key:
        return [], None, False
    try:
        entries = aa_api_entries(api_key)
    except Exception as exc:
        warn(f"AA API request failed ({exc})")
        return [], None, False
    if not entries:
        warn("AA API returned no intelligence scores")
        return [], None, False
    save_cache("aa-intelligence-v2", {"entries": entries, "source": "AA API v2"})
    return entries, "AA API v2", False


# ---------------------------------------------------------------------------
# Matching AA entries to OpenRouter model ids
# ---------------------------------------------------------------------------


def build_aa_lookup(entries):
    exact = {}
    fuzzy = []
    for entry in entries:
        key = entry.get("key") or ""
        name = entry.get("name") or key
        raw_index = entry.get("index")
        if not isinstance(raw_index, (int, float)) or isinstance(raw_index, bool):
            continue
        index = float(raw_index)
        # Non-finite indexes would serialize as NaN and poison quality.
        if not math.isfinite(index):
            continue

        def put(tier, raw):
            normalized = norm_key(raw)
            if normalized:
                exact.setdefault((tier, normalized), index)

        if "/" in key:
            put(0, key)
            put(1, key.rsplit("/", 1)[-1])
        else:
            put(1, key)
        put(2, name)
        put(3, PAREN_RE.sub(" ", name))
        tokens = set(norm_key(key).split()) | set(norm_key(name).split())
        if tokens:
            fuzzy.append((tokens, index))
    return exact, fuzzy


def match_quality(model, exact, fuzzy, allow_fuzzy=False):
    model_id = model["id"]
    name = model.get("name") or ""
    display = PROVIDER_PREFIX_RE.sub("", name).strip()
    # Assumes model_id contains a "/" (provider/base). build_candidates drops
    # malformed ids upstream, so base is never empty here.
    _, _, base = model_id.split(":", 1)[0].partition("/")

    for tier, raw in (
        (0, model_id),
        (1, base),
        (2, display),
        (3, PAREN_RE.sub(" ", display)),
    ):
        normalized = norm_key(raw)
        if normalized and (tier, normalized) in exact:
            return exact[(tier, normalized)]

    if not allow_fuzzy:
        return None
    own = set(norm_key(base).split()) | set(norm_key(display).split())
    if own:
        best = 0.0
        best_index = None
        for tokens, index in fuzzy:
            overlap = len(own & tokens) / len(own | tokens)
            if overlap > best:
                best = overlap
                best_index = index
        if best >= 0.5 and best_index is not None:
            return best_index
    return None


def base_model_id(model_id: str) -> str:
    """Candidate id without its :variant suffix (OR benchmarks are per base model)."""
    return model_id.split(":", 1)[0]


AA_BENCHMARK_FIELDS = ("intelligence_index", "coding_index", "agentic_index")


def build_aa_benchmarks(benchmarks) -> dict:
    """Map bare OpenRouter id -> OR-published AA trio from data.benchmarks.

    Keys are dated permaslugs (z-ai/glm-5.3-flash-20260826): the trailing
    -YYYYMMDD is stripped and undated keys map to themselves; when two keys
    strip to the same id the latest date wins. Values failing the
    finite-number guard are treated as absent; entries with at least one
    valid field are kept.
    """
    aa_by_id = {}
    seen_date = {}

    def accept(bare, date, trio):
        if bare in seen_date and seen_date[bare] >= date:
            return
        seen_date[bare] = date
        aa_by_id[bare] = trio

    for key, node in (benchmarks or {}).items():
        aa = node.get("aa") if isinstance(node, dict) else None
        if not isinstance(aa, dict):
            continue
        trio = {}
        for field in AA_BENCHMARK_FIELDS:
            value = aa.get(field)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                trio[field] = float(value)
        if not trio:
            continue
        match = re.match(r"^(.*)-(\d{8})$", key)
        if match:
            accept(match.group(1), match.group(2), trio)
        else:
            accept(key, "", trio)
    return aa_by_id


def resolve_quality(candidates, aa_by_id, exact, fuzzy, aa_source):
    """Per-candidate quality: OR benchmarks first, AA exact fallback, else None.

    OR's per-slug values cannot mispair; the AA fallback is exact-tier only
    (match_quality's fuzzy tier is the proven variant-conflation bug and
    stays off in production). Source values match the catalog contract:
    "openrouter", "api".
    """
    fallback_source = {"AA API v2": "api"}.get(aa_source)
    quality_by_id = {}
    source_by_id = {}
    for cand in candidates:
        bench = aa_by_id.get(base_model_id(cand["id"]))
        intelligence = bench.get("intelligence_index") if bench else None
        if intelligence is not None:
            quality_by_id[cand["id"]] = intelligence
            source_by_id[cand["id"]] = "openrouter"
            continue
        index = match_quality({"id": cand["id"], "name": cand["name"]}, exact, fuzzy)
        if index is not None:
            quality_by_id[cand["id"]] = index
            source_by_id[cand["id"]] = fallback_source
    return quality_by_id, source_by_id


def model_family(model_id: str) -> str | None:
    """Best-effort model family from the base slug (documented heuristic).

    The leading token delimited by dash, underscore or digit of the
    lowercased base slug: glm-5.3 -> glm, gpt-5.2-mini -> gpt,
    deepseek-chat-v4 -> deepseek. Oddballs yield oddballs (o4-mini -> "o");
    a bare letter-run followed by a trailing digit version (k2) yields None.
    """
    base = model_id.split(":", 1)[0].partition("/")[2].lower()
    parts = re.split(r"[-_\d]", base, maxsplit=1)
    token = parts[0]
    if len(parts) > 1 and not parts[1] and base[len(token) : len(token) + 1].isdigit():
        return None
    return token or None


def _per_1m(price_per_token: float) -> float:
    """USD per token -> USD per 1M tokens, 6 decimals."""
    return round(price_per_token * 1_000_000.0, 6)


def build_schedules(models, args, quality_by_id, aa_by_id, balanced_by_id):
    """Catalog-wide off-peak index built from the RAW model list (spec 5.4).

    Deliberately unfiltered by the candidate pool: sub-floor-context and
    non-ZDR schedule models are listed (that is the point -- today's live
    schedule models mostly sit below the 1M context floor). Router aliases
    (~ids) and malformed ids are excluded; :batch variants are included as
    distinct deals. Sorted by id.
    """
    entries = []
    for model in models:
        model_id = model.get("id") or ""
        if not model_id or model_id.startswith("~") or "/" not in model_id:
            continue
        eff = effective_pricing(
            model.get("pricing") or {}, args.min_context, args.input_share
        )
        top_in, top_out = eff["top_price_in"], eff["top_price_out"]
        if (
            top_in is None
            or top_out is None
            or top_in < 0
            or top_out < 0
            or not math.isfinite(top_in)
            or not math.isfinite(top_out)
        ):
            continue  # invalid top level: not a deal worth listing
        windows = eff["schedule"]
        if eff["schedule_error"] or not windows:
            continue
        peak = max(windows, key=lambda window: window["blended"])
        offpeak = min(windows, key=lambda window: window["blended"])
        quality = quality_by_id.get(model_id)
        if quality is None:
            benchmark = aa_by_id.get(base_model_id(model_id)) or {}
            quality = benchmark.get("intelligence_index")
        name = model.get("name") or model_id
        entries.append(
            {
                "id": model_id,
                "name": PROVIDER_PREFIX_RE.sub("", name).strip() or name,
                "context": coerce_int(model.get("context_length"), 0),
                "quality": quality,
                "score": balanced_by_id.get(model_id),
                "max_discount": eff["max_discount"],
                "sched_note": fmt_sched_note(eff["max_discount"]),
                "sched_detail": fmt_sched_detail(windows, eff["peak_blended"]),
                "peak": {
                    "input_per_1m": _per_1m(peak["price_in"]),
                    "output_per_1m": _per_1m(peak["price_out"]),
                    "blended_per_1m": _per_1m(peak["blended"]),
                },
                "offpeak": {
                    "input_per_1m": _per_1m(offpeak["price_in"]),
                    "output_per_1m": _per_1m(offpeak["price_out"]),
                    "blended_per_1m": _per_1m(offpeak["blended"]),
                },
                "schedule": [
                    {
                        "utc_days": window["utc_days"],
                        "utc_start": window["utc_start"],
                        "utc_end": window["utc_end"],
                        "coverage": window["coverage"],
                        "input_per_1m": _per_1m(window["price_in"]),
                        "output_per_1m": _per_1m(window["price_out"]),
                        "blended_per_1m": _per_1m(window["blended"]),
                    }
                    for window in windows
                ],
            }
        )
    entries.sort(key=lambda entry: entry["id"])
    return entries


def catalog_weights(candidates, quality_by_id):
    """Effective per-priority weights for the catalog document.

    Mirrors compute_scores' quality-blind rule: when no candidate has a
    quality score the quality weight is dropped and the rest renormalize,
    so scores.overall in the catalog is exactly reproducible downstream.
    """
    quality_blind = not any(c["id"] in quality_by_id for c in candidates)
    weights = {}
    for priority, base in PRIORITY_WEIGHTS.items():
        effective = dict(base)
        if quality_blind and "quality" in effective:
            effective.pop("quality")
            total = sum(effective.values())
            effective = {name: value / total for name, value in effective.items()}
        weights[priority] = effective
    return weights


# ---------------------------------------------------------------------------
# Filtering and scoring
# ---------------------------------------------------------------------------


def build_candidates(models, args, discounts, zdr_ids, filtered_out=None):
    now = time.time()
    require_tools = not args.no_require_tools
    dropped = {}
    candidates = []

    def drop(reason, model_id, name):
        dropped[reason] = dropped.get(reason, 0) + 1
        if filtered_out is None:
            return
        # Empty/malformed ids would fail the site validator's filtered-id
        # rule, so they stay counted in dropped only.
        if not model_id or "/" not in model_id:
            return
        filtered_out.append(
            {"id": model_id, "name": name or model_id, "reasons": [reason]}
        )

    for model in models:
        model_id = model.get("id") or ""
        if "/" not in model_id:
            drop("malformed id", model_id, model.get("name"))
            continue
        context = coerce_int(model.get("context_length"), 0)
        if context < args.min_context:
            drop("context", model_id, model.get("name"))
            continue
        pricing = model.get("pricing") or {}
        eff = effective_pricing(pricing, args.min_context, args.input_share)
        top_in, top_out = eff["top_price_in"], eff["top_price_out"]
        # The RAW top-level prices carry the validity drop (spec rules 1+7):
        # windowed models freeze their base at the peak window, so the gate
        # must check the top level itself. A bad override must never drop a
        # model, and effective prices inherit validity from the validated
        # base and override prices.
        if (
            top_in is None
            or top_out is None
            or top_in < 0
            or top_out < 0
            or not math.isfinite(top_in)
            or not math.isfinite(top_out)
        ):
            drop("pricing", model_id, model.get("name"))
            continue
        if eff["schedule_error"]:
            # No valid peak window means no deterministic base (spec rule 9):
            # schedule-only AND mixed models fail closed here.
            drop("schedule", model_id, model.get("name"))
            continue
        price_in, price_out = eff["price_in"], eff["price_out"]
        if args.exclude_free and price_in == 0 and price_out == 0:
            drop("free", model_id, model.get("name"))
            continue
        if not args.include_batch and model_id.endswith(":batch"):
            drop("batch", model_id, model.get("name"))
            continue
        discount = (discounts or {}).get(model_id)
        if args.discount and not has_discount(discount):
            drop("no discount", model_id, model.get("name"))
            continue
        if not args.no_zdr and model_id not in (zdr_ids or set()):
            drop("not ZDR", model_id, model.get("name"))
            continue
        modality = (model.get("architecture") or {}).get("modality") or ""
        output_modality = (
            modality.split("->")[-1].strip() if "->" in modality else "text"
        )
        if output_modality != "text":
            drop("modality", model_id, model.get("name"))
            continue
        params = model.get("supported_parameters") or []
        tool_calling = "tools" in params and "tool_choice" in params
        if require_tools and not tool_calling:
            drop("tool calling", model_id, model.get("name"))
            continue
        expiry = parse_iso_datetime(model.get("expiration_date"))
        expired = bool(expiry and expiry < datetime.now(timezone.utc))
        if expired:
            drop("expired", model_id, model.get("name"))
            continue
        created = model.get("created") or 0
        try:
            created = float(created)
        except (TypeError, ValueError):
            created = 0.0
        age_days = max(0.0, (now - created) / 86400.0) if created > 0 else None
        if args.max_age_days and age_days is not None and age_days > args.max_age_days:
            drop("age", model_id, model.get("name"))
            continue
        price_in_m = price_in * 1_000_000.0
        price_out_m = price_out * 1_000_000.0
        blended_m = (
            args.input_share * price_in_m + (1.0 - args.input_share) * price_out_m
        )
        candidates.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "context": context,
                "price_in": price_in_m,
                "price_out": price_out_m,
                "blended": blended_m,
                "base_price_in": eff["base_price_in"] * 1_000_000.0,
                "base_price_out": eff["base_price_out"] * 1_000_000.0,
                "tier_prompt_tokens": eff["tier_prompt_tokens"],
                "tiers": eff["tiers"],
                "schedule": eff["schedule"],
                "max_discount": eff["max_discount"],
                "tier_note": fmt_tier_note(eff["tier_prompt_tokens"]),
                "sched_note": fmt_sched_note(eff["max_discount"]),
                "sched_detail": fmt_sched_detail(eff["schedule"], eff["peak_blended"]),
                "age_days": age_days,
                "discount": discount,
                "created": created,
                "tool_calling": tool_calling,
                "zdr": None if args.no_zdr else model_id in (zdr_ids or set()),
                "expired": expired,
            }
        )

    return candidates, dropped


def compute_scores(candidates, args, quality_by_id):
    weights = dict(PRIORITY_WEIGHTS[args.priority])
    # Deliberate asymmetry: the quality weight is dropped (and the remaining
    # weights renormalized) only when *no* candidate has a quality score, so a
    # quality-blind pool is ranked purely on price/context/age. When *some*
    # candidates match, the quality weight is kept and unmatched candidates take
    # quality_score = 0 -- they are penalized rather than silently reweighted.
    if not any(c["id"] in quality_by_id for c in candidates):
        weights.pop("quality")
        total = sum(weights.values())
        weights = {name: value / total for name, value in weights.items()}

    prices = [c["blended"] for c in candidates]
    low = math.log10(min(prices) + 0.01)
    high = math.log10(max(prices) + 0.01)
    price_span = (high - low) or 1.0
    floor_ctx = max(args.min_context, 1000)
    cap_ctx = 4.0 * floor_ctx
    ctx_span = math.log(cap_ctx / floor_ctx) or 1.0

    for cand in candidates:
        price = cand["blended"]
        if price <= 0:
            cand["price_score"] = 1.0
        else:
            cand["price_score"] = max(
                0.0, min(1.0, 1.0 - (math.log10(price + 0.01) - low) / price_span)
            )
        quality = quality_by_id.get(cand["id"])
        cand["quality"] = quality
        if quality is None:
            cand["quality_score"] = 0.0
        else:
            cand["quality_score"] = max(0.0, min(1.0, quality / args.quality_ref))
        context = cand["context"]
        cand["context_score"] = (
            max(0.0, min(1.0, math.log(context / floor_ctx) / ctx_span))
            if context > floor_ctx
            else 0.0
        )
        age_days = cand["age_days"]
        cand["age_score"] = (
            0.5 if age_days is None else 0.5 ** (age_days / args.recency_half_life)
        )
        cand["score"] = weighted_score(cand, weights)

    candidates.sort(key=lambda c: ranking_key(c["score"], c))
    return weights


def weighted_score(cand, w):
    """The one unrounded weighted score, shared by compute_scores and
    catalog_rankings. Keep the operand order (quality, price, context, age):
    float addition is non-associative, and a reordered sum can differ by one
    ULP and flip a near-tie between the CLI and the catalog."""
    return (
        w.get("quality", 0.0) * cand["quality_score"]
        + w.get("price", 0.0) * cand["price_score"]
        + w.get("context", 0.0) * cand["context_score"]
        + w.get("age", 0.0) * cand["age_score"]
    )


def ranking_key(score, cand):
    """The one sort key: score desc, then raw quality desc, cheaper first, id."""
    return (-score, -(cand["quality"] or 0.0), cand["blended"], cand["id"])


def catalog_rankings(candidates, weights):
    """Per-priority full ordered id lists: the order a `--priority P` run
    produces. Pure: sorted() only, never writes onto the candidate dicts."""
    return {
        priority: [
            c["id"]
            for c in sorted(
                candidates,
                key=lambda c, w=weights[priority]: ranking_key(weighted_score(c, w), c),
            )
        ]
        for priority in PRIORITY_WEIGHTS
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def fmt_price(value: float) -> str:
    if value == 0:
        return "0"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_context(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


def fmt_age(age_days) -> str:
    if age_days is None:
        return "-"
    if age_days >= 365:
        return f"{age_days / 365:.1f}y"
    if age_days >= 60:
        return f"{age_days / 30.44:.0f}mo"
    return f"{age_days:.0f}d"


def has_discount(value) -> bool:
    # Only discounts that survive .0% rounding count; smaller slivers and
    # malformed (negative) values are treated as no discount everywhere:
    # the DISC column, --json output, and the --discount filter.
    return bool(value) and value > 0 and f"{value:.0%}" != "0%"


def fmt_discount(value) -> str:
    return f"{value:.0%}" if has_discount(value) else "--"


def fmt_prec(cand) -> str:
    """PREC cell: the precision the ranked price is quoted at.

    The maker's default precision when disclosed, the best available
    otherwise, "n/a" when none could be derived; "!" flags default routing
    that favors cheaper sub-default endpoints (annotations.risk_flag).
    Candidates that never went through the endpoint stage degrade to "n/a".
    """
    cell = cand.get("default_precision") or "n/a"
    if (cand.get("annotations") or {}).get("risk_flag"):
        cell += "!"
    return cell


def fmt_quant_note(annotations: dict) -> str | None:
    """One-line note on the cheapest sub-default endpoint (PREC tooltip).

    The exact string the site shows (Task 11 renders it verbatim), so the
    format is pinned by test. None when there is no non-eligible live
    endpoint -- nothing cheaper to route to. cheapest_quantized.price is
    stored per-token, the way build_annotations leaves it; the note shows
    USD per 1M like every other price the CLI prints. A missing spread
    (defensive) drops the parenthetical. A null quantization (third-party
    null == "unknown" for eligibility) and a null provider_name on
    malformed data render the design's "unknown" sentinel, never Python's
    None repr.
    """
    quant = (annotations or {}).get("cheapest_quantized")
    if not quant:
        return None
    quantization = quant["quantization"] or "unknown"
    provider = quant["provider"] or "unknown"
    note = (
        f"cheapest non-baseline: {quantization} @ {provider}, "
        f"${fmt_price(quant['price'] * 1_000_000)}/M"
    )
    spread = (annotations or {}).get("price_spread")
    if spread is not None:
        note += f" in ({spread:.2f}x baseline)"
    return note


def print_table(top, total_candidates, weights, quality_note):
    headers = [
        "RANK",
        "MODEL",
        "QUAL",
        "$IN/M",
        "$OUT/M",
        "DISC",
        "PREC",
        "TIER",
        "CTX",
        "AGE",
        "SCORE",
    ]
    rows = []
    for rank, cand in enumerate(top, 1):
        rows.append(
            [
                str(rank),
                cand["id"],
                "-" if cand["quality"] is None else f"{cand['quality']:.1f}",
                fmt_price(cand["price_in"]),
                fmt_price(cand["price_out"]),
                fmt_discount(cand["discount"]),
                fmt_prec(cand),
                cand["tier_note"] or cand["sched_note"] or "--",
                fmt_context(cand["context"]),
                fmt_age(cand["age_days"]),
                f"{cand['score']:.3f}",
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def emit(row):
        if row[1] == "MODEL":
            line = "  ".join(h.ljust(w) for h, w in zip(row, widths))
            print(line)
            print("  ".join("-" * w for w in widths))
        else:
            cells = [row[0].rjust(widths[0]), row[1].ljust(widths[1])]
            cells += [c.rjust(w) for c, w in zip(row[2:], widths[2:])]
            print("  ".join(cells))

    emit(headers)
    for row in rows:
        emit(row)
    weight_bits = ", ".join(
        f"{k} {v:.0%}" for k, v in sorted(weights.items(), reverse=True)
    )
    print()
    print(
        f"pool: {total_candidates} candidates | weights: {weight_bits} | {quality_note}"
    )
    print(
        "quality: Artificial Analysis intelligence index (- = unknown); prices in USD per 1M tokens; "
        "DISC = active discount; AGE = time since listed on OpenRouter"
    )
    print(
        "Prices shown are what a long session pays: several models bill at a "
        "higher rate once the prompt outgrows their cheap short-context tier."
    )
    print(
        "TIER marks how prices vary: a token threshold is the prompt size "
        "above which the higher rate applies; a percentage is a scheduled "
        "off-peak discount (price shown: the standard rate); SCHED marks a "
        "schedule whose discount is under 1%. --json lists each model's "
        "schedule."
    )
    print(
        "PREC = precision the ranked price is quoted at (maker's default when "
        "disclosed, best available otherwise; ! = default routing favors "
        "cheaper sub-default endpoints); prices are the cheapest live endpoint "
        "at that precision."
    )


def print_json(top):
    payload = [
        {
            "model": cand["id"],
            "opencode_model": opencode_model_id(cand["id"]),
            "name": cand["name"],
            "score": round(cand["score"], 4),
            "quality_index": cand["quality"],
            "input_usd_per_m": round(cand["price_in"], 6),
            "output_usd_per_m": round(cand["price_out"], 6),
            "blended_usd_per_m": round(cand["blended"], 6),
            "discount": round(cand["discount"], 4)
            if has_discount(cand["discount"])
            else None,
            # Display string from the DISC column's formatter, so the site
            # shows exactly what the CLI prints instead of re-rounding.
            "discount_pct": fmt_discount(cand["discount"]),
            # Tier/schedule fields follow the same precedent: the strings the
            # site shows are these, never re-formatted downstream.
            "pricing_tier_prompt_tokens": cand["tier_prompt_tokens"],
            "base_input_usd_per_m": round(cand["base_price_in"], 6),
            "base_output_usd_per_m": round(cand["base_price_out"], 6),
            "tier_note": cand["tier_note"],
            "sched_note": cand["sched_note"],
            "sched_detail": cand["sched_detail"],
            "time_schedule": (
                cand["sched_note"] + " " + cand["sched_detail"]
                if cand["sched_detail"]
                else cand["sched_note"]
            ),
            "context_tokens": cand["context"],
            "age_days": round(cand["age_days"], 1)
            if cand["age_days"] is not None
            else None,
            # Additive keys (Task 9's ROW_KEYS extension pins this exact
            # key set and order). Risk/spread come from the annotations
            # block; session_cost_usd is run()'s per-candidate slice of the
            # Task 7 envelope, shown for the display turn range (run()
            # attaches it before any output path). Candidates that never
            # went through either stage (historical test paths) degrade to
            # null, as does a model the dataset covers only in other turn
            # ranges.
            "default_precision": cand.get("default_precision"),
            "risk_flag": (cand.get("annotations") or {}).get("risk_flag"),
            "price_spread": (cand.get("annotations") or {}).get("price_spread"),
            "session_cost_usd": (cand.get("session_cost") or {}).get(
                SESSION_COST_DISPLAY_TURN_RANGE
            ),
            # The cheapest-sub-default note, exactly as the CLI prints it
            # (Task 11 shows it verbatim). Emit-first: Task 9's ROW_KEYS
            # pins this key last; pre-endpoint candidates degrade to None.
            "quant_note": fmt_quant_note(cand.get("annotations")),
            # One extra row key AFTER quant_note -- steering is NOT in
            # ROW_KEYS. build_site_data's row check is missing-key-only, so
            # extras pass through (tier_note precedent) and the site
            # ignores it. The OpenRouter provider routing object (design
            # section 4.7) the plugin contract reads; pre-endpoint
            # candidates degrade to None.
            "steering": cand.get("steering"),
        }
        for cand in top
    ]
    print(json.dumps(payload, indent=2))


def build_catalog(
    args,
    models,
    candidates,
    dropped,
    filtered,
    discounts,
    quality_by_id,
    aa_source,
    aa_by_id,
    quality_source_by_id,
    session_cost=None,
    session_cost_mode="skipped",
):
    """Assemble the full-catalog document (see README "Catalog output").

    Pure: reads candidates post-compute_scores (which already carry the
    component scores) and never mutates them. Deterministic apart from
    generated_at: models sort by (-overall.balanced, id), filtered by id,
    and age uses date precision so two runs in the same UTC day match.

    session_cost is the Task 7 fetch envelope (verbatim, medians included)
    or None; session_cost_mode is run()'s resolution of the key-presence
    fact ("ok" / "unavailable" / "skipped") -- build_catalog stays pure and
    never fetches or inspects the environment itself.
    """
    now = datetime.now(timezone.utc)
    weights = catalog_weights(candidates, quality_by_id)
    aa_modes = {"AA API v2": "api"}
    if aa_source is not None and aa_source not in aa_modes:
        # Never claim mode "none" for a source we do not know: the document
        # would contradict itself (mode none with matched quality scores).
        raise ValueError(f"unknown AA source: {aa_source!r}")
    matched_openrouter = sum(
        1 for s in quality_source_by_id.values() if s == "openrouter"
    )
    # fallback is what the AA API yielded on its own; mode
    # hides it behind "openrouter" as soon as one OpenRouter match exists.
    aa_fallback = aa_modes.get(aa_source, "none")
    aa_mode = "openrouter" if matched_openrouter else aa_fallback

    entries = []
    for cand in candidates:
        provider, _, _base = cand["id"].partition("/")
        created = cand["created"] or 0.0
        listed_date = (
            datetime.fromtimestamp(created, tz=timezone.utc).date()
            if created > 0
            else None
        )
        # Round the component scores first, then derive overall from the
        # rounded values so scores.overall is exactly reproducible from the
        # document's own numbers.
        scores = {
            "price": round(cand["price_score"], 4),
            "quality": round(cand["quality_score"], 4),
            "context": round(cand["context_score"], 4),
            "age": round(cand["age_score"], 4),
        }
        overall = {}
        for priority in PRIORITY_WEIGHTS:
            w = weights[priority]
            overall[priority] = round(
                w.get("quality", 0.0) * scores["quality"]
                + w.get("price", 0.0) * scores["price"]
                + w.get("context", 0.0) * scores["context"]
                + w.get("age", 0.0) * scores["age"],
                4,
            )
        bench = aa_by_id.get(base_model_id(cand["id"])) or {}
        entries.append(
            {
                "id": cand["id"],
                "name": PROVIDER_PREFIX_RE.sub("", cand["name"]).strip()
                or cand["name"],
                "provider": provider,
                "family": model_family(cand["id"]),
                "pricing": {
                    "input_per_1m": round(cand["price_in"], 6),
                    "output_per_1m": round(cand["price_out"], 6),
                    "blended_per_1m": round(cand["blended"], 6),
                    "base": {
                        "input_per_1m": round(cand["base_price_in"], 6),
                        "output_per_1m": round(cand["base_price_out"], 6),
                        "blended_per_1m": round(
                            args.input_share * cand["base_price_in"]
                            + (1 - args.input_share) * cand["base_price_out"],
                            6,
                        ),
                    },
                    "tiers": [
                        {
                            "min_prompt_tokens": tier["min_prompt_tokens"],
                            "input_per_1m": _per_1m(tier["price_in"]),
                            "output_per_1m": _per_1m(tier["price_out"]),
                            "blended_per_1m": _per_1m(
                                args.input_share * tier["price_in"]
                                + (1 - args.input_share) * tier["price_out"]
                            ),
                        }
                        for tier in cand["tiers"]
                    ],
                    "schedule": (
                        [
                            {
                                "utc_days": window["utc_days"],
                                "utc_start": window["utc_start"],
                                "utc_end": window["utc_end"],
                                "coverage": window["coverage"],
                                "input_per_1m": _per_1m(window["price_in"]),
                                "output_per_1m": _per_1m(window["price_out"]),
                                "blended_per_1m": _per_1m(window["blended"]),
                            }
                            for window in cand["schedule"]
                        ]
                        if cand["schedule"]
                        else None
                    ),
                },
                "context": cand["context"],
                "listed_at": listed_date.isoformat() if listed_date else None,
                "age_days": max(0, (now.date() - listed_date).days)
                if listed_date
                else None,
                "tool_calling": cand["tool_calling"],
                "zdr": cand["zdr"],
                "discount": round(cand["discount"], 4)
                if has_discount(cand["discount"])
                else None,
                "expired": cand["expired"],
                "aa": {
                    "intelligence_index": bench.get("intelligence_index"),
                    "coding_index": bench.get("coding_index"),
                    "agentic_index": bench.get("agentic_index"),
                },
                "quality": cand["quality"],
                "quality_match": quality_source_by_id.get(cand["id"]),
                "scores": {**scores, "overall": overall},
                # Additive provider-aware blocks (schema v3), verbatim from
                # the candidate. Candidates that never went through the
                # endpoint stage (historical test paths) degrade to null --
                # .get() with null defaults for these NEW keys only; the
                # pre-existing keys above stay direct-indexed.
                "endpoints": cand.get("endpoints"),
                "default_precision": cand.get("default_precision"),
                "advertised_pricing": cand.get("advertised_pricing"),
                "annotations": cand.get("annotations"),
                # The OpenRouter provider routing object (design section
                # 4.7) the plugin contract reads -- same null-default
                # degradation shape as the blocks above.
                "steering": cand.get("steering"),
            }
        )
    entries.sort(key=lambda e: (-e["scores"]["overall"]["balanced"], e["id"]))

    balanced_by_id = {
        entry["id"]: entry["scores"]["overall"]["balanced"] for entry in entries
    }

    return {
        "schedules": build_schedules(
            models, args, quality_by_id, aa_by_id, balanced_by_id
        ),
        "schema_version": CATALOG_SCHEMA_VERSION,
        "tool": "model-compare",
        "generated_at": now.isoformat(timespec="seconds"),
        "parameters": {
            "input_share": args.input_share,
            "quality_ref": args.quality_ref,
            "min_context": args.min_context,
            "recency_half_life": args.recency_half_life,
            "max_age_days": args.max_age_days,
            "zdr_required": not args.no_zdr,
            "require_tools": not args.no_require_tools,
            "exclude_free": args.exclude_free,
            "include_batch": args.include_batch,
            "weights": weights,
        },
        "sources": {
            "openrouter": "ok",
            "aa": {
                "mode": aa_mode,
                "fallback": aa_fallback,
                "matched": len(quality_by_id),
                "matched_openrouter": matched_openrouter,
            },
            "zdr": "skipped" if args.no_zdr else "ok",
            "discounts": "ok" if discounts else "unavailable",
            # The endpoint stage always ran before build_catalog is reached;
            # its failures surface as pool.dropped counts
            # (endpoint_fetch_failed / no_baseline_eligible_endpoint), never
            # as source state -- so this is deterministic "ok".
            "endpoints": "ok",
            # "ok" (envelope fetched) / "unavailable" (key configured, fetch
            # failed) / "skipped" (no key) -- resolved in run().
            "session_cost": session_cost_mode,
        },
        # Per-task session-cost medians (Task 7), verbatim from the fetch
        # envelope. Omitted entirely when unavailable: consumers branch on
        # key presence, never on a null shape.
        **({"session_cost": session_cost} if session_cost is not None else {}),
        "pool": {
            "listed": len(models),
            "candidates": len(candidates),
            "dropped": {
                reason: dropped.get(reason, 0) for reason in CATALOG_DROP_REASONS
            },
        },
        "models": entries,
        # The single ranking authority: every site artifact (data.json rows,
        # history tabs, the highlights diff, best.txt) projects from or is
        # checked against it. models keeps its own balanced sort.
        "rankings": catalog_rankings(candidates, weights),
        "filtered": sorted(filtered, key=lambda e: e["id"]),
    }


def print_catalog(document):
    print(json.dumps(document, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="model-compare",
        description="Pick the best value-for-money LLM on OpenRouter for coding-agent work.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("Examples:", 1)[-1],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--min-context",
        type=int,
        default=1_000_000,
        help="hard minimum context window in tokens (default: 1000000)",
    )
    parser.add_argument(
        "--priority",
        choices=sorted(PRIORITY_WEIGHTS),
        default="balanced",
        help="ranking emphasis (default: balanced)",
    )
    parser.add_argument(
        "--top", type=int, default=5, help="how many models to list (default: 5)"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="print only the #1 model as an opencode id (openrouter/<model>), for scripting",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the ranked candidates as JSON"
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="print the full model catalog (ranked candidates and filtered-out models with reasons) as one JSON document; ignores --top and --priority, cannot be combined with --best or --json",
    )
    parser.add_argument(
        "--input-share",
        type=float,
        default=0.75,
        help="input-token share used for the blended price, 0-1 (default: 0.75)",
    )
    parser.add_argument(
        "--recency-half-life",
        type=float,
        default=120.0,
        help="age decay half-life in days (default: 120)",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=0.0,
        help="drop models listed more than this many days ago (0 = off)",
    )
    parser.add_argument(
        "--quality-ref",
        type=float,
        default=70.0,
        help="AA intelligence index that counts as full quality score (default: 70)",
    )
    parser.add_argument(
        "--aa-api-key",
        default=None,
        help="Artificial Analysis API key (or set AA_API_KEY; free at artificialanalysis.ai)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY; free at openrouter.ai/keys)",
    )
    parser.add_argument(
        "--no-require-tools",
        action="store_true",
        help="do not require tool-calling support (coding agents want it)",
    )
    parser.add_argument(
        "--exclude-free",
        action="store_true",
        help="drop zero-cost listings such as rate-limited :free variants",
    )
    parser.add_argument(
        "--include-batch",
        action="store_true",
        help="keep ':batch' variants (asynchronous completion; cheaper but unsuitable for interactive agents)",
    )
    parser.add_argument(
        "--discount",
        action="store_true",
        help="only list models with an active discount",
    )
    parser.add_argument(
        "--no-zdr",
        action="store_true",
        help="consider all models, not just zero-data-retention (ZDR) ones",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the local response cache and refetch",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=6 * 3600,
        help="cache lifetime in seconds (default: 21600 = 6h)",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.input_share <= 1.0:
        parser.error("--input-share must be between 0 and 1")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.recency_half_life <= 0 or args.quality_ref <= 0:
        parser.error("--recency-half-life and --quality-ref must be positive")
    if args.catalog and (args.best or args.json):
        parser.error("--catalog cannot be combined with --best or --json")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        models, or_cached = fetch_openrouter_models(args)
    except Exception as exc:
        warn(f"could not fetch the OpenRouter catalog: {exc}")
        return 1
    try:
        return run(args, models, or_cached)
    except Exception as exc:
        warn(f"could not rank models: {exc}")
        return 1


def run(args, models, or_cached) -> int:
    discounts, zdr_ids, aa_by_id, frontend_cache_hits = fetch_openrouter_frontend(args)
    if not args.no_zdr and not zdr_ids:
        warn(
            "ZDR data unavailable; refusing to rank possibly non-ZDR models "
            "(--no-zdr to override)"
        )
        return 1
    aa_entries, aa_source, aa_cached = fetch_aa_entries(args)
    exact, fuzzy = build_aa_lookup(aa_entries)

    filtered = []
    candidates, dropped = build_candidates(models, args, discounts, zdr_ids, filtered)
    # The provider-aware stage: every surviving candidate is repriced at its
    # baseline endpoint (or dropped fail-closed) before any scoring runs.
    candidates, extra_dropped = apply_endpoint_pricing(candidates, args, filtered)
    dropped.update(extra_dropped)
    if not candidates:
        # When the endpoint stage wiped the pool, naming the filter knobs is
        # misleading (they passed; the drops came after) -- warn with the
        # endpoint-stage drop counts instead (both reasons, zero-filled).
        counts = {r: extra_dropped.get(r, 0) for r in ENDPOINT_DROP_REASONS}
        if any(counts.values()):
            bits = ", ".join(f"{k}={v}" for k, v in counts.items())
            warn(f"no models after the endpoint stage: {bits}")
        else:
            warn(
                f"no models satisfy the filters (min-context={args.min_context}, "
                f"tools={'off' if args.no_require_tools else 'required'}); "
                "relax them and retry"
            )
        return 2

    quality_by_id, quality_source_by_id = resolve_quality(
        candidates, aa_by_id, exact, fuzzy, aa_source
    )
    weights = compute_scores(candidates, args, quality_by_id)

    # Per-task session-cost medians (Task 7): one fetch per invocation, but
    # only on the output paths that surface it (--catalog block, --json
    # rows; publish runs --json per priority, so the cache makes the repeat
    # cost free). A keyed --best/plain run skips the paginated fetch
    # entirely -- the medians never reach those outputs. Keyless,
    # fetch_session_cost returns None before any network or cache access.
    if args.catalog or args.json:
        session_cost = fetch_session_cost(args)
        # The key-presence fact distinguishes "skipped" from "unavailable";
        # same expression fetch_session_cost uses for its own guard.
        session_cost_mode = (
            "skipped"
            if not (args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"))
            else ("ok" if session_cost else "unavailable")
        )
    else:
        session_cost = None
        session_cost_mode = "skipped"
    # run() owns the per-candidate attachment (build_catalog stays pure);
    # print_json reads it for the session_cost_usd display cell.
    for cand in candidates:
        cand["session_cost"] = (session_cost or {}).get("medians", {}).get(cand["id"])

    if args.catalog:
        print_catalog(
            build_catalog(
                args,
                models,
                candidates,
                dropped,
                filtered,
                discounts,
                quality_by_id,
                aa_source,
                aa_by_id,
                quality_source_by_id,
                session_cost,
                session_cost_mode,
            )
        )
        return 0

    limit = 1 if args.best else args.top
    top = candidates[:limit]

    if args.best:
        print(opencode_model_id(top[0]["id"]))
    elif args.json:
        print_json(top)
    else:
        drop_note = ""
        if dropped:
            bits = ", ".join(
                f"{v} {k}" for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])
            )
            drop_note = f" (dropped: {bits})"
        matched_or = sum(1 for s in quality_source_by_id.values() if s == "openrouter")
        matched_aa = len(quality_by_id) - matched_or
        unmatched = len(candidates) - len(quality_by_id)
        bits = []
        if matched_or:
            bits.append(f"OpenRouter benchmarks ({matched_or})")
        if matched_aa:
            bits.append(f"{aa_source} exact ({matched_aa})")
        if bits:
            quality_note = (
                f"quality via {' + '.join(bits)}: matched "
                f"{len(quality_by_id)}/{len(candidates)} candidates"
            )
            if unmatched:
                quality_note += "; unmatched candidates score 0 on quality"
        else:
            quality_note = "quality data unavailable, ranked on price/context/age"
        source_note = []
        if or_cached:
            source_note.append("catalog cached")
        if "discounts" in frontend_cache_hits:
            source_note.append("discounts cached")
        if "zdr" in frontend_cache_hits:
            source_note.append("ZDR cached")
        if "aa" in frontend_cache_hits:
            source_note.append("AA benchmarks cached")
        if aa_cached and aa_source:
            source_note.append("quality cached")
        if source_note:
            print(
                f"note: {', '.join(source_note)} (use --no-cache to refresh)",
                file=sys.stderr,
            )
        print(
            f"pool: {len(models)} listed -> {len(candidates)} candidates{drop_note}",
            file=sys.stderr,
        )
        print_table(top, len(candidates), weights, quality_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
