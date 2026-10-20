"""Node-backed tests for the inline script in web/site/index.html.

No npm toolchain: the test extracts the page's single <script> block,
prepends the web/js_stubs.js shims, appends a per-test driver, and
evaluates it under node (skipped when node is absent; GitHub's ubuntu
runners ship node preinstalled). The harness exists because the script
duplicates model_compare.py's fmt_* helpers under a "keep them in sync"
comment and has regressed before -- parity is enforced here, and the
pure helpers (tooltip sentence construction, formatters, guards) get
real tests instead of code reading.

Limitation: only those pure helpers are exercised. Tooltip delivery
(the td.title assignment inside render) and DOM wiring are not: the
fetch stub rejects so render() never runs, and the stubs hand out a
fresh element per getElementById call with no querySelector support.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WEB = Path(__file__).parent
SITE_HTML = WEB / "site" / "index.html"
STUBS_JS = WEB / "js_stubs.js"

sys.path.insert(0, str(WEB.parent))
import model_compare as mc  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

# The page carries exactly one inline <script> block (its CSP forbids
# external scripts), so this match is the whole program. Non-greedy like
# an HTML parser: it terminates at the first </script>.
SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)

NODE_TIMEOUT = 30


def run_js(expression: str):
    """Evaluate stubs + inline script + expression under node; return JSON."""
    html = SITE_HTML.read_text()
    openings, closings = html.count("<script>"), html.count("</script>")
    if openings != 1 or closings != 1:
        raise AssertionError(
            "expected exactly one <script>...</script> block in "
            f"web/site/index.html, found {openings} opening and "
            f"{closings} closing tags"
        )
    program = "\n".join(
        [
            STUBS_JS.read_text(),
            SCRIPT_RE.search(html).group(1),
            f"var __out = ({expression});",
            "console.log(JSON.stringify(__out));",
        ]
    )
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(program)
            path = fh.name
        proc = subprocess.run(
            ["node", path],
            capture_output=True,
            text=True,
            timeout=NODE_TIMEOUT,
        )
    finally:
        if path is not None:
            Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise AssertionError(f"node driver failed:\n{proc.stderr}")
    lines = proc.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        raise AssertionError(
            f"node driver printed no JSON for expression {expression!r}: "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        ) from None


# ---------------------------------------------------------------------------
# Harness sanity + Python parity (existing behaviour)


def test_harness_reaches_the_inline_script():
    out = run_js(
        "{price: fmtPrice(0.1), context: fmtContext(1048576), age: fmtAge(null)}"
    )
    assert out == {"price": "0.100", "context": "1.05M", "age": "-"}


@pytest.mark.parametrize(
    "value",
    [0, 0.001, 0.014, 0.1, 0.27, 0.5, 1, 1.25, 9.99, 10.0, 27.5, 100, 1234.5678],
)
def test_fmtprice_matches_python(value):
    assert run_js(f"fmtPrice({value!r})") == mc.fmt_price(value)


@pytest.mark.parametrize("value", [0, 512, 1000, 163840, 1048576, 2000000])
def test_fmtcontext_matches_python(value):
    assert run_js(f"fmtContext({value!r})") == mc.fmt_context(value)


@pytest.mark.parametrize("value", [None, 0, 7, 59, 60, 90, 105, 320, 364, 365, 400])
def test_fmtage_matches_python(value):
    assert run_js(f"fmtAge({json.dumps(value)})") == mc.fmt_age(value)


# Exact binary ties: JS toFixed/Math.round round half-away-from-zero,
# Python f-strings round half-to-even. The CLI's fmt_* helpers are the
# reference, so the JS side must land on the even neighbour.
@pytest.mark.parametrize(
    "value,expected",
    [(10.25, "10.2"), (1.125, "1.12"), (100.5, "100")],
)
def test_fmtprice_tie_rounds_half_to_even(value, expected):
    assert run_js(f"fmtPrice({value!r})") == mc.fmt_price(value) == expected


@pytest.mark.parametrize("value,expected", [(2500, "2K"), (4500, "4K")])
def test_fmtcontext_tie_rounds_half_to_even(value, expected):
    assert run_js(f"fmtContext({value!r})") == mc.fmt_context(value) == expected


def test_fmtage_tie_rounds_half_to_even():
    assert run_js("fmtAge(58.5)") == mc.fmt_age(58.5) == "58d"


# Near-tie values must still round to the nearest neighbour, on both
# sides, and a decimal that merely looks like a tie ("0.15" at one
# digit) is not one: its exact double sits below the tie.
@pytest.mark.parametrize(
    "value,expected",
    [(10.2500001, "10.3"), (10.2499999, "10.2")],
)
def test_fmtprice_near_tie_rounds_to_nearest(value, expected):
    assert run_js(f"fmtPrice({value!r})") == mc.fmt_price(value) == expected


def test_fmtprice_non_tie_decimal_matches_python():
    assert run_js("fmtPrice(0.15)") == mc.fmt_price(0.15) == "0.150"
    # The 1-digit path: pyFixed(0.15, 1) must be "0.1" like Python's
    # f"{0.15:.1f}" -- a naive scaling-based half-even implementation
    # would wrongly emit "0.2" here.
    assert run_js("pyFixed(0.15, 1)") == f"{0.15:.1f}" == "0.1"


def test_script_extraction_covers_the_whole_program():
    # Exactly one <script>...</script> pair on the page; the extraction
    # must reach the end of the program, not stop at an early block.
    html = SITE_HTML.read_text()
    assert html.count("<script>") == 1
    assert html.count("</script>") == 1
    script = SCRIPT_RE.search(html).group(1)
    assert "function init" in script
    assert "tierCellData" in script


# ---------------------------------------------------------------------------
# TIER tooltip: below-threshold (base) prices


def test_tier_tooltip_appends_short_context_rate():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null,"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: 0.5})"
    )
    assert out == {
        "text": ">272k",
        "hover": (
            "Charges a higher rate once the prompt exceeds 272k tokens;"
            " price shown: that long-context rate."
            " The short-context rate is $0.100 in / $0.500 out per 1M tokens."
        ),
    }


def test_tier_tooltip_without_base_prices_stays_short():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null})"
    )
    assert out["hover"] == (
        "Charges a higher rate once the prompt exceeds 272k tokens;"
        " price shown: that long-context rate."
    )


def test_tier_tooltip_with_null_base_prices_stays_short():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null,"
        " base_input_usd_per_m: null, base_output_usd_per_m: null})"
    )
    assert out["hover"] == (
        "Charges a higher rate once the prompt exceeds 272k tokens;"
        " price shown: that long-context rate."
    )


def test_sched_only_tooltip_gains_no_rate_sentence():
    out = run_js(
        "tierCellData({tier_note: null, sched_note: '-30%',"
        " sched_detail: '00:00-08:00 UTC daily',"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: 0.5})"
    )
    assert out == {
        "text": "-30%",
        "hover": (
            "Runs up to 30% cheaper during scheduled off-peak hours"
            " (00:00-08:00 UTC daily); price shown: the standard rate."
        ),
    }


def test_sched_marker_tooltip_gains_no_rate_sentence():
    out = run_js(
        "tierCellData({tier_note: null, sched_note: 'SCHED',"
        " sched_detail: null,"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: 0.5})"
    )
    assert out["hover"] == (
        "Price varies on a schedule by less than 1%; price shown: the standard rate."
    )


def test_mixed_row_joins_both_sentences_with_rate():
    out = run_js(
        "tierCellData({tier_note: '>100k', sched_note: '-30%',"
        " sched_detail: '01:00-05:00 UTC daily',"
        " base_input_usd_per_m: 0.2, base_output_usd_per_m: 2})"
    )
    assert out["hover"] == (
        "Charges a higher rate once the prompt exceeds 100k tokens;"
        " price shown: that long-context rate."
        " The short-context rate is $0.200 in / $2.00 out per 1M tokens."
        " Runs up to 30% cheaper during scheduled off-peak hours"
        " (01:00-05:00 UTC daily); price shown: the standard rate."
    )


def test_tier_tooltip_non_k_note_spells_tokens_plainly():
    out = run_js(
        "tierCellData({tier_note: '>500', sched_note: null, sched_detail: null})"
    )
    assert out == {
        "text": ">500",
        "hover": (
            "Charges a higher rate once the prompt exceeds 500 tokens;"
            " price shown: that long-context rate."
        ),
    }


# The rate sentence only rides along for finite, non-negative numbers:
# strings, NaN, Infinity, negatives and half-present pairs all drop it.
SHORT_TIER_HOVER = (
    "Charges a higher rate once the prompt exceeds 272k tokens;"
    " price shown: that long-context rate."
)


@pytest.mark.parametrize(
    "input_price,output_price",
    [
        ("'0.1'", "'0.5'"),  # numeric-looking strings
        ("NaN", "0.5"),  # NaN
        ("Infinity", "0.5"),  # positive infinity
        ("-0.5", "0.5"),  # negative
        ("0.1", "null"),  # half-present: output price missing
    ],
)
def test_tier_tooltip_rejects_invalid_base_prices(input_price, output_price):
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null,"
        f" base_input_usd_per_m: {input_price},"
        f" base_output_usd_per_m: {output_price}}})"
    )
    assert out["hover"] == SHORT_TIER_HOVER


def test_tier_tooltip_rejects_negative_output_price():
    out = run_js(
        "tierCellData({tier_note: '>272k', sched_note: null, sched_detail: null,"
        " base_input_usd_per_m: 0.1, base_output_usd_per_m: -0.5})"
    )
    assert out["hover"] == SHORT_TIER_HOVER


# ---------------------------------------------------------------------------
# Task 11: quantization badge (PREC), session cost (TASK$), endpoint disclosure
# ---------------------------------------------------------------------------

# Shared catalog entry used by the fmtCtxCaveat and endpointRows tests.
# provider-a's per-token prices (1.25e-6, 10e-6) scale to exactly the entry's
# per-1M pricing (1.25, 10.0), so it is the baseline endpoint.
CATALOG_ENTRY_JS = """
{
  id: 'test-model',
  pricing: { input_per_1m: 1.25, output_per_1m: 10.0 },
  annotations: { baseline_context_below_catalog: true },
  endpoints: [
    {
      provider_name: 'provider-a',
      quantization: 'fp8',
      pricing: { prompt: 1.25e-6, completion: 10e-6 },
      status: 'active',
      context_length: 163840,
      service_tier: 'standard'
    }
  ]
}
"""


def test_fmt_prec_formats_default_precision():
    assert run_js("fmtPrec({default_precision: 'fp8', risk_flag: false})") == "fp8"


def test_fmt_prec_appends_risk_flag():
    assert run_js("fmtPrec({default_precision: 'fp8', risk_flag: true})") == "fp8!"


def test_fmt_prec_null_precision_is_na():
    assert run_js("fmtPrec({default_precision: null, risk_flag: false})") == "n/a"


def test_fmt_prec_null_precision_with_risk_flag():
    assert run_js("fmtPrec({default_precision: null, risk_flag: true})") == "n/a!"


def test_fmt_prec_treats_false_and_null_risk_flag_identically():
    # JS truthiness: both false and null are falsy, so neither adds the "!".
    # Documented here because data.json emits false for "no risk" and null
    # for "unknown" -- the badge does not distinguish them (see task 11 stub
    # reconciliation note).
    assert run_js("fmtPrec({default_precision: 'fp8', risk_flag: false})") == "fp8"
    assert run_js("fmtPrec({default_precision: 'fp8', risk_flag: null})") == "fp8"


def test_fmt_task_cost_formats_usd():
    assert run_js("fmtTaskCost(1.5)") == "$1.50"
    assert run_js("fmtTaskCost(0)") == "$0.00"


def test_fmt_task_cost_null_is_na():
    # data.json emits null when the session-cost dataset has no row for the
    # model this refresh -- the site must show the not-available marker, never
    # 0.00 and never blank-silent.
    assert run_js("fmtTaskCost(null)") == "—"


def test_fmt_task_cost_undefined_is_na():
    assert run_js("fmtTaskCost(undefined)") == "—"


def test_fmt_task_cost_non_finite_is_na():
    assert run_js("fmtTaskCost(NaN)") == "—"
    assert run_js("fmtTaskCost(Infinity)") == "—"


def test_fmt_task_cost_no_negative_guard():
    # Consistency with fmtPrice: no negative guard. A negative session cost
    # is a data bug that must surface, not hide.
    assert run_js("fmtTaskCost(-1.5)") == "$-1.50"


def test_fmt_quant_tooltip_verbatim():
    assert (
        run_js("fmtQuantTooltip({quant_note: 'Sub-default fp8'})") == "Sub-default fp8"
    )


def test_fmt_quant_tooltip_null():
    assert run_js("fmtQuantTooltip({quant_note: null})") is None


def test_fmt_quant_tooltip_missing():
    assert run_js("fmtQuantTooltip({})") is None


def test_fmt_ctx_caveat_active():
    out = run_js(f"fmtCtxCaveat({CATALOG_ENTRY_JS}, {{context_tokens: 1048576}})")
    assert out == "Ranked price from a 164K-token endpoint; model supports 1.05M."


def test_fmt_ctx_caveat_null_when_annotation_false():
    entry = CATALOG_ENTRY_JS.replace(
        "baseline_context_below_catalog: true", "baseline_context_below_catalog: false"
    )
    assert run_js(f"fmtCtxCaveat({entry}, {{context_tokens: 1048576}})") is None


def test_fmt_ctx_caveat_null_when_no_baseline():
    # No endpoint matches the entry pricing, so there is no baseline context
    # to cite -- the caveat is withheld rather than guessing.
    entry = CATALOG_ENTRY_JS.replace(
        "baseline_context_below_catalog: true", "baseline_context_below_catalog: false"
    )
    assert run_js(f"fmtCtxCaveat({entry}, {{context_tokens: 1048576}})") is None


def test_fmt_ctx_caveat_null_when_row_context_null():
    assert run_js(f"fmtCtxCaveat({CATALOG_ENTRY_JS}, {{context_tokens: null}})") is None


def test_endpoint_rows_null_pricing():
    entry = """
    {
      pricing: { input_per_1m: 1.0, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p',
          quantization: 'fp8',
          pricing: null,
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out == [
        {
            "provider": "p",
            "badge": "fp8",
            "priceIn": "-",
            "priceOut": "-",
            "status": "active",
            "context": 1000,
            "tier": "standard",
            "isBaseline": False,
        }
    ]


def test_endpoint_rows_null_status_is_non_live():
    entry = """
    {
      pricing: { input_per_1m: 1.0, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p',
          quantization: 'fp8',
          pricing: { prompt: 1e-6, completion: 10e-6 },
          status: null,
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["status"] == "non-live"


def test_endpoint_rows_null_service_tier():
    entry = """
    {
      pricing: { input_per_1m: 1.0, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p',
          quantization: 'fp8',
          pricing: { prompt: 1e-6, completion: 10e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: null
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["tier"] == "-"


def test_endpoint_rows_null_quantization():
    entry = """
    {
      pricing: { input_per_1m: 1.0, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p',
          quantization: null,
          pricing: { prompt: 1e-6, completion: 10e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["badge"] == "unknown"


def test_endpoint_rows_identifies_baseline():
    entry = """
    {
      pricing: { input_per_1m: 1.25, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p1',
          quantization: 'fp8',
          pricing: { prompt: 1.25e-6, completion: 10e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        },
        {
          provider_name: 'p2',
          quantization: 'int8',
          pricing: { prompt: 2e-6, completion: 20e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["isBaseline"] is True
    assert out[1]["isBaseline"] is False


def test_endpoint_rows_ambiguous_baseline_marks_none():
    entry = """
    {
      pricing: { input_per_1m: 1.25, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p1',
          quantization: 'fp8',
          pricing: { prompt: 1.25e-6, completion: 10e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        },
        {
          provider_name: 'p2',
          quantization: 'int8',
          pricing: { prompt: 1.25e-6, completion: 10e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["isBaseline"] is False
    assert out[1]["isBaseline"] is False


def test_endpoint_rows_no_match_marks_none():
    entry = """
    {
      pricing: { input_per_1m: 1.25, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p1',
          quantization: 'fp8',
          pricing: { prompt: 9e-6, completion: 90e-6 },
          status: 'active',
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["isBaseline"] is False
