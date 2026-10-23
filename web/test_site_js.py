"""Node-backed tests for the site's app.js (web/site/app.js).

No npm toolchain: the test prepends the web/js_stubs.js shims to
app.js, appends a per-test driver, and evaluates it under node
(skipped when node is absent; GitHub's ubuntu runners ship node
preinstalled). The harness exists because app.js duplicates
model_compare.py's fmt_* helpers under a "keep them in sync" comment
and has regressed before -- parity is enforced here, and the pure
helpers (tooltip sentence construction, formatters, guards) get real
tests instead of code reading.

Limitation: the pure helpers, the built-element shape (endpointRow,
render on a stub tbody), the named render helpers that mutate captured
elements (renderFooter's revision/stamp flow), and the event wiring
(stubs record handlers per type; a driver fires clicks on the
view/consent buttons) are exercised; tooltip delivery (the td.title
assignment inside render) and the fetch-driven flow are not: the fetch
stub rejects so init()'s render() never runs and querySelectorAll
returns []. The stubs cache one element per id, so a driver can read
back what a helper wrote through els or document.getElementById.
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
SITE_CSS = WEB / "site" / "style.css"
SCRIPT_JS = WEB / "site" / "app.js"
STUBS_JS = WEB / "js_stubs.js"

sys.path.insert(0, str(WEB.parent))
import model_compare as mc  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

NODE_TIMEOUT = 30


def run_js(expression: str):
    """Evaluate stubs + app.js + expression under node; return JSON."""
    program = "\n".join(
        [
            STUBS_JS.read_text(),
            SCRIPT_JS.read_text(),
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


def test_harness_reaches_the_site_script():
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


# ---------------------------------------------------------------------------
# Site assembly: external assets, view toggle, cookie consent, verbadge


def test_site_references_the_external_assets():
    # The former inline CSS/JS blocks are external files now: the page
    # links them, and its CSP allows same-origin assets but no inline
    # code (the hand-marked JSON highlighting is markup, not script).
    html = SITE_HTML.read_text()
    assert '<link rel="stylesheet" href="style.css" />' in html
    assert '<script src="app.js" defer></script>' in html
    assert "style-src 'self'" in html
    assert "script-src 'self'" in html
    assert "unsafe-inline" not in html
    assert html.count("<style") == 0
    # every <script ...> carries a src: no inline script blocks remain
    assert re.search(r"<script(?![^>]*\bsrc=)", html) is None
    # CSP tripwires the tightened policy would silently block in
    # production while every behavioral test stays green:
    assert re.search(r"<[^>]*\son[a-z]+\s*=", html) is None  # inline handlers
    assert re.search(r"<[^>]*\sstyle\s*=", html) is None  # inline styles
    assert "javascript:" not in html
    css = SITE_CSS.read_text()
    assert "@import" not in css and "url(" not in css  # no external fetches
    assert SITE_CSS.is_file()
    script = SCRIPT_JS.read_text()
    assert "function init" in script
    assert "tierCellData" in script


def test_view_toggle_buttons_markup():
    html = SITE_HTML.read_text()
    tabs = re.search(r'<div class="tabs" id="tabs">.*?</div>', html, re.DOTALL).group(0)
    simple_btn = re.search(r'<button[^>]*id="view-simple"[^>]*>', tabs).group(0)
    expert_btn = re.search(r'<button[^>]*id="view-expert"[^>]*>', tabs).group(0)
    # Expert is the default view: only its button starts active.
    assert "active" not in simple_btn
    assert "active" in expert_btn
    # The view buttons carry no data-priority: the priority wiring keys
    # on that attribute and must not hijack the layout toggle.
    assert "data-priority" not in simple_btn
    assert "data-priority" not in expert_btn
    # Order on the line: priority group, centered stamp, view buttons.
    assert (
        tabs.index('data-priority="quality"')
        < tabs.index('id="generated-top"')
        < tabs.index('id="view-simple"')
    )


def test_priority_wiring_scoped_to_data_priority():
    # The view buttons live in #tabs too: the priority click wiring keys
    # on [data-priority] so clicking Simple/Expert cannot blank the
    # table by setting current to undefined.
    script = SCRIPT_JS.read_text()
    assert script.count('document.querySelectorAll("#tabs button[data-priority]")') == 2


def test_simple_view_headers():
    html = SITE_HTML.read_text()
    main_table = re.search(r'<table id="table">.*?</table>', html, re.DOTALL).group(0)
    # The four abbreviated expert headers gain unabbreviated simple
    # labels; the price columns keep their dollar units.
    for expert_label, simple_label in [
        ("QUAL", "QUALITY"),
        ("$IN/M", "IN $/M"),
        ("$OUT/M", "OUT $/M"),
        ("DISC", "DISCOUNT"),
    ]:
        assert f'<span class="lbl-expert">{expert_label}</span>' in main_table
        assert f'<span class="lbl-simple">{simple_label}</span>' in main_table


def test_simple_view_hidden_columns_and_legend():
    html = SITE_HTML.read_text()
    main_table = re.search(r'<table id="table">.*?</table>', html, re.DOTALL).group(0)
    ths = re.findall(r"<th.*?</th>", main_table, re.DOTALL)
    for col in ("PREC", "TIER", "CTX", "SCORE", "TASK$"):
        th = next(
            (t for t in ths if re.search(r">\s*" + re.escape(col) + r"\s*<", t)),
            None,
        )
        assert th is not None, f"no <th> for {col}"
        assert "hide-simple" in th, f"{col} th missing hide-simple"
    # Legend hints for the absent columns are hidden in simple view; the
    # long-session-prices hint stays in both (it explains the IN/OUT
    # columns, which are present in both views).
    assert '<p class="hint hide-simple">\n      <strong>PREC</strong>' in html
    assert (
        '<p class="hint hide-simple">\n      <strong><span class="tier">TIER</span></strong>'
        in html
    )
    assert '<p class="hint hide-simple">\n      <strong>TASK$</strong>' in html
    assert '<p class="hint">\n      Prices shown are what a long session pays' in html


def test_opencode_headings_keep_version_text_drop_badge():
    html = SITE_HTML.read_text()
    assert "verbadge" not in html
    assert "<h3>OpenCode v1 — Launch flag, shell alias</h3>" in html
    assert "<h3>OpenCode v2 — model-compare plugin</h3>" in html


def test_cookie_dialog_markup():
    html = SITE_HTML.read_text()
    assert re.search(r'<div[^>]*id="cookie-dialog"[^>]*hidden', html)
    assert 'id="cookie-ok"' in html
    assert 'id="cookie-reject"' in html


def test_cookie_dialog_a11y_markup():
    # The consent dialog is a real modal to assistive tech and can take
    # focus programmatically; the view toggles expose their state.
    html = SITE_HTML.read_text()
    dialog = re.search(r'<div[^>]*id="cookie-dialog"[^>]*>', html, re.DOTALL).group(0)
    assert 'role="dialog"' in dialog
    assert 'aria-modal="true"' in dialog
    assert "aria-label=" in dialog
    assert 'tabindex="-1"' in dialog
    simple = re.search(r'<button[^>]*id="view-simple"[^>]*>', html).group(0)
    expert = re.search(r'<button[^>]*id="view-expert"[^>]*>', html).group(0)
    assert 'aria-pressed="false"' in simple
    assert 'aria-pressed="true"' in expert


def test_top_stamp_reads_last_refresh():
    # "last refresh: date, time" -- the browser's locale formatting
    # prefixed with the label. The footer keeps its bare stamp.
    # Behavioral, not source-text: renderFooter drives both stamps.
    out = run_js(
        "(function () {"
        "  renderFooter({"
        "    generated_at: '2026-09-28T06:00:00Z',"
        "    source_revision: 'abc1234',"
        "  });"
        "  return {"
        "    top: document.getElementById('generated-top').textContent,"
        "    footer: document.getElementById('generated').textContent,"
        "  };"
        "})()"
    )
    assert out["top"].startswith("last refresh: ")
    assert out["footer"] in out["top"]


def test_view_toggles_reflect_aria_pressed():
    # applyView keeps aria-pressed in sync with the active view so the
    # toggle state is announced, not just styled.
    out = run_js(
        "(function () {"
        "  var simple = document.getElementById('view-simple');"
        "  var expert = document.getElementById('view-expert');"
        "  var before = {"
        "    simple: simple.getAttribute('aria-pressed'),"
        "    expert: expert.getAttribute('aria-pressed'),"
        "  };"
        "  document.getElementById('view-simple').dispatchEvent('click');"
        "  var afterSimple = {"
        "    simple: simple.getAttribute('aria-pressed'),"
        "    expert: expert.getAttribute('aria-pressed'),"
        "  };"
        "  document.getElementById('view-expert').dispatchEvent('click');"
        "  var afterExpert = {"
        "    simple: simple.getAttribute('aria-pressed'),"
        "    expert: expert.getAttribute('aria-pressed'),"
        "  };"
        "  return { before: before, afterSimple: afterSimple, afterExpert: afterExpert };"
        "})()"
    )
    assert out["before"] == {"simple": "false", "expert": "true"}
    assert out["afterSimple"] == {"simple": "true", "expert": "false"}
    assert out["afterExpert"] == {"simple": "false", "expert": "true"}


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
    # Annotation flag is set but no endpoint matches the entry pricing, so
    # no baseline endpoint can be pinned and there is no context to cite --
    # the caveat is withheld rather than guessing.
    entry = CATALOG_ENTRY_JS.replace(
        "pricing: { prompt: 1.25e-6, completion: 10e-6 }",
        "pricing: { prompt: 9e-6, completion: 90e-6 }",
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


def test_endpoint_rows_live_status_zero_is_live():
    # The real catalog carries status as an INTEGER: 0 is live (the falsy
    # value a `|| "non-live"` guard would swallow). Regression pin: the
    # string fixtures above never exercised the live case.
    entry = """
    {
      pricing: { input_per_1m: 1.0, output_per_1m: 10.0 },
      endpoints: [
        {
          provider_name: 'p',
          quantization: 'fp8',
          pricing: { prompt: 1e-6, completion: 10e-6 },
          status: 0,
          context_length: 1000,
          service_tier: 'standard'
        },
        {
          provider_name: 'q',
          quantization: 'fp8',
          pricing: { prompt: 2e-6, completion: 20e-6 },
          status: -2,
          context_length: 1000,
          service_tier: 'standard'
        }
      ]
    }
    """
    out = run_js(f"endpointRows({entry})")
    assert out[0]["status"] == "live"
    assert out[1]["status"] == -2


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


# ---------------------------------------------------------------------------
# Disclosure note (baseline ambiguity), artifact-level vs model-level text,
# and the MAIN_TABLE_COLS colSpan pin.


def test_baseline_note_null_for_single_baseline():
    # One endpoint matching the entry pricing: one baseline, no note.
    assert run_js(f"baselineNote({CATALOG_ENTRY_JS})") is None


def test_baseline_note_ambiguous():
    # Two endpoints at the same price: both match the pricing scalars, so
    # the "✓" would be a guess -- the note says so instead of silence.
    entry = CATALOG_ENTRY_JS.replace(
        "endpoints: [",
        """endpoints: [
    {
      provider_name: 'provider-a2',
      quantization: 'int8',
      pricing: { prompt: 1.25e-6, completion: 10e-6 },
      status: 'active',
      context_length: 163840,
      service_tier: 'standard'
    },""",
    )
    assert run_js(f"baselineNote({entry})") == "baseline endpoint ambiguous — no marker"


def test_baseline_note_no_match():
    # No endpoint matches: also ambiguous (the pricing scalars fail to pin
    # exactly one endpoint) -- the note covers the empty case too.
    entry = CATALOG_ENTRY_JS.replace(
        "pricing: { prompt: 1.25e-6, completion: 10e-6 }",
        "pricing: { prompt: 9e-6, completion: 90e-6 }",
    )
    assert run_js(f"baselineNote({entry})") == "baseline endpoint ambiguous — no marker"


def test_baseline_note_null_without_endpoints():
    # No endpoint list at all: nothing to mark, silence is honest.
    assert run_js("baselineNote(null)") is None
    assert run_js("baselineNote({})") is None
    assert run_js("baselineNote({endpoints: []})") is None


def test_main_table_colspans_pin_header_counts():
    # The disclosure row must span the visible main-table width in both
    # views. The counts are named constants pinned to the actual <th>
    # markup: MAIN_TABLE_COLS to the full count, SIMPLE_TABLE_COLS to the
    # count minus the hide-simple columns, so a column addition that
    # forgets the constants fails here instead of misaligning the table.
    html = SITE_HTML.read_text()
    main_table = re.search(r'<table id="table">.*?</table>', html, re.DOTALL).group(0)
    header_cols = len(re.findall(r"<th[\s>]", main_table))
    hidden_cols = len(re.findall(r"<th[^>]*class=\"[^\"]*hide-simple", main_table))
    script = SCRIPT_JS.read_text()
    assert f"var MAIN_TABLE_COLS = {header_cols};" in script
    assert f"var SIMPLE_TABLE_COLS = {header_cols - hidden_cols};" in script
    assert (
        'td.colSpan = view === "simple" ? SIMPLE_TABLE_COLS : MAIN_TABLE_COLS;'
        in script
    )


def test_endpoint_row_colspan_follows_view():
    # Behavior pin: the disclosure row spans 13 columns in expert view
    # and 8 in simple view (built-element shape, like the other DOM tests).
    expert = run_js(
        "(function () {"
        '  view = "expert";'
        "  var tr = endpointRow(null, false);"
        "  return tr.children[0].colSpan;"
        "})()"
    )
    assert expert == 13
    simple = run_js(
        "(function () {"
        '  view = "simple";'
        "  var tr = endpointRow(null, false);"
        "  return tr.children[0].colSpan;"
        "})()"
    )
    assert simple == 8


# One ranked row shaped exactly as data.json v2 provides it: every field
# render() touches is present, with tier/sched/quant annotations unset.
SIMPLE_VIEW_ROW = {
    "model": "m/x",
    "opencode_model": "openrouter/m/x",
    "quality_index": 40.0,
    "input_usd_per_m": 1.0,
    "output_usd_per_m": 2.0,
    "discount_pct": "10",
    "default_precision": "fp8",
    "risk_flag": False,
    "tier_note": None,
    "sched_note": None,
    "context_tokens": 1000,
    "age_days": 30,
    "score": 0.5,
    "session_cost_usd": 1.0,
    "quant_note": None,
}


def test_render_row_shapes_follow_view():
    # render() must drop exactly the five expert-only columns in simple
    # view: the row keeps 8 cells (rank, 7-day, model, quality, in, out,
    # discount, age) and the 8th visible cell is AGE, where the expert
    # row's 8th is PREC. The stub tbody never clears its children, so
    # the expert row lands at index 0 and the simple row at index 1.
    data_fix = {
        "priorities": {"balanced": [SIMPLE_VIEW_ROW]},
        "schedules": [],
    }
    out = run_js(
        "(function () {"
        f" data = {json.dumps(data_fix)};"
        ' view = "expert"; render();'
        " var expertTr = els.tbody.children[0];"
        ' view = "simple"; render();'
        " var simpleTr = els.tbody.children[1];"
        " return {"
        "   expertCells: expertTr.children.length,"
        "   expertCell7: expertTr.children[7].textContent,"
        "   simpleCells: simpleTr.children.length,"
        "   simpleCell7: simpleTr.children[7].textContent"
        " };"
        "})()"
    )
    assert out == {
        "expertCells": 13,
        "expertCell7": "fp8",
        "simpleCells": 8,
        "simpleCell7": "30d",
    }


DEALS_ROW = {
    "model": "d/y",
    "sched_note": "-30%",
    "sched_detail": "00:00-08:00 UTC daily",
    "quality": 50.0,
    "input_per_1m": 0.5,
    "output_per_1m": 1.5,
    "context": 128000,
    "score": 0.7,
}


def test_render_deals_rows_are_view_independent():
    # The layout toggle switches the main table only: renderDeals must
    # emit the same 8 cells per row in both views, with the score last
    # (a view-conditional skip here would drop values under the wrong
    # headers, since the deals headers never hide). The stub hands out
    # a fresh element per getElementById call, so the test patches the
    # lookup to capture each tbody the function builds into.
    data_fix = {"priorities": {"balanced": []}, "schedules": [DEALS_ROW]}
    out = run_js(
        "(function () {"
        f" data = {json.dumps(data_fix)};"
        " var captured = null;"
        " var realGet = document.getElementById;"
        " document.getElementById = function (id) {"
        "   var e = realGet(id);"
        '   if (id === "deals-tbody") captured = e;'
        "   return e;"
        " };"
        ' view = "expert"; renderDeals();'
        " var expertTr = captured.children[0];"
        " captured = null;"
        ' view = "simple"; renderDeals();'
        " var simpleTr = captured.children[0];"
        " document.getElementById = realGet;"
        " return {"
        "   expertCells: expertTr.children.length,"
        "   expertLast: expertTr.children[7].textContent,"
        "   simpleCells: simpleTr.children.length,"
        "   simpleLast: simpleTr.children[7].textContent"
        " };"
        "})()"
    )
    assert out == {
        "expertCells": 8,
        "expertLast": "0.700",
        "simpleCells": 8,
        "simpleLast": "0.700",
    }


# ---------------------------------------------------------------------------
# Consent-first view cookie


def test_should_ask_for_consent_fresh_load():
    # No cookie, nothing rejected: the dialog must ask.
    assert run_js("shouldAskForConsent()") is True


def test_should_ask_for_consent_stored_consent():
    # A stored model-compare-view=simple cookie means consent was given:
    # no dialog.
    assert (
        run_js(
            '(function () { document.cookie = "model-compare-view=simple";'
            " return shouldAskForConsent(); })()"
        )
        is False
    )


def test_should_ask_for_consent_rejected_this_load():
    # Reject stores nothing but suppresses re-asking for this page load.
    assert (
        run_js(
            "(function () { consentRejected = true; return shouldAskForConsent(); })()"
        )
        is False
    )


def test_deals_table_immune_to_view_toggle():
    # The layout toggle switches the MAIN table only. The deals table
    # shares header markup with it (same QUAL/CTX th text), so a stray
    # copy of the view markers here rendered "QUAL QUALITY" in both
    # views and misaligned the deals columns in simple view.
    html = SITE_HTML.read_text()
    deals = re.search(r'<table id="deals"[^>]*>.*?</table>', html, re.DOTALL).group(0)
    assert "lbl-expert" not in deals
    assert "lbl-simple" not in deals
    assert "hide-simple" not in deals
    assert '<th title="Intelligence index (Artificial Analysis)">QUAL</th>' in deals
    assert '<th title="Context window in tokens">CTX</th>' in deals


def test_view_cookie_round_trip():
    # writeViewCookie/clearViewCookie must agree on the cookie's
    # attributes (a write/delete disagreement silently breaks deletion
    # in a real browser), and clearing must restore the ask-again
    # contract: the full OK -> Expert -> Simple sequence re-asks.
    out = run_js(
        "(function () {"
        "  writeViewCookie();"
        "  var afterWrite = { value: readCookie('model-compare-view'), header: document.cookie };"
        "  clearViewCookie();"
        "  var afterClear = { value: readCookie('model-compare-view'), asks: shouldAskForConsent(),"
        "    header: document.cookie };"
        "  return { afterWrite: afterWrite, afterClear: afterClear };"
        "})()"
    )
    assert out["afterWrite"]["value"] == "simple"
    assert "model-compare-view=simple" in out["afterWrite"]["header"]
    assert "max-age=31536000" in out["afterWrite"]["header"]
    assert "SameSite=Lax" in out["afterWrite"]["header"]
    assert "path=/" in out["afterWrite"]["header"]
    assert out["afterClear"]["value"] is None
    assert out["afterClear"]["asks"] is True
    # The delete must agree with the write on path/attributes (a
    # disagreement silently breaks deletion in a real browser).
    assert "max-age=0" in out["afterClear"]["header"]
    assert "SameSite=Lax" in out["afterClear"]["header"]
    assert "path=/" in out["afterClear"]["header"]


def test_consent_wiring_ok_writes_cookie_only():
    # Click-level consent contract (the helpers alone do not prove the
    # wiring): firing the real buttons proves which handler is bound to
    # which element. Simple opens the dialog and stores nothing; only
    # the dialog's OK writes the cookie; Expert removes it; Reject
    # stores nothing and silences the dialog for this page load; a
    # stored-then-removed consent re-asks.
    out = run_js(
        "(function () {"
        "  var dlg = document.getElementById('cookie-dialog');"
        "  var simple = document.getElementById('view-simple');"
        "  var expert = document.getElementById('view-expert');"
        "  var ok = document.getElementById('cookie-ok');"
        "  var reject = document.getElementById('cookie-reject');"
        "  var state = function () {"
        "    return { dialogHidden: dlg.hidden,"
        "             cookie: readCookie('model-compare-view') };"
        "  };"
        "  expert.dispatchEvent('click');"
        "  var afterExpertFirst = state();"
        "  simple.dispatchEvent('click');"
        "  var afterSimple = state();"
        "  ok.dispatchEvent('click');"
        "  var afterOk = state();"
        "  var okHeader = document.cookie;"
        "  expert.dispatchEvent('click');"
        "  var afterExpert = state();"
        "  simple.dispatchEvent('click');"
        "  var reAsk = state();"
        "  reject.dispatchEvent('click');"
        "  var afterReject = state();"
        "  simple.dispatchEvent('click');"
        "  var afterRejectSimple = state();"
        "  return { afterExpertFirst: afterExpertFirst, afterSimple: afterSimple,"
        "    afterOk: afterOk, okHeader: okHeader, afterExpert: afterExpert,"
        "    reAsk: reAsk, afterReject: afterReject,"
        "    afterRejectSimple: afterRejectSimple };"
        "})()"
    )
    # Expert-first never opens the dialog (nothing to consent to).
    assert out["afterExpertFirst"] == {"dialogHidden": True, "cookie": None}
    # Simple asks; it must NOT write the cookie itself.
    assert out["afterSimple"] == {"dialogHidden": False, "cookie": None}
    # Only the dialog's OK stores consent and closes the dialog.
    assert out["afterOk"]["dialogHidden"] is True
    assert out["afterOk"]["cookie"] == "simple"
    assert "model-compare-view=simple" in out["okHeader"]
    # Expert removes the cookie (and never re-opens the dialog).
    assert out["afterExpert"] == {"dialogHidden": True, "cookie": None}
    # OK -> Expert -> Simple asks again: no consent decision is stored.
    assert out["reAsk"] == {"dialogHidden": False, "cookie": None}
    # Reject stores nothing but silences the dialog for this page load.
    assert out["afterReject"] == {"dialogHidden": True, "cookie": None}
    assert out["afterRejectSimple"] == {"dialogHidden": True, "cookie": None}


def test_view_cookie_scoped_to_page_directory():
    # canonical.github.io hosts every project page on one origin, so the
    # preference cookie must be scoped to this page's directory and
    # carry a project-specific name: a sibling project's "view" cookie
    # (or a leaked path=/ cookie) must not reach readCookie.
    out = run_js(
        "(function () {"
        "  var real = window.location.href;"
        "  window.location.href ="
        "    'https://canonical.github.io/model-compare/index.html';"
        "  writeViewCookie();"
        "  var header = document.cookie;"
        "  var value = readCookie('model-compare-view');"
        "  clearViewCookie();"
        "  window.location.href = real;"
        "  return { header: header, value: value };"
        "})()"
    )
    assert out["value"] == "simple"
    assert "model-compare-view=simple" in out["header"]
    assert "path=/model-compare/" in out["header"]


def test_view_cookie_secure_only_in_secure_context():
    # Secure keeps the preference off plaintext transport on the https
    # deploy; it is added only where the browser reports a secure
    # context (localhost previews count; a plain-http LAN preview does
    # not, and the cookie must keep working there).
    out = run_js(
        "(function () {"
        "  writeViewCookie();"
        "  var plain = document.cookie;"
        "  clearViewCookie();"
        "  window.isSecureContext = true;"
        "  writeViewCookie();"
        "  var secure = document.cookie;"
        "  clearViewCookie();"
        "  var secureClear = document.cookie;"
        "  delete window.isSecureContext;"
        "  return { plain: plain, secure: secure, secureClear: secureClear };"
        "})()"
    )
    assert "model-compare-view=simple" in out["plain"]
    assert "Secure" not in out["plain"]
    assert "model-compare-view=simple" in out["secure"]
    assert "; Secure" in out["secure"]
    # The secure write and its delete still agree on the other
    # attributes, so deletion works there too.
    assert "SameSite=Lax" in out["secure"]
    assert "max-age=0" in out["secureClear"]
    assert "SameSite=Lax" in out["secureClear"]
    assert "; Secure" in out["secureClear"]


def test_cookie_dialog_focus_moves_on_show_and_back():
    # Keyboard/AT contract: showing the dialog moves focus into it
    # (it carries tabindex="-1"); OK, Reject and Escape all return
    # focus to the toggle that opened it.
    out = run_js(
        "(function () {"
        "  var dlg = document.getElementById('cookie-dialog');"
        "  var simple = document.getElementById('view-simple');"
        "  simple.dispatchEvent('click');"
        "  var onShow = document.activeElement === dlg;"
        "  document.getElementById('cookie-ok').dispatchEvent('click');"
        "  var onOk = document.activeElement === simple;"
        "  simple.dispatchEvent('click');"
        "  document.getElementById('cookie-reject').dispatchEvent('click');"
        "  var onReject = document.activeElement === simple;"
        "  simple.dispatchEvent('click');"
        "  dlg.dispatchEvent('keydown', { key: 'Escape' });"
        "  var onEscape = document.activeElement === simple;"
        "  return { onShow: onShow, onOk: onOk, onReject: onReject,"
        "    onEscape: onEscape };"
        "})()"
    )
    assert out == {"onShow": True, "onOk": True, "onReject": True, "onEscape": True}


def test_cookie_dialog_escape_dismisses_without_storing():
    # Escape is a dismissal, not a consent: nothing is stored and the
    # dialog is not re-opened by the suppression this page load.
    out = run_js(
        "(function () {"
        "  var dlg = document.getElementById('cookie-dialog');"
        "  var simple = document.getElementById('view-simple');"
        "  simple.dispatchEvent('click');"
        "  dlg.dispatchEvent('keydown', { key: 'Escape' });"
        "  var afterEscape = {"
        "    hidden: dlg.hidden,"
        "    cookie: readCookie('model-compare-view'),"
        "  };"
        "  simple.dispatchEvent('click');"
        "  var afterResimple = { hidden: dlg.hidden };"
        "  return { afterEscape: afterEscape, afterResimple: afterResimple };"
        "})()"
    )
    assert out["afterEscape"] == {"hidden": True, "cookie": None}
    assert out["afterResimple"] == {"hidden": True}


def test_should_ask_for_consent_reask_after_revoke():
    # The full OK -> Expert -> Simple sequence via the real cookie
    # helpers: consent written, then removed, then Simple clicked again
    # -- no consent is stored any more, so the dialog must ask again
    # (the once-per-page-load bug the mockup review caught). Reject, by
    # contrast, keeps suppressing until the next page load.
    assert (
        run_js(
            "(function () {"
            "  writeViewCookie();"
            "  clearViewCookie();"
            "  return shouldAskForConsent();"
            "})()"
        )
        is True
    )
    assert (
        run_js(
            "(function () {"
            "  writeViewCookie();"
            "  clearViewCookie();"
            "  consentRejected = true;"
            "  return shouldAskForConsent();"
            "})()"
        )
        is False
    )


def test_read_cookie_parses_header_shaped_string():
    out = run_js(
        '(function () { document.cookie = "a=1; model-compare-view=simple; b=2";'
        ' return { view: readCookie("model-compare-view"), missing: readCookie("nope") };'
        "})()"
    )
    assert out == {"view": "simple", "missing": None}
    # An empty cookie value carries no consent (and matches what a real
    # browser reports after the max-age=0 deletion): read as absent.
    out = run_js(
        '(function () { document.cookie = "model-compare-view=; max-age=0";'
        ' return readCookie("model-compare-view"); })()'
    )
    assert out is None


def test_endpoint_row_artifact_vs_model_level_text():
    # Artifact-level failure (catalog.json missing/failed) vs model-level
    # join miss: two distinct sentences, keyed on the catalogLoaded flag.
    artifact = run_js(
        "(function () {"
        "  var tr = endpointRow(null, false);"
        "  return { text: tr.children[0].textContent };"
        "})()"
    )
    assert artifact["text"] == ("endpoint details unavailable — catalog fetch failed")
    model = run_js(
        "(function () {"
        "  var tr = endpointRow(null, true);"
        "  return { text: tr.children[0].textContent };"
        "})()"
    )
    assert model["text"] == "No endpoint data for this model."


# ---------------------------------------------------------------------------
# Footer: the data.json stamps and source-revision link (renderFooter)


def test_render_footer_links_source_revision():
    out = run_js(
        "(function () {"
        # Start from the markup's initial hidden state: the assertion
        # must prove renderFooter UNHID the block, not that the stub
        # default already matched.
        "  document.getElementById('rev-block').hidden = true;"
        "  renderFooter({generated_at: '2026-09-28T06:00:00Z',"
        "                source_revision: 'abc1234'});"
        "  return {"
        "    text: document.getElementById('rev-link').textContent,"
        "    href: document.getElementById('rev-link').href,"
        "    hidden: document.getElementById('rev-block').hidden,"
        "  };"
        "})()"
    )
    assert out == {
        "text": "abc1234",
        "href": "https://github.com/canonical/model-compare/commit/abc1234",
        "hidden": False,
    }


def test_render_footer_hides_revision_block_without_source_revision():
    out = run_js(
        "(function () {"
        # The markup ships with rev-block hidden (the hidden attribute);
        # a document without source_revision must leave it hidden and
        # the link untouched: no dead href, no placeholder text swap.
        "  var block = document.getElementById('rev-block');"
        "  var link = document.getElementById('rev-link');"
        "  block.hidden = true;"
        "  renderFooter({generated_at: '2026-09-28T06:00:00Z'});"
        "  return {hidden: block.hidden, text: link.textContent,"
        "          hasHref: 'href' in link};"
        "})()"
    )
    assert out == {"hidden": True, "text": "", "hasHref": False}


def test_render_footer_stamps_generated_time():
    out = run_js(
        "(function () {"
        "  renderFooter({generated_at: '2026-09-28T06:00:00Z'});"
        "  return {"
        "    generated: document.getElementById('generated').textContent,"
        "    generatedTop: document.getElementById('generated-top').textContent,"
        "  };"
        "})()"
    )
    assert isinstance(out["generated"], str) and out["generated"]
    assert isinstance(out["generatedTop"], str) and out["generatedTop"]
