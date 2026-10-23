"""#8283 — Settings update status line must not repeat the unit word.

The status line previously read "2 releases update(s) available": the count
string injected into the {count} placeholder already carries the unit
("WebUI (...): 2 releases"), so the template must not append another unit +
the redundant "(s)". The banner path already renders "2 releases available";
this makes the Settings line match.

Covers:
  - English: rendered status reads "<parts> available" with no doubled unit.
  - All locales: settings_updates_available no longer embeds a unit word or
    the "(s)" plural marker (the {count} argument supplies the unit).
"""

import json
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).parent.parent


def read(rel):
    return (REPO / rel).read_text(encoding='utf-8')


def extract_js_function(src: str, name: str) -> str:
    match = re.search(rf'(async\s+)?function\s+{re.escape(name)}\b', src)
    assert match, f"{name}() not found"
    open_paren = src.index("(", match.start())
    paren_depth = 1
    idx = open_paren + 1
    while paren_depth > 0 and idx < len(src):
        ch = src[idx]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        idx += 1
    brace = src.index("{", idx)
    depth = 0
    end = None
    for idx in range(brace, len(src)):
        ch = src[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    assert end is not None, f"{name}() body was not balanced"
    return src[match.start():end]


def _locale_update_templates():
    """Return {locale: raw settings_updates_available value} from i18n.js."""
    src = read('static/i18n.js')
    # Split into top-level locale blocks: "  xx: {" / "  'xx-YY': {".
    starts = [(m.start(), m.group(1)) for m in
              re.finditer(r"^\s{2}'?([A-Za-z][A-Za-z0-9_-]*)'?\s*:\s*\{", src, re.M)]
    out = {}
    for i, (pos, loc) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(src)
        block = src[pos:end]
        m = re.search(r"settings_updates_available:\s*'((?:[^'\\]|\\.)*)'", block)
        if m:
            out[loc] = m.group(1)
    return out


def test_english_status_line_has_no_doubled_unit():
    """Rendered Settings status must read '<parts> available', no 'update(s)'."""
    ui_src = read('static/ui.js')
    panels_src = read('static/panels.js')
    i18n_src = read('static/i18n.js')

    # Real English template value, straight from i18n.js.
    m = re.search(r"^\s{2}en\s*:\s*\{", i18n_src, re.M)
    assert m, "en locale block not found"
    en_block = i18n_src[m.start():]
    tv = re.search(r"settings_updates_available:\s*'((?:[^'\\]|\\.)*)'", en_block)
    assert tv, "settings_updates_available not found in en"
    updates_available = tv.group(1)

    format_fn = extract_js_function(ui_src, '_formatUpdateTargetStatus')
    instruction_fn = extract_js_function(ui_src, '_formatManualUpdateInstruction')
    error_fn = extract_js_function(ui_src, '_formatUpdateCheckError')
    check_fn = extract_js_function(panels_src, 'checkUpdatesNow')

    script = f"""
const state = {{
  btnCheckUpdatesNow: {{ disabled: false }},
  checkUpdatesLabel: {{ textContent: '' }},
  checkUpdatesSpinner: {{ style: {{ display: 'none' }} }},
  checkUpdatesStatus: {{ textContent: '', style: {{ color: '' }} }},
}};
const apiData = {{
  webui: {{ behind: 2, release_based: true, current_version: 'v0.52.498', latest_version: 'v0.52.499' }},
  agent: null,
}};
function $(id) {{ return state[id] || null; }}
const _VALUES = {json.dumps({'settings_updates_available': updates_available})};
function t(key, ...args) {{
  const values = Object.assign({{ settings_up_to_date: 'Up to date' }}, _VALUES);
  return (values[key] || key).replace(/\\{{(\\d+)\\}}/g, (_, i) => args[Number(i)] ?? '');
}}
async function api() {{ return apiData; }}
function _showUpdateBanner() {{}}
{format_fn}
{instruction_fn}
{error_fn}
{check_fn}
(async () => {{
  await checkUpdatesNow();
  const txt = state.checkUpdatesStatus.textContent;
  if (txt !== 'WebUI (v0.52.498 -> v0.52.499): 2 releases available') {{
    throw new Error('expected clean wording, got: ' + txt);
  }}
  if (/update/i.test(txt)) {{
    throw new Error('status must not repeat the unit word: ' + txt);
  }}
}})().catch(err => {{ console.error(err.message); process.exit(1); }});
"""
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


def test_no_locale_embeds_redundant_unit_word():
    """Every locale template must rely on {count} for the unit.

    A value that still contains 'update' (en/ko) or the '(s)' plural marker
    would double the unit that {count} already carries.
    """
    templates = _locale_update_templates()
    assert templates, "no settings_updates_available templates found"
    offenders = []
    for loc, val in templates.items():
        if 'update' in val.lower() or '(s)' in val:
            offenders.append((loc, val))
    assert not offenders, (
        "settings_updates_available must not embed a unit word / '(s)': "
        + json.dumps(offenders, ensure_ascii=False)
    )
