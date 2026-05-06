import argparse
import ast
import base64
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
from api_helpers import (
    call_api_with_retry,
    extract_reasoning,
    extract_usage,
    DEFAULT_BASE_URLS,
    normalize_reasoning_level,
)
from playwright.sync_api import sync_playwright


FINAL_RESPONSE_PREFIX = "<<FINAL>>"
FINAL_RESPONSE_TEMPLATE = '{"steps_summary":"<a concise step-by-step visual summary (visible observations + any tool actions) supporting your choice>","selected_option":"<selected_option>","errors":[]}'
FINAL_RESPONSE_EXACT = f"{FINAL_RESPONSE_PREFIX} {FINAL_RESPONSE_TEMPLATE}"
TOOL_REQUEST_PREFIX = "<<TOOL>>"
TOOL_REQUEST_TEMPLATE = '{"name":"desmos_plot","arguments":{"rationale":"<why this plot>","expressions":[{"latex":"<expr1>"}],"bounds":{"left":-10,"right":10,"bottom":-10,"top":10},"probe_points":[{"x":<x>,"y":<y>}],"label_intersections":"<bool>","label_extrema":"<bool>","label_intercepts":"<bool>","label_zeros":"<bool>"}}'
TOOL_REQUEST_EXACT = f"{TOOL_REQUEST_PREFIX} {TOOL_REQUEST_TEMPLATE}"

POI_CATEGORY_ORDER: Tuple[str, ...] = (
    "intersections",
    "extrema",
    "intercepts",
    "zeros",
)

POI_CATEGORY_POINT_COLOR: Dict[str, str] = {
    "intersections": "#1c7ed6",
    "extrema": "#e03131",
    "intercepts": "#2b8a3e",
    "zeros": "#f08c00",
}
POI_LABEL_ZERO_TOL = 1e-6

QUESTION_RULES_TEXT_BASE = (
    "Rules (follow strictly):\n"
    "1. Base your decision ONLY on visual evidence from the question, options, provided images, and Desmos plot screenshots. If reading a coordinate or feature from a graph, zoom in/out until it's clearly visible.\n"
    "2. Reason step by step from visual evidence only; each step must describe a visible observation, comparison, or direct visual read-off.\n"
    "3. Rely on visible cues such as intersections, axis/grid labels, relative positions, distances, and curve behavior.\n"
    "4. Do NOT solve analytically (no algebra/calculus/proofs/root-finding or heavy computation). Light arithmetic on clearly visible values is OK.\n"
    "5. Basic non-visual reasoning is allowed only to interpret visible evidence (for example, matching visible features to options or doing light arithmetic on visible labels), not to derive facts missing from the images/screenshots.\n"
    "6. Even if the answer looks easy to derive analytically, you must use visual evidence only.\n"
    "7. If the current visual evidence is insufficient and another screenshot is available, request another Desmos visualization instead of solving analytically or guessing.\n"
    "8. Do NOT fabricate missing information or make unsupported assumptions.\n"
    "9. You MUST make at least one successful desmos_plot call (it produces a screenshot) before giving the final answer.\n"
    "10. Each assistant message: either request a plot OR give the final answer (never both).\n"
    f"11. If the current visual evidence is sufficient, output the final answer using the required {FINAL_RESPONSE_PREFIX} JSON format.\n"
    "12. Output must be ONLY the prefix followed by ONE valid JSON object (double-quoted keys/strings; no trailing commas). No extra text/markdown.\n"
    "13. Tool request format (replace <...> placeholders):\n"
    f"{TOOL_REQUEST_EXACT}\n"
    "14. Final answer format (replace <...> placeholders):\n"
    f"{FINAL_RESPONSE_EXACT}\n"
)


def screenshot_budget_reminder_text(max_screenshots_per_question: int) -> str:
    limit = max(0, int(max_screenshots_per_question))
    if limit <= 0:
        return ""
    return (
        f"Screenshot budget: at most {limit} successful desmos_plot screenshot(s) per question. "
        "After the limit is reached, you MUST give the final answer from visible evidence; further tool requests will be rejected."
    )


def question_rules_text(
    max_completion_tokens_per_question: int,
    max_screenshots_per_question: int,
) -> str:
    # budget_text = completion_budget_rule_text(max_completion_tokens_per_question)
    # screenshot_text = screenshot_budget_rule_text(max_screenshots_per_question)
    parts = [QUESTION_RULES_TEXT_BASE]
    # if screenshot_text:
        # parts.append(screenshot_text)
    # if budget_text:
        # parts.append(budget_text)
    return "\n".join(part for part in parts if part)


TOOL_RESULT_PROMPT_BASE = (
    "Here is your requested plot screenshot.\n"
    "Base your next step ONLY on visible evidence from the question, options, provided images, and Desmos screenshots.\n"
    "If the evidence is sufficient, reply ONLY with a final answer in this format:\n"
    f"{FINAL_RESPONSE_EXACT}\n"
    "If the evidence is insufficient/unclear, request another plot (adjust parameters) instead of solving analytically or guessing, in this format:\n"
    f"{TOOL_REQUEST_EXACT}"
)

TOOL_RESULT_PROMPT_FINAL_ONLY = (
    "Here is your requested plot screenshot.\n"
    "You have reached the maximum number of screenshots allowed for this question.\n"
    "Use ONLY the visible evidence already shown; do not solve analytically or guess beyond the visual evidence.\n"
    "If that evidence is insufficient, use selected_option \"N/A\" and briefly explain the visual gap in errors.\n"
    "Reply ONLY with a final answer in this format:\n"
    f"{FINAL_RESPONSE_EXACT}"
)


def tool_result_prompt(max_screenshots_per_question: int, screenshots_so_far: int) -> str:
    limit = max(0, int(max_screenshots_per_question))
    used = max(0, int(screenshots_so_far))
    if limit > 0 and used >= limit:
        return TOOL_RESULT_PROMPT_FINAL_ONLY
    if limit > 0:
        remaining = max(0, limit - used)
        return f"{TOOL_RESULT_PROMPT_BASE}\n\nScreenshot budget: {remaining} remaining."
    return TOOL_RESULT_PROMPT_BASE


def final_nudge_text(
    max_completion_tokens_per_question: int,
    max_screenshots_per_question: int,
    screenshots_so_far: int,
) -> str:
    budget_text = completion_budget_rule_text(max_completion_tokens_per_question)
    screenshot_text = screenshot_budget_reminder_text(max_screenshots_per_question)
    budget_clause = f"\n{budget_text}" if budget_text else ""
    screenshot_clause = f"\n{screenshot_text}" if screenshot_text else ""

    limit = max(0, int(max_screenshots_per_question))
    used = max(0, int(screenshots_so_far))
    tool_allowed = not (limit > 0 and used >= limit)
    tool_clause = (
        "Tool request format:\n" + f"{TOOL_REQUEST_EXACT}\n"
        if tool_allowed
        else "(Tool requests are no longer allowed for this question; reply with the final answer only.)\n"
    )

    return (
        f"Reminder: Output ONLY: the prefix ({TOOL_REQUEST_PREFIX} or {FINAL_RESPONSE_PREFIX}) followed by one JSON object (no extra text). "
        "You MUST have at least one successful desmos_plot screenshot before the final answer. "
        "Justify using ONLY visible evidence from the question, options, images, and Desmos screenshots; do not solve analytically or guess.\n"
        f"{tool_clause}"
        "Final answer format:\n"
        f"{FINAL_RESPONSE_EXACT}"
        f"{screenshot_clause}"
        f"{budget_clause}"
    )

def format_tool_error_message(
    error_code: str,
    detail: str,
    fix: Optional[str] = None,
    include_schema_hint: bool = False,
) -> str:
    parts = [f"Tool ERROR [code={error_code}]: {str(detail or '').strip()}"]
    if fix:
        parts.append(f"Fix: {str(fix).strip()}")
    if include_schema_hint:
        parts.append(
            f"Re-send exactly one tool request in this exact format (no extra text/markdown): {TOOL_REQUEST_EXACT}"
        )
    return " ".join(part for part in parts if part)


def format_tool_repair_text(error_code: Optional[str], error_detail: Optional[str]) -> str:
    guidance_by_code = {
        "empty_tool_payload": (
            "Add exactly one JSON object immediately after the tool prefix."
        ),
        "invalid_json": (
            "Return strictly valid JSON (double-quoted keys/strings, no trailing commas, no extra markers)."
        ),
        "tool_payload_not_object": (
            "The payload must be a JSON object with top-level keys 'name' and 'arguments'."
        ),
        "missing_tool_name": (
            "Add the top-level 'name' field and set it to 'desmos_plot'."
        ),
        "missing_or_invalid_arguments": (
            "Set 'arguments' to a JSON object with rationale, expressions, bounds, optional probe_points, and optional POI label booleans."
        ),
        "missing_rationale": (
            "Add a non-empty 'rationale' string inside 'arguments'."
        ),
    }
    normalized_code = str(error_code or "invalid_tool_request").strip() or "invalid_tool_request"
    detail_text = str(error_detail or "Malformed tool request payload.").strip()
    guidance = guidance_by_code.get(str(error_code or "").strip())
    return format_tool_error_message(
        error_code=normalized_code,
        detail=detail_text,
        fix=guidance,
        include_schema_hint=True,
    )

SYSTEM_PROMPT_BASE = """\
Developer: You are a helpful assistant specializing in visual problem-solving for multiple-choice math questions.
Use the question text to decide what to plot, but base your justification ONLY on visible evidence from images and Desmos screenshots. Options may be provided as text or images. 
You have exactly one allowed tool: desmos_plot (a graphing calculator). You MUST obtain at least one successful desmos_plot screenshot before giving the final answer. Do NOT use Python, code execution, any calculator outside desmos_plot, web search, or any other external tool.

Reasoning constraints:
- Reason step by step from visual evidence only from the provided images and Desmos screenshots.
- Each reasoning step must describe a visible observation, comparison, or direct visual read-off from the supplied visualizations.
- Do NOT solve analytically (no algebra/calculus/proofs/root-finding). For instance, you may define f(x) and plot f'(x) or f'(x)=0 in Desmos, but use those only as visual evidence, not as an analytic calculus derivation.
- Light arithmetic on clearly visible values is allowed.
- Basic non-visual reasoning is allowed only to interpret visible evidence, such as matching plotted features to options, comparing positions, or doing light arithmetic on clearly visible labels. Do not use it to derive facts that are not visible in the images/screenshots.
- Your answer MUST rely ONLY on the provided images, Desmos plot screenshots, and the question/options.
- Rely on visible cues such as intersections, axis labels, grid labels, relative positions, distances, and curve behavior to pick the option.
- Do NOT verify, approximate, or complete the solution using algebra, symbolic manipulation, calculus derivations, root-finding, proofs, or heavy computation when the current image is insufficient.
- Do NOT fabricate missing information or make unsupported assumptions.
- When reading off a plot, zoom in/out until the relevant features, grid labels, and curve behavior are clearly visible before deciding.
- Do not guess when evidence is missing; Do not output "selected_option":"N/A" while more Desmos screenshots may still be available; request another plot instead.
- If no successful Desmos screenshot is available yet, request a plot instead of finalizing.

Output rule:
- In each assistant message, output ONLY: the prefix followed by exactly one valid JSON object. No extra text or markdown.
(A) Tool request format (replace <...> placeholders):
{TOOL_REQUEST_EXACT}
(B) Final answer format (replace <...> placeholders):
{FINAL_RESPONSE_EXACT}

Tool request notes:
- arguments.rationale must be a non-empty string.
- Provide a non-empty expressions list (one latex per object) and bounds; probe_points optional.
- Use probe_points to click/label approximate coordinates; zoom in/out by tightening/loosening bounds while maintaining the other parameters.
- Optional coordinate labels are available for POIs (gray points on the screenshots); set any of these booleans in arguments to true (only when needed to avoid clutter): label_intersections, label_extrema, label_intercepts, label_zeros.
- IMPORTANT JSON escaping: inside the JSON string `arguments.expressions[].latex`, every LaTeX backslash MUST be JSON-escaped as `\\\\` (double backslash). For example: use `\\\\frac{1}{2}x`, `\\\\cot(x)`, and `\\\\left|x\\\\right|` (not `\\frac`, not `\\cot`, not `\\left`).
- desmos_plot expects Desmos-friendly LaTeX-ish expressions. You may find these JSON-escaped examples useful:
    - \\\\left|x\\\\right| (absolute value function)
    - floor(x) (floor function)
    - \\\\sqrt[3]{x^5} or x^{5/3} (rational exponent)
    - \\\\frac{1}{\\\\left(x-3 \\\\right)^{4.5}} (fractional expression)
    - (x+1)^2\\\\left\\\\{x>-1\\\\right\\\\} (domain-restricted formula)
    - x=y^3 (function-style expression, can be useful for inverse functions)
    - x^2+y^2=2^2 \\\\left\\\\{0<=y<=2\\\\right\\\\}\\\\left\\\\{0<=x<=2\\\\right\\\\} (relation-style expression)
    - x^2=x+2 (to find solutions to equations rather than solving analytically)
    - f(x)=x^7-\\\\sin(x)*\\\\cos(x), then f'(x) or f'(x)=0 (define a function and plot its derivative or visually locate critical points/solutions)
    - x^4>x^2+3 (inequality shading)
    - 2*\\\\log_2 (10) (a non-graphable numeric expression is fine, can be used to calculate and probe specific values)
- Define free parameters (a, m, k) before plotting like a=-3.

Final answer notes:
- Output exactly these JSON keys and no others: steps_summary, selected_option, errors.
- steps_summary must briefly describe the visual reasoning and, if you used desmos_plot, what you plotted; keep it visual, not algebraic.
- selected_option should be "1", "2", "3", "4", or "N/A".
- Use "N/A" only if there is insufficient visual evidence to answer AND no further Desmos screenshot will be provided; explain why briefly in errors.
"""

def system_prompt_text(
    max_completion_tokens_per_question: int,
    max_screenshots_per_question: int,
) -> str:
    budget_text = completion_budget_rule_text(max_completion_tokens_per_question)
    screenshot_text = screenshot_budget_reminder_text(max_screenshots_per_question)
    base_prompt = SYSTEM_PROMPT
    if not budget_text and not screenshot_text:
        return base_prompt
    tail_parts: List[str] = []
    if screenshot_text:
        tail_parts.append(screenshot_text)
    if budget_text:
        tail_parts.append(f"Token budget reminder: {budget_text}")
    return f"{base_prompt}\n" + "\n".join(tail_parts)


def completion_budget_rule_text(max_completion_tokens_per_question: int) -> str:
    budget = max(0, int(max_completion_tokens_per_question))
    if budget <= 0:
        return ""
    return (
        f"This question has a total completion-token budget of {budget} tokens; "
        "keep answers concise and DO NOT waste tokens."
    )

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.replace("{FINAL_RESPONSE_PREFIX}", FINAL_RESPONSE_PREFIX)
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{FINAL_RESPONSE_EXACT}", FINAL_RESPONSE_EXACT)
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{TOOL_REQUEST_PREFIX}", TOOL_REQUEST_PREFIX)
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{TOOL_REQUEST_EXACT}", TOOL_REQUEST_EXACT)

def normalize_desmos_latex(expr: str) -> str:
    if not isinstance(expr, str):
        return expr
    text = expr.strip().strip("`")
    # Models sometimes emit LaTeX in JSON strings where JSON interprets sequences like:
    #   \f (form-feed), \r (carriage-return), \t (tab), \b (backspace), \n (newline)
    # as control characters. When that happens, the resulting LaTeX macro becomes corrupted.
    # Undo those control-character escapes back into their intended backslash-letter.
    text = (
        text.replace("\x0c", "\\f")  # \f  -> \f
        .replace("\x0d", "\\r")     # \r  -> \r
        .replace("\x09", "\\t")     # \t  -> \t
        .replace("\x08", "\\b")     # \b  -> \b
        .replace("\x0a", "\\n")     # \n  -> \n
    )
    text = re.sub(r"\s*=\s*", "=", text)

    # Normalize domain/piecewise restrictions to Desmos literal braces: \{...\}.
    # This preserves semantics for inputs like {x>0} and over-escaped \\{x>0\\}.
    text = re.sub(r"\\{2,}\{", r"\\{", text)
    text = re.sub(r"\\{2,}\}", r"\\}", text)
    # Only treat braces as domain/piecewise restrictions when they contain a real
    # inequality marker. Guard \le/\ge so \left and similar commands are not
    # mistaken for \le.
    text = re.sub(
        r"(?<!\\)\{([^{}]*(?:<=|>=|<|>|\\leq?(?![A-Za-z])|\\geq?(?![A-Za-z])|:)[^{}]*)\}",
        r"\\{\1\\}",
        text,
    )

    def replace_func_calls(raw: str, func_name: str, make_replacement: Any) -> str:
        cursor = 0
        chunks: List[str] = []
        while True:
            match = re.search(rf"(?i)\b{re.escape(func_name)}\(", raw[cursor:])
            if not match:
                chunks.append(raw[cursor:])
                break
            start = cursor + match.start()
            chunks.append(raw[cursor:start])
            index = start + match.end() - match.start()
            depth = 1
            while index < len(raw) and depth > 0:
                char = raw[index]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                index += 1
            if depth != 0:
                chunks.append(raw[start:index])
                cursor = index
                continue
            inner = raw[start + len(func_name) + 1 : index - 1].strip()
            inner = normalize_desmos_latex(inner)
            chunks.append(make_replacement(inner))
            cursor = index
        return "".join(chunks)

    text = text.replace("\\lvert", "|").replace("\\rvert", "|")
    # Convert floor-bracket syntax to Desmos operator form.
    text = text.replace("\\lfloor", "\\operatorname{floor}\\left(").replace("\\rfloor", "\\right)")
    text = replace_func_calls(text, "abs", lambda inner: f"\\left|{inner}\\right|")
    text = replace_func_calls(text, "sqrt", lambda inner: f"\\sqrt{{{inner}}}")
    text = replace_func_calls(
        text,
        "floor",
        lambda inner: f"\\operatorname{{floor}}\\left({inner}\\right)",
    )
    # Collapse accidental double-backslashes before known LaTeX commands.
    # Example: "f(x)=\\\\cot(x)" should become "f(x)=\\cot(x)" for Desmos.
    text = re.sub(
        r"\\\\(left|right|frac|sqrt|sin|cos|tan|cot|sec|csc|ln|log)\b",
        r"\\\1",
        text,
        flags=re.I,
    )
    # Also collapse double-backslashes before literal brace escapes.
    # Models sometimes emit "\\{...\\}" inside JSON strings, which can confuse parsing.
    text = re.sub(r"\\\\([{}])", r"\\\1", text)

    # Wrap inequality/piecewise restrictions with \left\{...\right\}. Desmos's
    # evaluator graphs the bare \{...\} form correctly, but its MathQuill input
    # renderer leaves the entry blank in the expression panel. The \left/\right
    # form renders properly *and* graphs identically. The negative lookbehind
    # avoids double-wrapping inputs that already use \left\{...\right\}.
    text = re.sub(
        r"(?<!\\left)\\\{([^{}]*(?:<=|>=|<|>|\\leq?(?![A-Za-z])|\\geq?(?![A-Za-z])|:)[^{}]*)\\\}",
        r"\\left\\{\1\\right\\}",
        text,
    )
    text = re.sub(r"(?i)(?<!\\)\barctan\s*\(", r"\\arctan(", text)
    text = re.sub(r"(?i)(?<!\\)\barcsin\s*\(", r"\\arcsin(", text)
    text = re.sub(r"(?i)(?<!\\)\barccos\s*\(", r"\\arccos(", text)
    text = re.sub(r"(?i)(?<!\\)\bcot\s*\(", r"\\cot(", text)
    text = re.sub(
        r"(?<!\\)\b(sin|cos|tan|sec|csc|cot|ln|log)\s*\(",
        r"\\\1(",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?<!\\)(?<=[0-9\)])(sin|cos|tan|sec|csc|cot|ln|log)\s*\(",
        r"\\\1(",
        text,
        flags=re.I,
    )

    rebuilt: List[str] = []
    bar_depth = 0
    idx = 0
    opening_context = set("=([,{:+-*/^")
    while idx < len(text):
        if text.startswith("\\left|", idx):
            rebuilt.append("\\left|")
            idx += len("\\left|")
            continue
        if text.startswith("\\right|", idx):
            rebuilt.append("\\right|")
            idx += len("\\right|")
            continue
        char = text[idx]
        if char == "|":
            prev = None
            j = idx - 1
            while j >= 0:
                if not text[j].isspace():
                    prev = text[j]
                    break
                j -= 1
            should_open = bar_depth == 0 or (prev in opening_context)
            if should_open:
                rebuilt.append("\\left|")
                bar_depth += 1
            else:
                rebuilt.append("\\right|")
                bar_depth = max(0, bar_depth - 1)
        else:
            rebuilt.append(char)
        idx += 1
    text = "".join(rebuilt)
    text = re.sub(r"(?<![A-Za-z\\])pi(?![A-Za-z])", r"\\pi", text)
    # Defensive rewrite of scientific notation so Desmos cannot reinterpret the
    # ``e`` as Euler's constant. Matches numeric literals like ``4.441e-16`` or
    # ``1.5E+3`` only when the ``e`` sits between a digit/dot and a signed
    # integer, and is not part of a longer identifier.
    text = re.sub(
        r"(?<![A-Za-z\\])(\d+(?:\.\d+)?|\.\d+)[eE]([+-]?\d+)(?![A-Za-z0-9])",
        r"\1\\cdot10^{\2}",
        text,
    )
    return text


def format_desmos_compact_g(
    value: Any,
    *,
    precision: int = 4,
    zero_tol: Optional[float] = None,
) -> str:
    """Format like ``:.4g`` for point-coordinate labels embedded in LaTeX.

    Desmos treats bare ``e`` as Euler's constant, so outputs such as ``8.882e-16``
    from ``g``-format break point placement. Here we round with ``precision`` in
    the same way as Python's ``g`` format, then rewrite scientific notation into
    ``mantissa\\cdot10^{exp}``.
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "0"
    if not math.isfinite(numeric):
        return "0"
    if zero_tol is not None:
        tol = abs(float(zero_tol))
        if abs(numeric) < tol:
            numeric = 0.0

    text = format(numeric, f".{int(precision)}g")
    text_strip = text.strip()
    match = re.fullmatch(r"([+-]?(?:\d+\.?\d*|\.\d+))[eE]([+-]?\d+)", text_strip)
    if match:
        mantissa, exp_str = match.group(1), match.group(2)
        exponent_value = int(exp_str)
        return f"{mantissa}\\cdot10^{{{exponent_value}}}"
    return text_strip


def is_nonblank_png(png_path: Path) -> bool:
    try:
        from PIL import Image, ImageStat

        image = Image.open(png_path).convert("L")
        stats = ImageStat.Stat(image)
        mean = stats.mean[0]
        stddev = stats.stddev[0]
        return not (mean > 250 and stddev < 2.0)
    except Exception:
        try:
            return png_path.stat().st_size > 20_000
        except OSError:
            return True


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def relpath_for_output(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), Path.cwd()).replace(os.sep, "/")


def sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "item"


DESMOS_HTML_TEMPLATE = """\
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Desmos Agent</title>
    <style>
      html, body {{ margin: 0; padding: 0; background: white; }}
      #calculator {{ width: {width}px; height: {height}px; }}
    </style>
    <script src="https://www.desmos.com/api/v1.11/calculator.js?apiKey={desmos_api_key}"></script>
  </head>
  <body>
    <div id="calculator"></div>
    <script>
      const elt = document.getElementById("calculator");
            window.calc = Desmos.GraphingCalculator(elt, {{
                expressions: true,
                expressionsCollapsed: false,
                expressionsTopbar: false,
                capExpressionSize: true,
                border: false,
                settingsMenu: false,
                zoomButtons: false,
                keypad: false,
                projectorMode: true,
                pointsOfInterest: true,
                images: false,
                folders: false,
                notes: false,
                sliders: false,
                actions: false,
                substitutions: false,
                links: false,
                distributions: false,
                xAxisArrowMode: Desmos.AxisArrowModes.BOTH,
                xAxisLabel: "x",
                yAxisArrowMode: Desmos.AxisArrowModes.BOTH, 
                yAxisLabel: "y",
                randomSeed: "42",
                //invertedColors: true,
                audio: false,
                tone: false,
                intervalComprehensions: false,
                trace: true
            }});
    </script>
  </body>
</html>
"""


def render_desmos_plot(
    desmos_api_key: str,
    expressions: List[str],
    bounds: Dict[str, float],
    out_png: Path,
    probe_points: Optional[List[Dict[str, float]]] = None,
    poi_display: Optional[Dict[str, bool]] = None,
    width: int = 1280,
    height: int = 720,
    save_desmos_internals: bool = False,
    max_poi_overlay_points: Optional[int] = None,
) -> Dict[str, Any]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_png.parent / ".desmos_agent.html"
    html_path.write_text(
        DESMOS_HTML_TEMPLATE.format(
            width=width,
            height=height,
            desmos_api_key=desmos_api_key,
        ),
        encoding="utf-8",
    )

    rendered_expressions: List[Dict[str, Any]] = []
    for index, latex in enumerate(expressions):
        expr: Dict[str, Any] = {
            "id": f"expr_{index}",
            "latex": latex,
            "hidden": False,
            "showPOI": True,
            "showHighlight": True,
        }
        rendered_expressions.append(expr)

    probe_expressions: List[Dict[str, Any]] = []
    for probe_index, point in enumerate(probe_points or []):
        x = point.get("x")
        y = point.get("y")
        if x is None or y is None:
            continue
        x_val = float(x)
        y_val = float(y)
        probe_expressions.append(
            {
                "id": f"probe_{probe_index}",
                "latex": (
                    f"({format_desmos_compact_g(x_val)}, "
                    f"{format_desmos_compact_g(y_val)})"
                ),
                "label": "",
                "showLabel": True,
            }
        )

    if probe_expressions:
        rendered_expressions.extend(probe_expressions)

    poi_display_options = {
        category: bool((poi_display or {}).get(category))
        for category in POI_CATEGORY_ORDER
    }
    needs_poi_overlay = poi_display_requested(poi_display_options)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.wait_for_function(
            "() => window.calc && typeof window.calc.setExpressions === 'function'"
        )
        page.evaluate(
            """({exprs, bounds}) => {
                window.calc.setBlank();
                window.calc.setExpressions(exprs);
                window.calc.setMathBounds(bounds);
            }""",
            {"exprs": rendered_expressions, "bounds": bounds},
        )
        time.sleep(0.8)
        expression_analysis = page.evaluate(
            """(exprs) => {
                const analysis = window.calc.expressionAnalysis || {};
                return exprs.map((expr) => {
                    const a = analysis[expr.id] || null;
                    return {
                        id: expr.id,
                        latex: expr.latex,
                        is_graphable: a ? Boolean(a.isGraphable) : null,
                        has_error: a ? Boolean(a.isError) : null,
                        error_message: a ? (a.errorMessage || null) : null,
                    };
                });
            }""",
            rendered_expressions,
        )
        clicked_probe_points: List[Dict[str, float]] = []
        for point in probe_points or []:
            x = point.get("x")
            y = point.get("y")
            if x is None or y is None:
                continue
            pixel_point = page.evaluate(
                """(point) => {
                    if (!window.calc || typeof window.calc.mathToPixels !== 'function') return null;
                    const pixel = window.calc.mathToPixels({ x: point.x, y: point.y });
                    if (!pixel || !isFinite(pixel.x) || !isFinite(pixel.y)) return null;
                    return { x: pixel.x, y: pixel.y };
                }""",
                {"x": float(x), "y": float(y)},
            )
            if not isinstance(pixel_point, dict):
                continue
            px = pixel_point.get("x")
            py = pixel_point.get("y")
            if px is None or py is None:
                continue
            page.mouse.move(float(px), float(py))
            page.mouse.click(float(px), float(py))
            clicked_probe_points.append({"x": float(x), "y": float(y)})
            time.sleep(0.25)

        screenshot_captured = False
        if not needs_poi_overlay:
            page.locator("#calculator").screenshot(path=str(out_png.resolve()))
            screenshot_captured = True

        collect_poi = bool(needs_poi_overlay) or bool(save_desmos_internals)

        expression_click_debug: Dict[str, Any] = {
            "attempted": False,
            "target_count": 0,
            "available_items": 0,
            "forward_clicks": 0,
            "reverse_clicks": 0,
        }
        plot_expr_count = sum(
            1
            for expr in rendered_expressions
            if str(expr.get("id") or "").startswith("expr_")
        )
        if collect_poi and plot_expr_count > 0:
            expression_click_debug["attempted"] = True
            expression_items = page.locator(".dcg-expressionitem")
            try:
                page.wait_for_selector(".dcg-expressionitem", timeout=1500)
            except Exception:
                pass

            try:
                available_items = int(expression_items.count())
            except Exception:
                available_items = 0
            expression_click_debug["available_items"] = available_items

            target_count = min(plot_expr_count, available_items)
            expression_click_debug["target_count"] = target_count

            for index in range(target_count):
                item = expression_items.nth(index)
                try:
                    item.scroll_into_view_if_needed(timeout=750)
                except Exception:
                    pass
                try:
                    item.click(timeout=1000, force=True)
                    expression_click_debug["forward_clicks"] += 1
                    time.sleep(0.6)
                except Exception:
                    continue

            for index in range(target_count - 1, -1, -1):
                item = expression_items.nth(index)
                try:
                    item.scroll_into_view_if_needed(timeout=750)
                except Exception:
                    pass
                try:
                    item.click(timeout=1000, force=True)
                    expression_click_debug["reverse_clicks"] += 1
                    time.sleep(0.6)
                except Exception:
                    continue

            if (
                expression_click_debug["forward_clicks"] > 0
                or expression_click_debug["reverse_clicks"] > 0
            ):
                time.sleep(1.0)

        points_of_interest = page.evaluate(
            """async ({exprs, includeDebugInternals, collectPoi}) => {
                if (!collectPoi) return [];
                const calc = window.calc;
                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

                const isFiniteNumber = (value) => (
                    typeof value === "number" && Number.isFinite(value)
                );

                const pickCoordinate = (raw) => {
                    if (!raw || typeof raw !== "object") return null;
                    const candidates = [
                        raw,
                        raw.point,
                        raw.xy,
                        raw.graphPoint,
                        raw.mathPoint,
                        raw.coord,
                        raw.coords,
                    ];
                    for (const candidate of candidates) {
                        if (!candidate || typeof candidate !== "object") continue;
                        const x = candidate.x;
                        const y = candidate.y;
                        if (isFiniteNumber(x) && isFiniteNumber(y)) {
                            return { x, y };
                        }
                    }
                    return null;
                };

                const pickKind = (raw) => {
                    if (!raw || typeof raw !== "object") return null;
                    const values = [
                        raw.kind,
                        raw.label,
                        raw.title,
                        raw.text,
                    ];
                    for (const value of values) {
                        if (typeof value !== "string") continue;
                        const cleaned = value.trim();
                        if (cleaned) return cleaned;
                    }
                    return null;
                };

                const pickPoiType = (raw) => {
                    if (!raw || typeof raw !== "object") return null;
                    const values = [raw.type, raw.poiType];
                    for (const value of values) {
                        const parsed = Number(value);
                        if (Number.isFinite(parsed)) {
                            return parsed;
                        }
                    }
                    return null;
                };

                const normalizeToken = (value) => {
                    if (value === null || value === undefined) return null;
                    const t = typeof value;
                    if (
                        t === "string"
                        || t === "number"
                        || t === "boolean"
                        || t === "bigint"
                    ) {
                        const text = String(value).trim();
                        return text || null;
                    }
                    return null;
                };

                const pickIntersectsWith = (raw) => {
                    if (!raw || typeof raw !== "object") return null;
                    const candidates = [
                        raw.intersects_with,
                        raw.intersectsWith,
                        raw.intersects,
                        raw.otherId,
                        raw.other_id,
                    ];
                    for (const candidate of candidates) {
                        if (Array.isArray(candidate) && candidate.length > 0) {
                            const token = normalizeToken(candidate[0]);
                            if (token) return token;
                            continue;
                        }
                        const token = normalizeToken(candidate);
                        if (token) return token;
                    }
                    return null;
                };

                const toTraceSafeValue = (
                    value,
                    depth = 0,
                    seen = new WeakSet(),
                    maxDepth = 6,
                    maxArrayItems = 200,
                    maxObjectKeys = 200,
                ) => {
                    if (value === null || value === undefined) return null;
                    const valueType = typeof value;

                    if (
                        valueType === "string"
                        || valueType === "number"
                        || valueType === "boolean"
                    ) {
                        return value;
                    }
                    if (valueType === "bigint") return value.toString();
                    if (valueType === "function") return "[Function]";
                    if (valueType === "symbol") return String(value);
                    if (valueType !== "object") return String(value);
                    if (seen.has(value)) return "[Circular]";
                    if (depth >= maxDepth) return "[MaxDepth]";

                    if (value instanceof Date) return value.toISOString();
                    if (value instanceof RegExp) return String(value);

                    seen.add(value);
                    try {
                        if (Array.isArray(value)) {
                            const out = [];
                            const limit = Math.min(value.length, maxArrayItems);
                            for (let index = 0; index < limit; index += 1) {
                                out.push(
                                    toTraceSafeValue(
                                        value[index],
                                        depth + 1,
                                        seen,
                                        maxDepth,
                                        maxArrayItems,
                                        maxObjectKeys,
                                    )
                                );
                            }
                            if (value.length > limit) {
                                out.push(`[Truncated ${value.length - limit} items]`);
                            }
                            return out;
                        }

                        const out = {};
                        const keys = Object.keys(value);
                        const limit = Math.min(keys.length, maxObjectKeys);
                        for (let index = 0; index < limit; index += 1) {
                            const key = keys[index];
                            try {
                                out[key] = toTraceSafeValue(
                                    value[key],
                                    depth + 1,
                                    seen,
                                    maxDepth,
                                    maxArrayItems,
                                    maxObjectKeys,
                                );
                            } catch (error) {
                                out[key] = `[ReadError: ${String(error?.message || error)}]`;
                            }
                        }
                        if (keys.length > limit) {
                            out.__truncated_keys__ = keys.length - limit;
                        }
                        return out;
                    } finally {
                        seen.delete(value);
                    }
                };

                const normalizePOIItems = (items, sourceField, branchIndex) => {
                    if (!Array.isArray(items)) return [];
                    const rows = [];
                    for (const raw of items) {
                        const coordinate = pickCoordinate(raw);
                        if (!coordinate) continue;
                        rows.push({
                            x: coordinate.x,
                            y: coordinate.y,
                            kind: pickKind(raw),
                            // poi_type: pickPoiType(raw),
                            // source_field: sourceField,
                            // branch_index: branchIndex,
                            // intersects_with: pickIntersectsWith(raw),
                        });
                    }
                    return rows;
                };

                const normalizePOIStructured = (poiObject, sourceField, branchIndex) => {
                    if (!poiObject || typeof poiObject !== "object" || Array.isArray(poiObject)) {
                        return [];
                    }

                    const rows = [];

                    const appendXY = (
                        rawSub,
                        kind,
                        poiType,
                        localSourceField,
                        intersectsValues = null,
                    ) => {
                        if (!rawSub || typeof rawSub !== "object") return;
                        const xsRaw = rawSub.x;
                        const ysRaw = rawSub.y;
                        const xs = Array.isArray(xsRaw) ? xsRaw : [xsRaw];
                        const ys = Array.isArray(ysRaw) ? ysRaw : [ysRaw];
                        const length = Math.min(xs.length, ys.length);
                        for (let index = 0; index < length; index += 1) {
                            const x = Number(xs[index]);
                            const y = Number(ys[index]);
                            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

                            let intersectsWith = null;
                            if (Array.isArray(intersectsValues)) {
                                if (index < intersectsValues.length) {
                                    intersectsWith = normalizeToken(intersectsValues[index]);
                                }
                            } else {
                                intersectsWith = normalizeToken(intersectsValues);
                            }

                            rows.push({
                                x,
                                y,
                                kind,
                                poi_type: poiType,
                                source_field: localSourceField,
                                branch_index: branchIndex,
                                intersects_with: intersectsWith,
                            });
                        }
                    };

                    appendXY(poiObject.zeros, "zero", 1002, `${sourceField}.zeros`);
                    appendXY(poiObject.intercept, "intercept", 1003, `${sourceField}.intercept`);
                    appendXY(poiObject.intercepts, "intercept", 1003, `${sourceField}.intercepts`);
                    appendXY(poiObject.extrema, "extremum", 1004, `${sourceField}.extrema`);
                    appendXY(poiObject.extremum, "extremum", 1004, `${sourceField}.extremum`);

                    const intersections = poiObject.intersections || poiObject.intersection;
                    if (intersections && typeof intersections === "object") {
                        appendXY(
                            intersections,
                            "intersection",
                            1001,
                            `${sourceField}.intersections`,
                            intersections.intersects,
                        );
                    }

                    const knownFields = new Set([
                        "zeros",
                        "intercept",
                        "intercepts",
                        "extrema",
                        "extremum",
                        "intersections",
                        "intersection",
                    ]);
                    for (const [key, value] of Object.entries(poiObject)) {
                        if (knownFields.has(key)) continue;
                        if (!value || typeof value !== "object" || Array.isArray(value)) continue;
                        const rawPoiType = pickPoiType(value);
                        const poiType = Number.isFinite(Number(rawPoiType))
                            ? Number(rawPoiType)
                            : null;
                        appendXY(
                            value,
                            String(key),
                            poiType,
                            `${sourceField}.${key}`,
                            value.intersects,
                        );
                    }

                    return rows;
                };

                const normalizePOIMapItems = (itemsMap, sourceField, branchIndex) => {
                    if (!itemsMap || typeof itemsMap !== "object" || Array.isArray(itemsMap)) {
                        return [];
                    }
                    const rows = [];
                    for (const raw of Object.values(itemsMap)) {
                        const coordinate = pickCoordinate(raw);
                        if (!coordinate) continue;
                        rows.push({
                            x: coordinate.x,
                            y: coordinate.y,
                            kind: pickKind(raw),
                            poi_type: pickPoiType(raw),
                            source_field: sourceField,
                            branch_index: branchIndex,
                            intersects_with: pickIntersectsWith(raw),
                        });
                    }
                    return rows;
                };

                const getSketch = (expressionId) => (
                    calc?.controller?.grapher2d?.graphSketches?.[expressionId] || null
                );

                const collectSketchPOI = (sketch) => {
                    if (!sketch || typeof sketch !== "object") return [];
                    const rows = [];
                    const branches = Array.isArray(sketch.branches) ? sketch.branches : [];
                    for (let index = 0; index < branches.length; index += 1) {
                        const branch = branches[index];
                        rows.push(...normalizePOIItems(branch?.__cachedPOI, "__cachedPOI", index));
                        rows.push(...normalizePOIStructured(branch?.__cachedPOI, "__cachedPOI", index));
                        rows.push(...normalizePOIItems(branch?.poi, "poi", index));
                        rows.push(...normalizePOIStructured(branch?.poi, "poi", index));
                        rows.push(...normalizePOIMapItems(branch?.__cachedPOIIs, "__cachedPOIIs", index));
                        rows.push(...normalizePOIMapItems(branch?.__cachedPOIIds, "__cachedPOIIds", index));
                        rows.push(...normalizePOIMapItems(branch?.cachedPOIIds, "cachedPOIIds", index));
                        rows.push(...normalizePOIMapItems(branch?.poiIs, "poiIs", index));
                        rows.push(...normalizePOIMapItems(branch?.__poiIds, "__poiIds", index));
                        rows.push(...normalizePOIMapItems(branch?.poiIds, "poiIds", index));
                    }
                    rows.push(...normalizePOIItems(sketch.__cachedPOI, "sketch.__cachedPOI", -1));
                    rows.push(...normalizePOIStructured(sketch.__cachedPOI, "sketch.__cachedPOI", -1));
                    rows.push(...normalizePOIItems(sketch.poi, "sketch.poi", -1));
                    rows.push(...normalizePOIStructured(sketch.poi, "sketch.poi", -1));
                    rows.push(...normalizePOIMapItems(sketch.__cachedPOIIs, "sketch.__cachedPOIIs", -1));
                    rows.push(...normalizePOIMapItems(sketch.__cachedPOIIds, "sketch.__cachedPOIIds", -1));
                    rows.push(...normalizePOIMapItems(sketch.cachedPOIIds, "sketch.cachedPOIIds", -1));
                    rows.push(...normalizePOIMapItems(sketch.poiIs, "sketch.poiIs", -1));
                    rows.push(...normalizePOIMapItems(sketch.__poiIds, "sketch.__poiIds", -1));
                    rows.push(...normalizePOIMapItems(sketch.poiIds, "sketch.poiIds", -1));

                    const uniqueRows = [];
                    const seen = new Set();
                    for (const row of rows) {
                        const poiTypeToken = Number.isFinite(Number(row.poi_type))
                            ? String(Number(row.poi_type))
                            : "";
                        const intersectsToken = normalizeToken(row.intersects_with) || "";
                        const key = `${row.x.toFixed(10)}|${row.y.toFixed(10)}|${poiTypeToken}|${intersectsToken}`;
                        if (seen.has(key)) continue;
                        seen.add(key);
                        uniqueRows.push(row);
                    }
                    return uniqueRows;
                };

                const enableExpressionPOI = (expressionId) => {
                    try {
                        calc?.setExpression?.({
                            id: expressionId,
                            hidden: false,
                            showPOI: true,
                            showHighlight: true,
                        });
                    } catch (error) {
                        // Best effort only.
                    }
                    try {
                        const sketch = getSketch(expressionId);
                        if (sketch && typeof sketch === "object") {
                            sketch.showPOI = true;
                            sketch.showHighlight = true;
                        }
                    } catch (error) {
                        // Best effort only.
                    }
                };

                const pointFingerprint = (rows) => {
                    if (!Array.isArray(rows) || rows.length === 0) return "";
                    return rows
                        .map((row) => {
                            const poiTypeToken = Number.isFinite(Number(row?.poi_type))
                                ? String(Number(row.poi_type))
                                : "";
                            const xToken = Number(row?.x).toFixed(10);
                            const yToken = Number(row?.y).toFixed(10);
                            return `${xToken}|${yToken}|${poiTypeToken}`;
                        })
                        .sort()
                        .join("||");
                };

                const intersectionPoiCount = (rows) => {
                    if (!Array.isArray(rows)) return 0;
                    let count = 0;
                    for (const row of rows) {
                        if (!row || typeof row !== "object") continue;
                        if (Number(row.poi_type) === 1001) count += 1;
                    }
                    return count;
                };

                const isBetterPoiSet = (candidateRows, bestRows) => {
                    const candidateCount = Array.isArray(candidateRows) ? candidateRows.length : 0;
                    const bestCount = Array.isArray(bestRows) ? bestRows.length : 0;
                    if (candidateCount > bestCount) return true;
                    if (candidateCount < bestCount) return false;

                    const candidateIntersections = intersectionPoiCount(candidateRows);
                    const bestIntersections = intersectionPoiCount(bestRows);
                    return candidateIntersections > bestIntersections;
                };

                const readPoiUntilStable = async (
                    expressionId,
                    maxAttempts = 120,
                    minAttempts = 60,
                    settleTicks = 16,
                    reselectEvery = 8,
                ) => {
                    let bestRows = [];
                    let stableTicks = 0;
                    let previousFingerprint = "";
                    let attempts = 0;
                    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
                        attempts = attempt + 1;

                        if (attempt % reselectEvery === 0) {
                            enableExpressionPOI(expressionId);
                            try {
                                calc?.controller?.dispatch?.({ type: "select-expression", id: expressionId });
                            } catch (error) {
                                // Best effort only.
                            }
                        }

                        const rows = collectSketchPOI(getSketch(expressionId));
                        if (isBetterPoiSet(rows, bestRows)) {
                            bestRows = rows;
                        }

                        const currentFingerprint = pointFingerprint(rows);
                        if (currentFingerprint && currentFingerprint === previousFingerprint) {
                            stableTicks += 1;
                        } else {
                            stableTicks = 0;
                        }
                        previousFingerprint = currentFingerprint;

                        if (
                            attempts >= minAttempts
                            && bestRows.length > 0
                            && stableTicks >= settleTicks
                        ) {
                            return {
                                rows: bestRows,
                                attempts,
                                timed_out: false,
                            };
                        }
                        await sleep(50);
                    }

                    return {
                        rows: bestRows,
                        attempts,
                        timed_out: true,
                    };
                };

                const forcePoiComputation = async (expressionId) => {
                    enableExpressionPOI(expressionId);
                    try {
                        calc?.controller?.dispatch?.({ type: "select-expression", id: expressionId });
                    } catch (error) {
                        // Best effort only, continue with passive polling.
                    }
                    // Real DOM clicks in Python pre-trigger POI/intersection computation.
                    return await readPoiUntilStable(expressionId, 60, 20, 8, 5);
                };

                const plotExpressions = (exprs || []).filter((expr) => {
                    if (!expr || !expr.id) return false;
                    return String(expr.id).startsWith("expr_");
                });

                const forceMetaById = {};
                for (const expr of plotExpressions) {
                    const expressionId = String(expr.id);
                    enableExpressionPOI(expressionId);
                }
                // Second pass (phase 1): force POI computation for every plotted expression.
                for (const expr of plotExpressions) {
                    const expressionId = String(expr.id);
                    forceMetaById[expressionId] = await forcePoiComputation(expressionId);
                }

                const out = [];
                // Second pass (phase 2): read POIs after force pass has run for all expressions.
                for (const expr of plotExpressions) {
                    const expressionId = String(expr.id);
                    enableExpressionPOI(expressionId);
                }
                for (const expr of plotExpressions) {
                    const expressionId = String(expr.id);
                    const forceMeta = forceMetaById[expressionId] || {};
                    const forceRows = Array.isArray(forceMeta.rows) ? forceMeta.rows : [];
                    const readMeta = await readPoiUntilStable(expressionId, 60, 20, 8, 5);
                    const readRows = Array.isArray(readMeta.rows) ? readMeta.rows : [];
                    const poiRows = isBetterPoiSet(readRows, forceRows) ? readRows : forceRows;
                    const finalSketch = getSketch(expressionId);

                    const expressionResult = {
                        id: expressionId,
                        latex: typeof expr.latex === "string" ? expr.latex : "",
                        poi_count: poiRows.length,
                        // force_attempts: Number(forceMeta.attempts || 0),
                        // force_timed_out: Boolean(forceMeta.timed_out),
                        // read_attempts: Number(readMeta.attempts || 0),
                        // read_timed_out: Boolean(readMeta.timed_out),
                        poi: poiRows,
                    };
                    if (includeDebugInternals) {
                        expressionResult.final_graph_sketch = toTraceSafeValue(finalSketch);
                    }
                    out.push(expressionResult);
                }
                return out;
            }""",
            {
                "exprs": rendered_expressions,
                "includeDebugInternals": bool(save_desmos_internals),
                "collectPoi": bool(collect_poi),
            },
        )
        poi_overlay_points, poi_overlay_summary = build_poi_overlay_points(
            points_of_interest,
            poi_display_options,
            max_points=max_poi_overlay_points,
        )
        poi_overlay_expressions = build_poi_overlay_expressions(poi_overlay_points)
        if poi_overlay_expressions:
            page.evaluate(
                """(overlayExprs) => {
                    if (!window.calc || typeof window.calc.setExpressions !== "function") return;
                    window.calc.setExpressions(overlayExprs);
                }""",
                poi_overlay_expressions,
            )
            time.sleep(0.35)

        if needs_poi_overlay or not screenshot_captured:
            page.locator("#calculator").screenshot(path=str(out_png.resolve()))

        graph_sketches: List[Dict[str, Any]] = []
        if save_desmos_internals and isinstance(points_of_interest, list):
            for expression in points_of_interest:
                if not isinstance(expression, dict):
                    continue
                graph_sketches.append(
                    {
                        "id": expression.get("id"),
                        "graph_sketch_id": expression.get("graph_sketch_id"),
                        "latex": expression.get("latex"),
                        "final_graph_sketch": expression.get("final_graph_sketch"),
                    }
                )
        browser.close()
    html_path.unlink(missing_ok=True)

    png_b64 = base64.b64encode(out_png.read_bytes()).decode("utf-8")
    result_payload = {
        "ok": True,
        "png_path": out_png.resolve(),
        "png_b64": png_b64,
        "diagnostic": None,
        "expression_analysis": json_safe(expression_analysis),
        "probe_points": json_safe(clicked_probe_points),
        "points_of_interest": json_safe(points_of_interest),
        "poi_overlay": json_safe(poi_overlay_summary),
    }
    if save_desmos_internals:
        result_payload["graph_sketches"] = json_safe(graph_sketches)
        result_payload["expression_click_debug"] = json_safe(expression_click_debug)
    return result_payload

def image_part_to_data_url(part: Dict[str, Any]) -> str:
    mime = part.get("mime", "image/png")
    if part.get("b64"):
        payload = part["b64"]
    elif part.get("path"):
        payload = base64.b64encode(Path(part["path"]).read_bytes()).decode("utf-8")
    else:
        raw = part.get("data")
        if isinstance(raw, bytes):
            payload = base64.b64encode(raw).decode("utf-8")
        elif isinstance(raw, str):
            payload = raw
        else:
            raise ValueError("Image part is missing path, b64, or data")
    return f"data:{mime};base64,{payload}"


def canonical_tool_call_id(name: str, arguments: Any, index: int) -> str:
    payload = {
        "name": str(name),
        "arguments": json_safe(arguments),
        "index": int(index),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"call_{index}_{digest}"


def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif part.get("type") == "image":
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_part_to_data_url(part),
                            },
                        }
                    )
                else:
                    parts.append({"type": "text", "text": json.dumps(json_safe(part))})
            payload.append({"role": role, "content": parts})
        else:
            payload.append({"role": role, "content": str(content)})
    return payload


def extract_text_from_response(raw_response: Dict[str, Any]) -> str:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(str(part["text"]))
            else:
                maybe_text = getattr(part, "text", None)
                if maybe_text:
                    chunks.append(str(maybe_text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def extract_finish_reason(raw_response: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        return None
    text = str(finish_reason).strip()
    return text or None


def find_prefix_in_text(text: str, target_prefix: str, forbidden_prefix: Optional[str] = None) -> Optional[int]:
    if not text or not target_prefix:
        return None
    
    idx = text.find(target_prefix)
    if idx < 0:
        return None
    
    # Check for multiple instances of target prefix
    if text.find(target_prefix, idx + 1) >= 0:
        return None
    
    # Check for forbidden prefix if specified
    if forbidden_prefix is not None:
        if text.find(forbidden_prefix) >= 0:
            return None
    
    return idx


def _extract_leading_json_object_candidate(text: str) -> Optional[Tuple[str, int, bool]]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    in_string = False
    escaping = False
    depth = 0
    idx = start
    while idx < len(text):
        ch = text[idx]
        if in_string:
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1], idx + 1, False
        idx += 1

    # Truncated object: return the remaining text and mark it as incomplete.
    return text[start:], len(text), True


def _repair_invalid_backslashes_in_json_strings(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    in_string = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            idx += 1
            continue

        # In a JSON string.
        if ch == '"':
            out.append(ch)
            in_string = False
            idx += 1
            continue
        if ch != "\\":
            out.append(ch)
            idx += 1
            continue

        # Backslash inside string.
        if idx + 1 >= len(text):
            out.append("\\\\")
            idx += 1
            continue

        nxt = text[idx + 1]
        if nxt in '"\\/bfnrt':
            out.append("\\")
            out.append(nxt)
            idx += 2
            continue

        if nxt == "u":
            hex_part = text[idx + 2 : idx + 6]
            if len(hex_part) == 4 and re.fullmatch(r"[0-9a-fA-F]{4}", hex_part):
                out.append("\\u")
                out.append(hex_part)
                idx += 6
                continue

        # Invalid escape (e.g. \sqrt): keep literal backslash by escaping it.
        out.append("\\\\")
        out.append(nxt)
        idx += 2

    return "".join(out)


def _remove_trailing_json_commas(text: str) -> str:
    updated = text
    while True:
        reduced = re.sub(r",\s*([}\]])", r"\1", updated)
        if reduced == updated:
            return updated
        updated = reduced


def _is_json_number_token(token: str) -> bool:
    t = str(token or "").strip()
    if not t:
        return False
    return bool(re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", t))


def _quote_non_json_numbers_in_known_fields(text: str) -> Tuple[str, bool]:
    """Quote values like 2*\\pi in known numeric fields so json.loads can succeed.

    Models often emit arithmetic expressions in numeric fields (JSON doesn't allow this), e.g.:
      "right": 2*\\pi
    We repair by quoting the token:
      "right": "2*\\pi"
    Then later we safely evaluate it to a float.
    """

    if not text:
        return text, False

    updated = text
    changed = False
    keys = ("left", "right", "bottom", "top", "x", "y", "width", "height")
    for key in keys:
        pattern = re.compile(
            rf'(\"{re.escape(key)}\"\s*:\s*)([^\"\[\{{\s][^,\}}\]]*)'
        )

        def repl(match: re.Match) -> str:
            nonlocal changed
            prefix = match.group(1)
            raw_value = match.group(2)
            value = str(raw_value).strip()
            if not value:
                return match.group(0)
            if value in {"true", "false", "null"}:
                return match.group(0)
            if _is_json_number_token(value):
                return match.group(0)
            changed = True
            return f'{prefix}"{value}"'

        rewritten = pattern.sub(repl, updated)
        updated = rewritten

    return updated, changed


def _safe_eval_math_expression(expr: Any) -> Optional[float]:
    if expr is None:
        return None
    if isinstance(expr, bool):
        return None
    if isinstance(expr, (int, float)):
        value = float(expr)
        return value if math.isfinite(value) else None

    text = str(expr).strip()
    if not text:
        return None

    # Normalize common spellings.
    text = text.replace("π", "pi")
    text = text.replace("\\pi", "pi")
    text = text.replace("^", "**")

    # Only allow a small character set.
    if re.search(r"[^0-9eEpi\+\-\*\./\(\)\s]", text):
        return None

    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    def eval_node(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            name = str(n.id).strip().lower()
            if name == "pi":
                return float(math.pi)
            if name == "e":
                return float(math.e)
            raise ValueError("unsupported name")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = eval_node(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            left = eval_node(n.left)
            right = eval_node(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            return left ** right
        raise ValueError("unsupported expression")

    try:
        value = float(eval_node(node))
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_numeric_field(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        evaluated = _safe_eval_math_expression(value)
        if evaluated is not None:
            return float(evaluated)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_bool_field(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(default)


def parse_poi_display_options(args: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "intersections": parse_bool_field(args.get("label_intersections"), False),
        "extrema": parse_bool_field(args.get("label_extrema"), False),
        "intercepts": parse_bool_field(args.get("label_intercepts"), False),
        "zeros": parse_bool_field(args.get("label_zeros"), False),
    }


def poi_display_requested(options: Dict[str, bool]) -> bool:
    return any(bool(options.get(category)) for category in POI_CATEGORY_ORDER)


def classify_poi_category(row: Dict[str, Any]) -> Optional[str]:
    poi_type_raw = row.get("poi_type")
    try:
        poi_type = int(float(poi_type_raw))
    except (TypeError, ValueError):
        poi_type = None

    if poi_type == 1001:
        return "intersections"
    if poi_type == 1002:
        return "zeros"
    if poi_type == 1003:
        return "intercepts"
    if poi_type == 1004:
        return "extrema"

    kind = str(row.get("kind") or "").strip().lower()
    source_field = str(row.get("source_field") or "").strip().lower()
    text = f"{kind} {source_field}"

    if "intersect" in text:
        return "intersections"
    if "extrem" in text or "vertex" in text or "maximum" in text or "minimum" in text:
        return "extrema"
    if "zero" in text or "root" in text or "x_intercept" in text or "x-intercept" in text:
        return "zeros"
    if "intercept" in text:
        return "intercepts"
    return None


def build_poi_overlay_points(
    points_of_interest: Any,
    poi_display: Dict[str, bool],
    max_points: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    requested = {category: bool(poi_display.get(category)) for category in POI_CATEGORY_ORDER}
    shown_counts = {category: 0 for category in POI_CATEGORY_ORDER}
    summary: Dict[str, Any] = {
        "requested": requested,
        "shown_counts": shown_counts,
        "total_shown": 0,
        "truncated": 0,
        "max_points": max_points,
    }
    if not poi_display_requested(poi_display):
        return [], summary
    if not isinstance(points_of_interest, list):
        return [], summary

    overlay_points: List[Dict[str, Any]] = []
    seen = set()
    truncated = 0
    for expression in points_of_interest:
        if not isinstance(expression, dict):
            continue
        poi_rows = expression.get("poi")
        if not isinstance(poi_rows, list):
            continue
        for row in poi_rows:
            if not isinstance(row, dict):
                continue
            category = classify_poi_category(row)
            if category is None or not requested.get(category):
                continue
            try:
                x = float(row.get("x"))
                y = float(row.get("y"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue

            key = f"{category}|{x:.4f}|{y:.4f}"
            if key in seen:
                continue
            seen.add(key)

            if max_points is not None and len(overlay_points) >= max_points:
                truncated += 1
                continue

            overlay_points.append(
                {
                    "x": x,
                    "y": y,
                    "category": category,
                }
            )
            shown_counts[category] += 1

    summary["total_shown"] = len(overlay_points)
    summary["truncated"] = int(truncated)
    return overlay_points, summary


def build_poi_overlay_expressions(overlay_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expressions: List[Dict[str, Any]] = []
    for index, point in enumerate(overlay_points):
        category = str(point.get("category") or "").strip().lower()
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        coordinate_label = (
            f"({format_desmos_compact_g(x, zero_tol=POI_LABEL_ZERO_TOL)}, "
            f"{format_desmos_compact_g(y, zero_tol=POI_LABEL_ZERO_TOL)})"
        )
        expressions.append(
            {
                "id": f"poi_overlay_{index}",
                "latex": coordinate_label,
                "label": coordinate_label,
                "showLabel": True,
                "color": POI_CATEGORY_POINT_COLOR.get(category, "#495057"),
            }
        )
    return expressions


def format_poi_overlay_note(tool_payload: Dict[str, Any]) -> str:
    overlay = tool_payload.get("poi_overlay")
    if not isinstance(overlay, dict):
        return ""

    requested_raw = overlay.get("requested")
    shown_raw = overlay.get("shown_counts")
    requested = requested_raw if isinstance(requested_raw, dict) else {}
    shown_counts = shown_raw if isinstance(shown_raw, dict) else {}

    requested_categories: List[str] = []
    shown_parts: List[str] = []
    category_label_colors = {
        "intersections": "blue labels",
        "extrema": "red labels",
        "intercepts": "green labels",
        "zeros": "orange labels",
    }
    for category in POI_CATEGORY_ORDER:
        if bool(requested.get(category)):
            requested_categories.append(category)
        count_raw = shown_counts.get(category)
        try:
            count = int(count_raw)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            color_note = category_label_colors.get(category)
            if color_note:
                shown_parts.append(f"{category} ({color_note})={count}")
            else:
                shown_parts.append(f"{category}={count}")

    if not requested_categories:
        return ""

    try:
        truncated = max(0, int(overlay.get("truncated") or 0))
    except (TypeError, ValueError):
        truncated = 0

    try:
        max_points_raw = overlay.get("max_points")
        max_points = None if max_points_raw is None else max(0, int(max_points_raw))
    except (TypeError, ValueError):
        max_points = None

    if shown_parts:
        note = "Number of POI markers rendered on plot: " + ", ".join(shown_parts) + "."
        if truncated > 0:
            if max_points is None:
                note += f" (+{truncated} more hidden by cap)"
            else:
                note += f" (+{truncated} more hidden by cap={max_points})"
        return note

    return (
        "POI markers were requested "
        f"({', '.join(requested_categories)}) but none were detected in the current view."
    )


def _split_top_level_csv_items(text: str) -> List[str]:
    items: List[str] = []
    start = 0
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escaping = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            brace_depth += 1
            continue
        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            continue
        if ch == "[":
            bracket_depth += 1
            continue
        if ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            continue
        if ch == "," and brace_depth == 0 and bracket_depth == 0:
            part = text[start:idx].strip()
            if part:
                items.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _expand_duplicate_latex_objects_in_expressions(text: str) -> Tuple[str, bool]:
    marker_match = re.search(r'"expressions"\s*:\s*\[', text)
    if not marker_match:
        return text, False

    array_start = marker_match.end() - 1
    idx = array_start
    in_string = False
    escaping = False
    depth = 0
    while idx < len(text):
        ch = text[idx]
        if in_string:
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
        idx += 1

    if idx >= len(text) or depth != 0:
        return text, False

    array_end = idx
    inner = text[array_start + 1 : array_end]
    items = _split_top_level_csv_items(inner)
    if not items:
        return text, False

    changed = False
    rewritten_items: List[str] = []
    latex_value_pattern = re.compile(r'"latex"\s*:\s*"((?:\\.|[^"\\])*)"')
    key_pattern = re.compile(r'"([^"\\]+)"\s*:')

    for raw_item in items:
        item = raw_item.strip()
        if not (item.startswith("{") and item.endswith("}")):
            rewritten_items.append(item)
            continue
        latex_values = latex_value_pattern.findall(item)
        if len(latex_values) <= 1:
            rewritten_items.append(item)
            continue

        keys = key_pattern.findall(item)
        if any(key != "latex" for key in keys):
            rewritten_items.append(item)
            continue

        changed = True
        rewritten_items.extend([f'{{"latex":"{value}"}}' for value in latex_values])

    if not changed:
        return text, False

    new_inner = ",".join(rewritten_items)
    rebuilt = text[: array_start + 1] + new_inner + text[array_end:]
    return rebuilt, True


def _try_parse_tool_payload(payload_text: str) -> Tuple[Optional[Dict[str, Any]], int, List[str], Optional[str]]:
    payload_text_stripped = payload_text.lstrip()
    if not payload_text_stripped:
        return None, 0, [], "empty_tool_payload"

    candidate_info = _extract_leading_json_object_candidate(payload_text_stripped)
    if candidate_info is None:
        return None, 0, [], "invalid_json"
    candidate_raw, consumed_end, was_truncated = candidate_info
    candidate = candidate_raw
    repair_notes: List[str] = []
    if was_truncated:
        # Close missing braces for the common "missing final }" failure.
        opens = candidate.count("{")
        closes = candidate.count("}")
        if opens > closes:
            candidate = candidate + ("}" * (opens - closes))
            repair_notes.append("appended_missing_closing_braces")

    variants: List[Tuple[str, List[str]]] = []

    quoted_candidate, quoted_changed = _quote_non_json_numbers_in_known_fields(candidate)
    if quoted_changed:
        variants.append((quoted_candidate, repair_notes + ["quoted_non_json_numeric_expressions"]))
    expanded_candidate, expanded = _expand_duplicate_latex_objects_in_expressions(candidate)
    if expanded:
        variants.append((expanded_candidate, repair_notes + ["expanded_duplicate_latex_expression_objects"]))
    variants.append((candidate, list(repair_notes)))

    if quoted_changed:
        expanded_quoted, expanded_quoted_changed = _expand_duplicate_latex_objects_in_expressions(quoted_candidate)
        if expanded_quoted_changed:
            variants.append(
                (
                    expanded_quoted,
                    repair_notes
                    + [
                        "quoted_non_json_numeric_expressions",
                        "expanded_duplicate_latex_expression_objects",
                    ],
                )
            )
    escaped_variant = _repair_invalid_backslashes_in_json_strings(candidate)
    if quoted_changed:
        escaped_quoted_variant = _repair_invalid_backslashes_in_json_strings(quoted_candidate)
        if escaped_quoted_variant != quoted_candidate:
            variants.append(
                (
                    escaped_quoted_variant,
                    repair_notes
                    + ["quoted_non_json_numeric_expressions", "escaped_invalid_backslashes"],
                )
            )
    escaped_expanded, escaped_expanded_changed = _expand_duplicate_latex_objects_in_expressions(escaped_variant)
    if escaped_expanded_changed:
        notes = list(repair_notes)
        if escaped_variant != candidate:
            notes.append("escaped_invalid_backslashes")
        notes.append("expanded_duplicate_latex_expression_objects")
        variants.append((escaped_expanded, notes))
    if escaped_variant != candidate:
        variants.append((escaped_variant, repair_notes + ["escaped_invalid_backslashes"]))
    comma_variant = _remove_trailing_json_commas(candidate)
    if comma_variant != candidate:
        variants.append((comma_variant, repair_notes + ["removed_trailing_commas"]))
    comma_escaped_variant = _remove_trailing_json_commas(escaped_variant)
    if comma_escaped_variant not in {v[0] for v in variants}:
        notes = list(repair_notes)
        if escaped_variant != candidate:
            notes.append("escaped_invalid_backslashes")
        if comma_escaped_variant != escaped_variant:
            notes.append("removed_trailing_commas")
        variants.append((comma_escaped_variant, notes))

    for variant_text, variant_notes in variants:
        try:
            parsed = json.loads(variant_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return json_safe(parsed), consumed_end, variant_notes, None

    # Last resort: tolerate Python-style dicts with single quotes. ??
    try:
        literal_value = ast.literal_eval(candidate)
    except Exception:
        return None, 0, [], "invalid_json"
    if isinstance(literal_value, dict):
        notes = repair_notes + ["python_literal_eval_fallback"]
        return json_safe(literal_value), consumed_end, notes, None
    return None, 0, [], "tool_payload_not_object"


def extract_prompt_tool_call(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    
    # Use helper to find tool prefix, ensuring no final prefix exists
    tool_prefix_idx = find_prefix_in_text(stripped, TOOL_REQUEST_PREFIX, forbidden_prefix=FINAL_RESPONSE_PREFIX)
    
    # Tool prefix not found or validation failed
    if tool_prefix_idx is None:
        return None

    payload_text = stripped[tool_prefix_idx + len(TOOL_REQUEST_PREFIX) :].strip()
    if not payload_text:
        return {
            "invalid_tool_request": True,
            "raw_payload_text": payload_text,
            "error_code": "empty_tool_payload",
            "error_detail": "No JSON object was found after the tool prefix.",
        }

    payload_text_stripped = payload_text.lstrip()
    payload, parsed_end, repair_notes, parse_error_code = _try_parse_tool_payload(payload_text)
    if payload is None:
        error_detail = "Unable to parse tool JSON payload."
        if parse_error_code == "invalid_json":
            try:
                decoder = json.JSONDecoder()
                decoder.raw_decode(payload_text_stripped)
            except json.JSONDecodeError as exc:
                error_detail = (
                    f"JSON parse error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
                )
        return {
            "invalid_tool_request": True,
            "raw_payload_text": payload_text,
            "error_code": parse_error_code or "invalid_json",
            "error_detail": error_detail,
        }

    trailing_text = payload_text_stripped[parsed_end:].strip()

    if not isinstance(payload, dict):
        return {
            "invalid_tool_request": True,
            "raw_payload_text": payload_text,
            "error_code": "tool_payload_not_object",
            "error_detail": f"Expected a JSON object, got {type(payload).__name__}.",
        }

    name = str(payload.get("name") or "").strip()
    arguments = payload.get("arguments")
    if not name:
        return {
            "invalid_tool_request": True,
            "raw_payload_text": payload_text,
            "error_code": "missing_tool_name",
            "error_detail": "Missing required top-level 'name' field.",
        }
    if not isinstance(arguments, dict):
        got = type(arguments).__name__ if arguments is not None else "null"
        return {
            "invalid_tool_request": True,
            "raw_payload_text": payload_text,
            "error_code": "missing_or_invalid_arguments",
            "error_detail": f"Field 'arguments' must be an object; got {got}.",
        }

    arguments_clean = dict(arguments)
    rationale = str(arguments_clean.pop("rationale", "") or "").strip()
    if not rationale:
        return {
            "invalid_tool_request": True,
            "raw_payload_text": payload_text,
            "missing_rationale": True,
            "error_code": "missing_rationale",
            "error_detail": "Tool request is missing arguments.rationale.",
        }
    tool_call = {
        "name": name,
        "arguments": json_safe(arguments_clean),
        "call_id": canonical_tool_call_id(name=name, arguments=arguments, index=0),
        "rationale": rationale,
    }
    if repair_notes:
        tool_call["json_repaired"] = True
        tool_call["json_repair_notes"] = repair_notes
    if trailing_text:
        tool_call["trailing_text_after_tool_json"] = trailing_text[:240]
        if trailing_text.startswith(FINAL_RESPONSE_PREFIX):
            tool_call["format_warning"] = "tool_and_final_in_same_turn"
        else:
            tool_call["format_warning"] = "extra_text_after_tool_json"
    return tool_call


def make_openai_compatible_generator(
    *,
    api_provider: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    top_p: Optional[float] = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    seed: Optional[int] = None,
    request_timeout: Optional[float] = None,
    reasoning_level: Optional[str] = None,
    openrouter_provider_order: Optional[List[str]] = None,
    openrouter_allow_fallbacks: bool = False,
    max_retries: int = 5,
    retry_base_sleep: float = 5.0,
) -> Callable[[List[Dict[str, Any]], Optional[int]], Dict[str, Any]]:
    def generate(
        messages: List[Dict[str, Any]],
        max_tokens_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        converted_messages = convert_messages(messages)
        effective_max_tokens = None if max_tokens_override is None else max(0, int(max_tokens_override))

        raw_response, error = call_api_with_retry(
            messages=converted_messages,
            api_provider=api_provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=effective_max_tokens,
            request_timeout=request_timeout,
            seed=seed,
            reasoning_level=reasoning_level,
            openrouter_provider_order=openrouter_provider_order,
            openrouter_allow_fallbacks=openrouter_allow_fallbacks,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )
        if error is not None or raw_response is None:
            raise RuntimeError(error or "API call failed")

        text = extract_text_from_response(raw_response)
        reasoning = json_safe(extract_reasoning(raw_response))
        tool_calls: List[Dict[str, Any]] = []
        prompt_tool_request_invalid = False
        prompt_tool_request_raw = None
        prompt_tool_request_error_code = None
        prompt_tool_request_error_detail = None
        prompt_tool_call = extract_prompt_tool_call(text)
        if prompt_tool_call is not None:
            if prompt_tool_call.get("invalid_tool_request"):
                prompt_tool_request_invalid = True
                prompt_tool_request_raw = prompt_tool_call.get("raw_payload_text")
                prompt_tool_request_error_code = prompt_tool_call.get("error_code")
                prompt_tool_request_error_detail = prompt_tool_call.get("error_detail")
            else:
                tool_calls = [prompt_tool_call]
        return {
            "text": text,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "raw_response": raw_response,
            "prompt_tool_request_invalid": prompt_tool_request_invalid,
            "prompt_tool_request_raw": prompt_tool_request_raw,
            "prompt_tool_request_error_code": prompt_tool_request_error_code,
            "prompt_tool_request_error_detail": prompt_tool_request_error_detail,
        }

    return generate


def desmos_plot(
    desmos_api_key: str,
    args: Dict[str, Any],
    out_png: Path,
    return_b64: bool = False,
    save_desmos_internals: bool = False,
    max_poi_overlay_points: Optional[int] = None,
) -> Dict[str, Any]:
    expressions_raw = args.get("expressions") or []
    if not isinstance(expressions_raw, list) or not expressions_raw:
        raise ValueError("desmos_plot requires a non-empty 'expressions' list")

    expressions: List[str] = []
    for item in expressions_raw:
        if not isinstance(item, dict) or "latex" not in item:
            raise ValueError("Each expression must be an object with a 'latex' field")
        normalized = normalize_desmos_latex(str(item["latex"]))
        expressions.append(normalized)

    bounds_input = args.get("bounds") or {}
    bounds = {
        "left": parse_numeric_field(bounds_input.get("left"), -10),
        "right": parse_numeric_field(bounds_input.get("right"), 10),
        "bottom": parse_numeric_field(bounds_input.get("bottom"), -10),
        "top": parse_numeric_field(bounds_input.get("top"), 10),
    }
    width = int(parse_numeric_field(args.get("width"), 1280))
    height = int(parse_numeric_field(args.get("height"), 720))
    probe_points_raw = args.get("probe_points") or []
    probe_points: List[Dict[str, float]] = []
    if probe_points_raw:
        if not isinstance(probe_points_raw, list):
            raise ValueError("probe_points must be a list of {x, y} objects")
        for item in probe_points_raw:
            if not isinstance(item, dict) or "x" not in item or "y" not in item:
                raise ValueError("Each probe point must be an object with numeric 'x' and 'y' fields")
            probe_points.append(
                {
                    "x": parse_numeric_field(item.get("x"), 0.0),
                    "y": parse_numeric_field(item.get("y"), 0.0),
                }
            )
    poi_display = parse_poi_display_options(args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            render_desmos_plot,
            desmos_api_key,
            expressions,
            bounds,
            out_png,
            probe_points,
            poi_display,
            width,
            height,
            save_desmos_internals,
            max_poi_overlay_points,
        ).result()

    analysis = result.get("expression_analysis") or []
    errors = [row for row in analysis if row.get("has_error")]
    graphable = [row for row in analysis if row.get("is_graphable")]
    png_path = Path(result["png_path"])
    has_content = is_nonblank_png(png_path) if png_path.exists() else False
    # Only treat the call as failed if every expression Desmos saw was a hard
    # parsing error, or no PNG was produced at all. A non-graphable expression
    # (e.g. a numeric calculation like `2*\log_2(10)`) is still useful: Desmos
    # shows the computed value in the expression panel, which is part of the
    # screenshot.
    all_errored = bool(analysis) and len(errors) == len(analysis)
    result["ok"] = bool(png_path.exists() and not all_errored)

    diagnostics: List[str] = []
    warnings: List[str] = []
    if errors:
        message = (
            "Desmos parsing errors: "
            + "; ".join(
                f"{row.get('latex')}: {row.get('error_message') or 'unknown error'}"
                for row in errors[:3]
            )
        )
        if all_errored:
            diagnostics.append(message)
        else:
            warnings.append(message)
    if not graphable and not all_errored:
        warnings.append(
            "No expression is graphable in this view; the screenshot still "
            "shows the expression panel where computed values appear. "
            "If you intended to plot a curve, fix the expression or bounds."
        )
    if not has_content and not all_errored:
        warnings.append(
            "The graph area looks blank; check the expression panel in the "
            "screenshot for any computed values, or adjust bounds."
        )
    if diagnostics:
        result["diagnostic"] = " ".join(diagnostics)
    if warnings:
        result["warning"] = " ".join(warnings)
    if not return_b64:
        result["png_b64"] = None
    return result


def desmos_plot_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    expression_analysis = result.get("expression_analysis") or []
    invalid_expressions = [
        {
            "latex": row.get("latex"),
            "error_message": row.get("error_message"),
        }
        for row in expression_analysis
        if row.get("has_error")
    ]
    trace_payload = {
        "ok": result.get("ok"),
        "diagnostic": result.get("diagnostic"),
        "warning": result.get("warning"),
        "expression_analysis": expression_analysis,
        "invalid_expressions": invalid_expressions,
        "probe_points": result.get("probe_points") or [],
        "points_of_interest": result.get("points_of_interest") or [],
        "poi_overlay": result.get("poi_overlay") or {},
    }
    if "graph_sketches" in result:
        trace_payload["graph_sketches"] = result.get("graph_sketches") or []
    if "expression_click_debug" in result:
        trace_payload["expression_click_debug"] = result.get("expression_click_debug") or {}
    return trace_payload


def format_invalid_expression_note(tool_payload: Dict[str, Any]) -> str:
    invalid_expressions = tool_payload.get("invalid_expressions") or []
    if not invalid_expressions:
        return ""

    parts: List[str] = []
    for row in invalid_expressions[:3]:
        latex = str(row.get("latex") or "").strip() or "<unknown>"
        error_message = str(row.get("error_message") or "invalid syntax").strip()
        parts.append(f"{latex} -> {error_message}")
    return (
        "Desmos rejected these expressions: "
        + "; ".join(parts)
    )


def format_compact_tool_error(tool_payload: Dict[str, Any]) -> str:
    diagnostic = str(tool_payload.get("diagnostic") or "").strip()
    invalid_expressions = tool_payload.get("invalid_expressions") or []

    if not invalid_expressions and not diagnostic:
        return (
            "Desmos rejected the request. Fix expression syntax or bounds and retry desmos_plot."
        )

    if invalid_expressions:
        unique_issue_messages: List[str] = []
        for row in invalid_expressions:
            message = str(row.get("error_message") or "invalid syntax").strip()
            if message and message not in unique_issue_messages:
                unique_issue_messages.append(message)
        issue_clause = ""
        if unique_issue_messages:
            issue_clause = f" Main issue: {unique_issue_messages[0]}"
        return (
            f"Desmos rejected {len(invalid_expressions)} expression(s)."
            f"{issue_clause} Fix syntax then retry desmos_plot."
        )

    return diagnostic


def image_part_from_ref(dataset_dir: Path, image_ref: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(image_ref, dict):
        return None
    raw_path = str(image_ref.get("path", "") or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate if candidate.exists() else None
    else:
        local_candidate = (dataset_dir / candidate).resolve()
        resolved = local_candidate if local_candidate.exists() else None
    if resolved is None:
        return None
    mime = str(image_ref.get("content_type") or "image/png")
    return {
        "type": "image",
        "mime": mime,
        "path": str(resolved),
    }


def question_image_parts(dataset_dir: Path, question: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for index, image_ref in enumerate(question.get("question_images") or []):
        image_part = image_part_from_ref(dataset_dir, image_ref)
        if image_part is None:
            continue
        parts.append({"type": "text", "text": f"Question image {index + 1}:"})
        parts.append(image_part)
    return parts


def option_content_parts(dataset_dir: Path, question: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = question.get("options") or []
    parts: List[Dict[str, Any]] = []
    has_option_images = any(isinstance(option, dict) and option.get("image") for option in options)

    if has_option_images:
        parts.append({"type": "text", "text": "Options:"})
        for option in options:
            if not isinstance(option, dict):
                parts.append({"type": "text", "text": str(option)})
                continue
            label = str(option.get("label") or "").strip()
            parts.append({"type": "text", "text": f"{label}."})
            option_image = image_part_from_ref(dataset_dir, option.get("image"))
            if option_image is not None:
                parts.append(option_image)
            elif option.get("text"):
                parts.append({"type": "text", "text": str(option.get("text") or "")})
    else:
        opts_text = "\n".join(
            [
                f"{o.get('label')}. {o.get('text') or ''}".strip()
                if isinstance(o, dict)
                else str(o)
                for o in options
            ]
        )
        parts.append({"type": "text", "text": f"Options:\n{opts_text}"})
    return parts


def serialize_messages(messages: List[Dict[str, Any]], output_root: Path) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for message in messages:
        item: Dict[str, Any] = {"role": message.get("role")}
        content = message.get("content")
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                part_copy = json_safe(part)
                if "path" in part_copy:
                    part_copy["path"] = relpath_for_output(Path(part_copy["path"]), output_root)
                parts.append(part_copy)
            item["content"] = parts
        else:
            item["content"] = json_safe(content)
        serialized.append(item)
    return serialized


def append_user_message_trace(
    messages: List[Dict[str, Any]],
    trace: List[Dict[str, Any]],
    *,
    step: int,
    text: str,
    output_root: Path,
    event: str,
    image_path: Optional[Path] = None,
    trace_field: Optional[str] = "follow_up_user_prompt",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    if image_path is None:
        message = {"role": "user", "content": text}
    else:
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "mime": "image/png", "path": str(image_path)},
            ],
        }
    messages.append(message)
    trace_entry: Dict[str, Any] = {"step": step, "event": event}
    if trace_field is not None:
        trace_entry[trace_field] = serialize_messages([message], output_root)[0]
    if extra_fields:
        trace_entry.update(extra_fields)
    trace.append(trace_entry)


def save_request_payload_debug(
    response_dir: Path,
    step: int,
    messages: List[Dict[str, Any]],
    max_tokens_override: Optional[int],
) -> Path:
    payload_dir = response_dir / "api_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"step{step:03d}_request.json"
    payload = {
        "step": int(step),
        "max_tokens_override": max_tokens_override,
        "messages": convert_messages(messages),
    }
    payload_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload_path


def build_result_entry(
    question: Dict[str, Any],
    *,
    final_response_text: Optional[str],
    final_reasoning: Optional[Any],
    finish_reason: Optional[str] = None,
    error: Optional[str] = None,
    final_response: Optional[Dict[str, Any]] = None,
    raw_response: Optional[Dict[str, Any]] = None,
    screenshots: Optional[List[str]] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    api_provider: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    base_url: Optional[str] = None,
    reasoning_level: Optional[str] = None,
    openrouter_provider: Optional[str] = None,
    openrouter_allow_fallbacks: Optional[bool] = None,
    max_completion_tokens_per_question: Optional[int] = None,
    deterministic: Optional[bool] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    last_step_prompt_tokens: Optional[int] = None,
    last_step_completion_tokens: Optional[int] = None,
    last_step_total_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    num_input_images = len(question.get("question_images") or [])
    num_input_images += sum(
        1
        for option in (question.get("options") or [])
        if isinstance(option, dict) and option.get("image")
    )
    has_image_options = any(
        isinstance(option, dict) and option.get("image")
        for option in (question.get("options") or [])
    )
    final_without_successful_plot = bool(
        final_response_text and screenshots is not None and not screenshots
    )

    result_entry = {
        "id": question.get("id"),
        "qnum": question.get("qnum"),
        "difficulty": question.get("difficulty"),
        "variant": "desmos_agent",
        "num_input_images": num_input_images,
        "has_image_options": has_image_options,
        "final_response_text": final_response_text,
        "final_without_successful_plot": final_without_successful_plot,
        "finish_reason": finish_reason,
        "gt": str(question.get("answer")) if question.get("answer") is not None else None,
        # Backward-compatible cumulative usage fields.
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # Explicit cumulative aliases for easier downstream interpretation.
        "cumulative_prompt_tokens": prompt_tokens,
        "cumulative_completion_tokens": completion_tokens,
        "cumulative_total_tokens": total_tokens,
        # Explicit last model-response usage to avoid confusion with cumulative totals.
        "last_step_prompt_tokens": last_step_prompt_tokens,
        "last_step_completion_tokens": last_step_completion_tokens,
        "last_step_total_tokens": last_step_total_tokens,
        "final_reasoning": final_reasoning,
    }
    result_entry["meta"] = {
        "model": model,
        "api_provider": api_provider,
        "base_url": base_url,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "seed": seed,
        "reasoning_level": reasoning_level,
        "openrouter_provider": openrouter_provider,
        "openrouter_allow_fallbacks": openrouter_allow_fallbacks,
        "max_completion_tokens_per_question": max_completion_tokens_per_question,
        "deterministic": deterministic,
    }
    if final_response is not None:
        result_entry["final_response"] = final_response
    if raw_response is not None:
        result_entry["raw_response"] = raw_response
    if screenshots is not None:
        result_entry["screenshots"] = screenshots
    if trace is not None:
        result_entry["trace"] = trace
    if error is not None:
        result_entry["error"] = error
    return result_entry


def token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def solve_one_question(
    generate_response: Callable[[List[Dict[str, Any]], Optional[int]], Dict[str, Any]],
    desmos_tool: Callable[[Dict[str, Any], Path, bool], Dict[str, Any]],
    dataset_path: Path,
    output_json_path: Path,
    responses_dir: Path,
    save_raw_response: bool,
    save_sent_messages: bool,
    save_api_payloads: bool,
    max_completion_tokens_per_question: int,
    max_screenshots_per_question: int,
    model_id: str,
    api_provider: str,
    base_url: str,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    min_p: Optional[float],
    seed: Optional[int],
    reasoning_level: Optional[str],
    openrouter_provider: Optional[str],
    openrouter_allow_fallbacks: bool,
    deterministic: bool,
    require_successful_plot: bool,
    question: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_path = dataset_path.resolve()
    dataset_dir = dataset_path.parent
    output_json_path = output_json_path.resolve()
    output_root = output_json_path.parent.resolve()
    responses_root = responses_dir.resolve()

    qid = str(question.get("id", question.get("qnum", "unknown")))
    response_dir = responses_root / sanitize_filename_component(qid)
    question_dir = response_dir / "screenshots"
    response_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)
    per_question_path = response_dir / "result.json"

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Question: {question.get('question_text', '')}"}
    ]
    user_content.extend(question_image_parts(dataset_dir, question))
    user_content.extend(option_content_parts(dataset_dir, question))
    screenshot_limit = max(0, int(max_screenshots_per_question))
    user_content.append(
        {
            "type": "text",
            "text": question_rules_text(
                max_completion_tokens_per_question,
                screenshot_limit,
            ),
        }
    )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt_text(
                max_completion_tokens_per_question,
                screenshot_limit,
            ),
        },
        {"role": "user", "content": user_content},
    ]
    trace: List[Dict[str, Any]] = []
    trace.append(
        {
            "event": "initial_prompt",
            "messages": serialize_messages(messages, output_root),
        }
    )
    screenshot_paths: List[str] = []
    plot_count = 0
    last_response: Optional[Dict[str, Any]] = None
    question_prompt_tokens = 0
    question_completion_tokens = 0
    question_total_tokens = 0
    last_step_prompt_tokens: Optional[int] = None
    last_step_completion_tokens: Optional[int] = None
    last_step_total_tokens: Optional[int] = None
    question_token_budget = max(0, int(max_completion_tokens_per_question))
    question_budget_exhausted = False

    def build_question_result(
        *,
        final_response_text: Optional[str],
        final_reasoning: Optional[Any],
        finish_reason: Optional[str],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = build_result_entry(
            question,
            final_response_text=final_response_text,
            final_reasoning=json_safe(final_reasoning) if final_reasoning is not None else None,
            finish_reason=finish_reason,
            final_response=None,
            screenshots=screenshot_paths,
            trace=trace,
            model=model_id,
            api_provider=api_provider,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            reasoning_level=reasoning_level,
            openrouter_provider=openrouter_provider,
            openrouter_allow_fallbacks=openrouter_allow_fallbacks,
            max_completion_tokens_per_question=max_completion_tokens_per_question,
            deterministic=deterministic,
            prompt_tokens=question_prompt_tokens,
            completion_tokens=question_completion_tokens,
            total_tokens=question_total_tokens,
            last_step_prompt_tokens=last_step_prompt_tokens,
            last_step_completion_tokens=last_step_completion_tokens,
            last_step_total_tokens=last_step_total_tokens,
            error=error,
        )
        if isinstance(entry.get("meta"), dict):
            entry["meta"]["max_screenshots_per_question"] = screenshot_limit or None
            entry["meta"]["require_successful_plot"] = bool(require_successful_plot)
            entry["meta"]["enforce_successful_plot"] = bool(require_successful_plot)
        return entry

    def should_stop_now() -> bool:
        return question_budget_exhausted

    def append_tool_error_trace(
        step: int,
        error_message: str,
        *,
        tool_call: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra_fields: Dict[str, Any] = {
            "tool_error": error_message,
            "question_budget_exhausted": question_budget_exhausted,
            "max_screenshots_per_question": screenshot_limit or None,
            "screenshots_so_far": len(screenshot_paths),
            "screenshot_budget_exhausted": bool(screenshot_limit > 0 and len(screenshot_paths) >= screenshot_limit),
        }
        if tool_call is not None:
            extra_fields["tool_call"] = json_safe(tool_call)
        if tool_result is not None:
            extra_fields["tool_result"] = json_safe(tool_result)
        append_user_message_trace(
            messages,
            trace,
            step=step,
            text=error_message,
            output_root=output_root,
            event="tool_error",
            trace_field=None,
            extra_fields=extra_fields,
        )

    def process_tool_calls(step: int, tool_calls: List[Dict[str, Any]]) -> None:
        nonlocal plot_count
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", ""))
            tool_rationale = str(tool_call.get("rationale") or "").strip()
            tool_missing_rationale = bool(tool_call.get("missing_rationale"))
            tool_args = tool_call.get("arguments") or {}
            if tool_name != "desmos_plot":
                error_message = format_tool_error_message(
                    error_code="unknown_tool",
                    detail=f"Unknown tool '{tool_name}'.",
                )
                append_tool_error_trace(step, error_message)
                if should_stop_now():
                    return
                continue

            if screenshot_limit > 0 and len(screenshot_paths) >= screenshot_limit:
                append_user_message_trace(
                    messages,
                    trace,
                    step=step,
                    text=(
                        f"Screenshot limit reached ({screenshot_limit}). You cannot request another desmos_plot screenshot. "
                        "Only reason over the visible evidence from the screenshots and images you already have. "
                        "If that evidence is still insufficient, use selected_option \"N/A\" and briefly explain the visual gap in errors. "
                        "Reply ONLY with a final answer in this format:\n"
                        f"{FINAL_RESPONSE_EXACT}"
                    ),
                    output_root=output_root,
                    event="follow_up_prompt",
                    extra_fields={
                        "screenshot_budget_exhausted": True,
                        "max_screenshots_per_question": screenshot_limit,
                        "screenshots_so_far": len(screenshot_paths),
                        "blocked_tool_call": json_safe(tool_call),
                        "question_budget_exhausted": question_budget_exhausted,
                    },
                )
                if should_stop_now():
                    return
                continue

            if tool_missing_rationale or not tool_rationale:
                error_message = format_tool_error_message(
                    error_code="missing_rationale",
                    detail="desmos_plot request is missing a non-empty arguments.rationale.",
                )
                append_tool_error_trace(step, error_message, tool_call=tool_call)
                if should_stop_now():
                    return
                continue

            if not tool_args:
                error_message = format_tool_error_message(
                    error_code="missing_arguments",
                    detail="desmos_plot request is missing arguments.",
                )
                append_tool_error_trace(step, error_message, tool_call=tool_call)
                if should_stop_now():
                    return
                continue

            out_png = question_dir / f"step{step}_plot{plot_count}.png"
            plot_count += 1

            try:
                result = desmos_tool(tool_args, out_png=out_png)
            except Exception as exc:
                error_message = format_tool_error_message(
                    error_code="tool_exception",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                append_tool_error_trace(step, error_message)
                if should_stop_now():
                    return
                continue

            tool_payload = desmos_plot_trace(result)
            if not result.get("ok"):
                png_path = Path(result.get("png_path") or out_png)
                try:
                    if png_path.exists():
                        png_path.unlink()
                except OSError:
                    pass
                compact_error = format_compact_tool_error(tool_payload)
                error_message = format_tool_error_message(
                    error_code="desmos_plot_rejected",
                    detail=compact_error,
                )
                append_tool_error_trace(step, error_message, tool_result=tool_payload)
                if should_stop_now():
                    return
                continue

            relative_png = relpath_for_output(Path(result["png_path"]), output_root)
            screenshot_paths.append(relative_png)
            invalid_note = format_invalid_expression_note(tool_payload)
            poi_overlay_note = format_poi_overlay_note(tool_payload)
            warning_note = str(tool_payload.get("warning") or "").strip()
            follow_up_prompt = tool_result_prompt(screenshot_limit, len(screenshot_paths))
            if invalid_note:
                follow_up_prompt = (
                    f"{follow_up_prompt}\n\n"
                    "Diagnostics to fix before next tool call:\n"
                    f"{invalid_note}"
                )
            if warning_note:
                follow_up_prompt = (
                    f"{follow_up_prompt}\n\n"
                    f"Warning: {warning_note}"
                )
            if poi_overlay_note:
                follow_up_prompt = (
                    f"{follow_up_prompt}\n\n"
                    f"{poi_overlay_note}"
                )
            append_user_message_trace(
                messages,
                trace,
                step=step,
                text=follow_up_prompt,
                output_root=output_root,
                event="tool_result",
                image_path=Path(result["png_path"]),
                extra_fields={
                    "tool_result": tool_payload,
                    "question_budget_exhausted": question_budget_exhausted,
                    "max_screenshots_per_question": screenshot_limit or None,
                    "screenshots_so_far": len(screenshot_paths),
                    "screenshot_budget_exhausted": bool(screenshot_limit > 0 and len(screenshot_paths) >= screenshot_limit),
                },
            )
            if should_stop_now():
                return

    for step in itertools.count():
        remaining_completion_budget: Optional[int] = None
        if question_token_budget > 0:
            remaining_completion_budget = max(0, question_token_budget - question_completion_tokens)
            if remaining_completion_budget <= 0:
                question_budget_exhausted = True
                break

        if save_api_payloads:
            payload_path = save_request_payload_debug(
                response_dir=response_dir,
                step=step,
                messages=messages,
                max_tokens_override=remaining_completion_budget,
            )

        try:
            last_response = generate_response(
                messages=messages,
                max_tokens_override=remaining_completion_budget,
            )
        except Exception as exc:
            error_text = f"question_failed: {type(exc).__name__}: {exc}"
            print(f"{qid} failed: {error_text}", file=sys.stderr)
            trace.append(
                {
                    "step": step,
                    "event": "model_error",
                    "error": error_text,
                    "question_budget_exhausted": question_budget_exhausted,
                }
            )
            failed_result = build_question_result(
                final_response_text=last_response.get("text", "") if last_response else None,
                final_reasoning=last_response.get("reasoning") if last_response else None,
                finish_reason=extract_finish_reason(last_response.get("raw_response") if last_response else None),
                error=error_text,
            )
            write_json(per_question_path, failed_result)
            return failed_result
        usage_info = extract_usage(last_response.get("raw_response"))
        last_step_prompt_tokens = token_count(usage_info.get("prompt_tokens"))
        last_step_completion_tokens = token_count(usage_info.get("completion_tokens"))
        last_step_total_tokens = token_count(usage_info.get("total_tokens"))
        question_prompt_tokens += token_count(usage_info.get("prompt_tokens"))
        question_completion_tokens += token_count(usage_info.get("completion_tokens"))
        question_total_tokens += token_count(usage_info.get("total_tokens"))
        if question_token_budget > 0 and question_completion_tokens >= question_token_budget:
            question_budget_exhausted = True
        trace.append(
            {
                "step": step,
                "event": "model_response",
                "model_text": last_response.get("text", ""),
                "model_reasoning": json_safe(last_response.get("reasoning")),
                "tool_calls": [json_safe(call) for call in last_response.get("tool_calls", [])],
                "usage": usage_info,
                "question_budget_exhausted": question_budget_exhausted,
                "max_screenshots_per_question": screenshot_limit or None,
                "screenshots_so_far": len(screenshot_paths),
                "screenshot_budget_exhausted": bool(screenshot_limit > 0 and len(screenshot_paths) >= screenshot_limit),
                "prompt_tool_request_invalid": bool(last_response.get("prompt_tool_request_invalid")),
                "prompt_tool_request_raw": last_response.get("prompt_tool_request_raw"),
                "prompt_tool_request_error_code": last_response.get("prompt_tool_request_error_code"),
                "prompt_tool_request_error_detail": last_response.get("prompt_tool_request_error_detail"),
                "raw_response": last_response.get("raw_response") if save_raw_response else None,
            }
        )
        if save_sent_messages:
            trace[-1]["sent_messages"] = serialize_messages(messages, output_root)

        assistant_text = str(last_response.get("text", "") or "").strip()
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

        # If the budget is exhausted, never send more follow-ups. Still accept a final
        # answer already returned in this model response unless successful plots are
        # explicitly required and none exist yet.
        if should_stop_now():
            if assistant_text:
                final_prefix_idx = find_prefix_in_text(
                    assistant_text,
                    FINAL_RESPONSE_PREFIX,
                    forbidden_prefix=TOOL_REQUEST_PREFIX,
                )
                if final_prefix_idx is not None:
                    if screenshot_paths or not require_successful_plot:
                        result = build_question_result(
                            final_response_text=assistant_text,
                            final_reasoning=last_response.get("reasoning"),
                            finish_reason=extract_finish_reason(last_response.get("raw_response") if last_response else None),
                        )
                        write_json(per_question_path, result)
                        return result

                    result = build_question_result(
                        final_response_text=assistant_text,
                        final_reasoning=last_response.get("reasoning"),
                        finish_reason=extract_finish_reason(last_response.get("raw_response") if last_response else None),
                        error="final_before_successful_plot",
                    )
                    write_json(per_question_path, result)
                    return result
            break

        if last_response.get("prompt_tool_request_invalid"):
            if screenshot_limit > 0 and len(screenshot_paths) >= screenshot_limit:
                append_user_message_trace(
                    messages,
                    trace,
                    step=step,
                    text=(
                        f"Screenshot limit reached ({screenshot_limit}). You cannot request another desmos_plot screenshot. "
                        "ONLY reason over the visible evidence from the screenshots and images you already have. "
                        "If that evidence is still insufficient, use selected_option \"N/A\" and briefly explain the visual gap in errors. "
                        "Reply ONLY with a final answer in this format:\n"
                        f"{FINAL_RESPONSE_EXACT}"
                    ),
                    output_root=output_root,
                    event="follow_up_prompt",
                    extra_fields={
                        "screenshot_budget_exhausted": True,
                        "max_screenshots_per_question": screenshot_limit,
                        "screenshots_so_far": len(screenshot_paths),
                        "question_budget_exhausted": question_budget_exhausted,
                    },
                )
                if should_stop_now():
                    break
                continue
            tool_request_error_code = str(last_response.get("prompt_tool_request_error_code") or "").strip() or None
            tool_request_error_detail = str(last_response.get("prompt_tool_request_error_detail") or "").strip() or None
            repair_prompt = format_tool_repair_text(tool_request_error_code, tool_request_error_detail)
            append_user_message_trace(
                messages,
                trace,
                step=step,
                text=repair_prompt,
                output_root=output_root,
                event="follow_up_prompt",
                extra_fields={
                    "tool_request_error": tool_request_error_code or "invalid_tool_request",
                    "tool_request_error_detail": tool_request_error_detail,
                    "question_budget_exhausted": question_budget_exhausted,
                },
            )
            if should_stop_now():
                break
            continue

        tool_calls = last_response.get("tool_calls") or []
        if tool_calls:
            process_tool_calls(step, tool_calls)
            if should_stop_now():
                break
            continue

        if assistant_text:
            # Check if assistant_text contains final response prefix anywhere (not just at start)
            # Use helper to find final prefix, ensuring no tool prefix exists
            final_prefix_idx = find_prefix_in_text(assistant_text, FINAL_RESPONSE_PREFIX, forbidden_prefix=TOOL_REQUEST_PREFIX)
            
            if final_prefix_idx is not None:
                if screenshot_paths or not require_successful_plot:
                    result = build_question_result(
                        final_response_text=assistant_text,
                        final_reasoning=last_response.get("reasoning"),
                        finish_reason=extract_finish_reason(last_response.get("raw_response") if last_response else None),
                    )
                    write_json(per_question_path, result)
                    return result

                # Enforce tool usage: require at least one successful plot screenshot
                # before allowing a final answer.
                append_user_message_trace(
                    messages,
                    trace,
                    step=step,
                    text=(
                        "You tried to give the final answer before any successful desmos_plot screenshot was produced. "
                        "Request a plot first so the final answer can be justified from visible evidence. "
                        "Reply with a tool call in this exact format: "
                        f"{TOOL_REQUEST_EXACT}"
                    ),
                    output_root=output_root,
                    event="follow_up_prompt",
                    extra_fields={
                        "final_rejected_missing_screenshot": True,
                        "question_budget_exhausted": question_budget_exhausted,
                    },
                )
                if should_stop_now():
                    break
                continue

        append_user_message_trace(
            messages,
            trace,
            step=step,
            text=final_nudge_text(
                max_completion_tokens_per_question,
                screenshot_limit,
                len(screenshot_paths),
            ),
            output_root=output_root,
            event="follow_up_prompt",
            extra_fields={
                "question_budget_exhausted": question_budget_exhausted,
                "max_screenshots_per_question": screenshot_limit or None,
                "screenshots_so_far": len(screenshot_paths),
                "screenshot_budget_exhausted": bool(screenshot_limit > 0 and len(screenshot_paths) >= screenshot_limit),
            },
        )
        if should_stop_now():
            break

    result = build_question_result(
        final_response_text=last_response.get("text", "") if last_response else None,
        final_reasoning=last_response.get("reasoning") if last_response else None,
        finish_reason=extract_finish_reason(last_response.get("raw_response") if last_response else None),
    )
    result["error"] = "completion_token_budget_exceeded" if question_budget_exhausted else "not_completed"
    write_json(per_question_path, result)
    return result


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("The input dataset must be a JSON list of question objects")
    return data


def resolve_output_paths(
    raw_out: str,
    model: str,
    max_completion_tokens_per_question: int,
    max_screenshots_per_question: int,
    reasoning_level: Optional[str],
    temperature: float,
    top_p: float,
    deterministic: bool,
) -> Tuple[Path, Path]:
    safe_model = sanitize_filename_component(model)
    max_tokens = max(0, int(max_completion_tokens_per_question))
    max_screenshots = max(0, int(max_screenshots_per_question))
    reasoning = sanitize_filename_component((reasoning_level or "none").strip().lower() or "none")
    temp_compact = sanitize_filename_component(f"{float(temperature):g}")
    top_p_compact = sanitize_filename_component(f"{float(top_p):g}")
    det_tag = "det" if deterministic else "stochastic"
    run_config_suffix = (
        f"mt{max_tokens}_ms{max_screenshots}_reasoning_{reasoning}_temp{temp_compact}_topp{top_p_compact}_{det_tag}"
    )
    output_root = Path(raw_out).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_dir = output_root / f"{safe_model}_{run_config_suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_out_path = model_dir / f"results_{safe_model}.json"
    per_response_dir = model_dir / "responses"
    per_response_dir.mkdir(parents=True, exist_ok=True)
    return final_out_path, per_response_dir


def write_error_log(output_path: Path, results: List[Dict[str, Any]]) -> Path:
    output_path = output_path.resolve()
    log_path = output_path.with_name(f"{output_path.stem}_errors.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        error_text = item.get("error")
        if not error_text:
            continue
        cum_prompt = item.get("cumulative_prompt_tokens", item.get("prompt_tokens"))
        cum_completion = item.get(
            "cumulative_completion_tokens",
            item.get("completion_tokens"),
        )
        cum_total = item.get("cumulative_total_tokens", item.get("total_tokens"))
        entry = {
            "index": index,
            "id": item.get("id"),
            "qnum": item.get("qnum"),
            "error": error_text,
            "gt": item.get("gt"),
            # Explicit semantics: these totals sum usage from successful model responses
            # only (see usage_note); e.g. an HTTP failure on the next call adds no usage here.
            "usage_note": (
                "prompt_tokens/completion_tokens/total_tokens below are cumulative for this "
                "question over successful /chat/completions responses only—they exclude "
                "any failing request (e.g. HTTP 400) and are not a single-call tally."
            ),
            "cumulative_prompt_tokens": cum_prompt,
            "cumulative_completion_tokens": cum_completion,
            "cumulative_total_tokens": cum_total,
            "last_step_prompt_tokens": item.get("last_step_prompt_tokens"),
            "last_step_completion_tokens": item.get("last_step_completion_tokens"),
            "last_step_total_tokens": item.get("last_step_total_tokens"),
            # Short names kept for backwards compatibility (same values as cumulative_*).
            "prompt_tokens": item.get("prompt_tokens"),
            "completion_tokens": item.get("completion_tokens"),
            "total_tokens": item.get("total_tokens"),
        }
        lines.append(json.dumps(json_safe(entry), ensure_ascii=False, sort_keys=True))

    text = "\n".join(lines)
    if text:
        text += "\n"
    log_path.write_text(text, encoding="utf-8")
    return log_path


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def is_completed_record(record: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False
    return isinstance(record.get("final_response_text"), str) or isinstance(record.get("response_text"), str)


def resolve_reasoning_level(api_provider: str, reasoning_level: str) -> Optional[str]:
    requested_level = (reasoning_level or "none").strip().lower() or "none"
    level = normalize_reasoning_level(requested_level)
    known_reasoning_values = {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "min",
        "x-high",
        "very-high",
        "very_high",
        "no",
        "off",
        "on",
        "enable",
        "disable",
        "disabled",
        "enabled",
    }
    if requested_level not in known_reasoning_values:
        print(
            "Ignoring unsupported reasoning level "
            f"'{reasoning_level}'; expected one of "
            "none/minimal/low/medium/high/xhigh (or on/off alias). Falling back to none."
        )
        level = normalize_reasoning_level("none")

    if api_provider not in {"openrouter", "vllm", "llamacpp"}:
        print(
            f"Ignoring --reasoning-level for api_provider={api_provider}; supported for openrouter, vllm, and llamacpp only."
        )
        return None

    if api_provider == "vllm" and level != "none":
        print(
            "Note: vLLM reasoning output requires the server to be launched with a compatible "
            "--reasoning-parser."
        )

    return level


def resolve_base_url(api_provider: str, base_url: str) -> str:
    resolved = (base_url or DEFAULT_BASE_URLS.get(api_provider, "")).rstrip("/")
    if not resolved.endswith("/v1"):
        resolved = f"{resolved}/v1"
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VAMPS Desmos agent on a dataset and save raw results."
    )
    parser.add_argument("--input", required=True, help="Path to the input dataset JSON file.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--model-id", required=True, help="Model identifier for the target endpoint.")
    parser.add_argument("--api-key", required=True, help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--desmos-api-key", required=True, help="Desmos API key.")
    parser.add_argument(
        "--api-provider",
        default="openrouter",
        choices=["llamacpp", "openrouter", "vllm"],
        help="OpenAI-compatible API provider name used for request shaping and retries.",
    )
    parser.add_argument(
        "--openrouter-provider",
        type=str,
        default="",
        help="Comma-separated OpenRouter provider order, e.g. 'OpenAI,Anthropic'.",
    )
    parser.add_argument(
        "--openrouter-allow-fallbacks",
        action="store_true",
        help="Allow OpenRouter fallback providers when provider order is set.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Custom chat completions base URL. If empty, the provider default is used.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability mass.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--min-p", type=float, default=None, help="Minimum probability threshold sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Optional seed for supported providers.")
    parser.add_argument(
        "--max-completion-tokens-per-question",
        "--max-total-output-tokens",
        dest="max_completion_tokens_per_question",
        type=int,
        default=4096,
        help="Maximum cumulative completion tokens for each question. Set to 0 to disable the budget.",
    )
    parser.add_argument(
        "--max-screenshots-per-question",
        dest="max_screenshots_per_question",
        type=int,
        default=4,
        help=(
            "Maximum number of successful desmos_plot screenshots allowed per question. "
            "Once the limit is reached, further tool requests are rejected and the model is forced to finalize. "
            "Set to 0 to disable the screenshot limit."
        ),
    )
    parser.add_argument(
        "--require-successful-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enforce in code that at least one successful desmos_plot screenshot must exist before accepting a final answer. "
            "Disabled by default so early final answers can reveal non-visual/algebraic model behavior."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Request timeout in seconds for each model call. Disabled when omitted.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retry attempts for transient API errors.",
    )
    parser.add_argument(
        "--retry-base-sleep",
        type=float,
        default=5.0,
        help="Base backoff seconds before jitter for transient API retries.",
    )
    parser.add_argument(
        "--python-workers",
        "--parallel-workers",
        dest="parallel_workers",
        type=int,
        default=16,
        help="Number of questions to solve in parallel.",
    )
    parser.add_argument("--limit", type=int, help="Optional limit for a partial run.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Apply deterministic settings (temperature=0, top_p=1, seed default 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if results already exist.",
    )
    parser.add_argument(
        "--reasoning-level",
        type=str,
        default="none",
        help=(
            "Reasoning level to use (none/minimal/low/medium/high/xhigh). "
            "Aliases: off->none, on->high. "
            "A single canonical value is used and mapped to provider-specific request fields internally."
        ),
    )
    parser.add_argument(
        "--save-raw-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store the full raw API response payload in output files for debugging. Disabled by default.",
    )
    parser.add_argument(
        "--save-sent-messages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store the exact prompt messages sent on each turn in the trace. Disabled by default.",
    )
    parser.add_argument(
        "--save-api-payloads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store exact API request payloads (messages + max_tokens_override) per step for debugging. Disabled by default.",
    )
    parser.add_argument(
        "--save-desmos-internals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw Desmos internal sketch dumps (graph_sketches/expression_click_debug) in traces for debugging. Disabled by default.",
    )
    parser.add_argument(
        "--max-poi-on-plot",
        type=int,
        default=30,
        help=(
            "Maximum number of POI coordinate labels rendered on the screenshot. "
            "Set to 0 for unlimited (default 30)."
        ),
    )
    return parser.parse_args()


def run_agent_dataset(
    *,
    input_path: Path,
    output_path: Path,
    model_id: str,
    api_key: str,
    desmos_api_key: str,
    api_provider: str,
    base_url: str,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    min_p: Optional[float],
    seed: Optional[int],
    max_completion_tokens_per_question: int,
    max_screenshots_per_question: int,
    require_successful_plot: bool,
    request_timeout: Optional[float],
    max_retries: int,
    retry_base_sleep: float,
    parallel_workers: int,
    openrouter_provider: str,
    openrouter_allow_fallbacks: bool,
    force: bool,
    reasoning_level: str,
    save_raw_response: bool,
    save_sent_messages: bool,
    save_api_payloads: bool,
    save_desmos_internals: bool,
    max_poi_overlay_points: Optional[int],
    limit: Optional[int] = None,
    deterministic: bool = False,
) -> Dict[str, Any]:
    dataset = load_dataset(input_path)
    if limit is not None:
        dataset = dataset[:limit]

    effective_temperature = 0.0 if deterministic else temperature
    effective_top_p = 1.0 if deterministic else top_p
    effective_seed = 42 if deterministic and seed is None else seed
    reasoning_level_norm = resolve_reasoning_level(api_provider, reasoning_level)
    openrouter_provider_order = [part.strip() for part in (openrouter_provider or "").split(",") if part.strip()]

    output_path, responses_dir = resolve_output_paths(
        str(output_path),
        model_id,
        max_completion_tokens_per_question,
        max_screenshots_per_question,
        reasoning_level_norm or None,
        effective_temperature,
        effective_top_p,
        deterministic,
    )

    generate_response = make_openai_compatible_generator(
        api_provider=api_provider,
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=effective_temperature,
        top_p=effective_top_p,
        top_k=top_k,
        min_p=min_p,
        seed=effective_seed,
        request_timeout=request_timeout,
        reasoning_level=reasoning_level_norm,
        openrouter_provider_order=openrouter_provider_order,
        openrouter_allow_fallbacks=openrouter_allow_fallbacks,
        max_retries=max_retries,
        retry_base_sleep=retry_base_sleep,
    )
    tool = partial(
        desmos_plot,
        desmos_api_key,
        save_desmos_internals=bool(save_desmos_internals),
        max_poi_overlay_points=max_poi_overlay_points,
    )
    def solve_one(question: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(question.get("id") or question.get("qnum") or "unknown")
        per_question_dir = responses_dir / sanitize_filename_component(qid)
        per_question_path = per_question_dir / "result.json"
        if force and per_question_dir.exists():
            shutil.rmtree(per_question_dir, ignore_errors=True)
        existing_record = None if force else load_json_if_exists(per_question_path)
        if is_completed_record(existing_record):
            print(f"{question.get('id') or question.get('qnum') or 'unknown'} skipped (already exists)")
            return existing_record
        per_question_dir.mkdir(parents=True, exist_ok=True)
        try:
            return solve_one_question(
                generate_response=generate_response,
                desmos_tool=tool,
                dataset_path=input_path,
                output_json_path=output_path,
                responses_dir=responses_dir,
                save_raw_response=save_raw_response,
                save_sent_messages=save_sent_messages,
                save_api_payloads=save_api_payloads,
                max_completion_tokens_per_question=max_completion_tokens_per_question,
                max_screenshots_per_question=max_screenshots_per_question,
                model_id=model_id,
                api_provider=api_provider,
                base_url=base_url,
                temperature=effective_temperature,
                top_p=effective_top_p,
                top_k=top_k,
                min_p=min_p,
                seed=effective_seed,
                reasoning_level=reasoning_level_norm or None,
                openrouter_provider=openrouter_provider or None,
                openrouter_allow_fallbacks=openrouter_allow_fallbacks,
                deterministic=deterministic,
                require_successful_plot=require_successful_plot,
                question=question,
            )
        except Exception as exc:
            error_text = f"question_failed: {type(exc).__name__}: {exc}"
            print(f"{qid} failed: {error_text}", file=sys.stderr)
            failed_result = build_result_entry(
                question,
                final_response_text=None,
                final_reasoning=None,
                error=error_text,
                model=model_id,
                api_provider=api_provider,
                base_url=base_url,
                temperature=effective_temperature,
                top_p=effective_top_p,
                top_k=top_k,
                min_p=min_p,
                seed=effective_seed,
                reasoning_level=reasoning_level_norm or None,
                openrouter_provider=openrouter_provider or None,
                openrouter_allow_fallbacks=openrouter_allow_fallbacks,
                max_completion_tokens_per_question=max_completion_tokens_per_question,
                deterministic=deterministic,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            write_json(per_question_path, failed_result)
            return failed_result

    worker_count = max(1, int(parallel_workers))
    results: List[Optional[Dict[str, Any]]] = [None] * len(dataset)
    per_question_completion_budget = max(0, int(max_completion_tokens_per_question))
    if worker_count == 1:
        for index, question in enumerate(tqdm(dataset, desc="Questions")):
            result = solve_one(question)
            results[index] = result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(solve_one, question): index for index, question in enumerate(dataset)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Questions"):
                index = futures[future]
                result = future.result()
                results[index] = result

    resolved_results = [result for result in results if result is not None]
    error_log_path = write_error_log(output_path, resolved_results)

    output = {
        "meta": {
            "model": model_id,
            "api_provider": api_provider,
            "output_path": str(output_path),
            "base_url": base_url,
            "temperature": effective_temperature,
            "top_p": effective_top_p,
            "top_k": top_k,
            "min_p": min_p,
            "seed": effective_seed,
            "reasoning_level": reasoning_level_norm or None,
            "openrouter_provider": openrouter_provider or None,
            "openrouter_allow_fallbacks": bool(openrouter_allow_fallbacks),
            "max_completion_tokens_per_question": per_question_completion_budget or None,
            "max_screenshots_per_question": max(0, int(max_screenshots_per_question)) or None,
            "require_successful_plot": bool(require_successful_plot),
            "enforce_successful_plot": bool(require_successful_plot),
            "max_retries": max(1, int(max_retries)),
            "retry_base_sleep": max(0.0, float(retry_base_sleep)),
            "parallel_workers": worker_count,
            "force": bool(force),
            "save_raw_response": bool(save_raw_response),
            "save_sent_messages": bool(save_sent_messages),
            "save_api_payloads": bool(save_api_payloads),
            "save_desmos_internals": bool(save_desmos_internals),
            "max_poi_on_plot": max_poi_overlay_points,
            "limit": limit,
            "input": str(input_path),
            "error_log": str(error_log_path),
            "responses_dir": str(responses_dir),
        },
        "results": resolved_results,
    }
    write_json(output_path, output)
    return output


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        print(f"Input dataset not found: {input_path}", file=sys.stderr)
        return 1
    base_url = resolve_base_url(args.api_provider, args.base_url)

    try:
        max_poi_on_plot = max(0, int(args.max_poi_on_plot))
        max_poi_overlay_points = None if max_poi_on_plot == 0 else max_poi_on_plot
        output_payload = run_agent_dataset(
            input_path=input_path,
            output_path=output_path,
            model_id=args.model_id,
            api_key=args.api_key,
            desmos_api_key=args.desmos_api_key,
            api_provider=args.api_provider,
            base_url=base_url,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=args.top_k,
            min_p=args.min_p,
            seed=args.seed,
            max_completion_tokens_per_question=args.max_completion_tokens_per_question,
            max_screenshots_per_question=args.max_screenshots_per_question,
            require_successful_plot=bool(args.require_successful_plot),
            request_timeout=args.request_timeout,
            max_retries=max(1, int(args.max_retries)),
            retry_base_sleep=max(0.0, float(args.retry_base_sleep)),
            parallel_workers=args.parallel_workers,
            openrouter_provider=args.openrouter_provider,
            openrouter_allow_fallbacks=bool(args.openrouter_allow_fallbacks),
            force=bool(args.force),
            reasoning_level=args.reasoning_level,
            save_raw_response=bool(args.save_raw_response),
            save_sent_messages=bool(args.save_sent_messages),
            save_api_payloads=bool(args.save_api_payloads),
            save_desmos_internals=bool(args.save_desmos_internals),
            max_poi_overlay_points=max_poi_overlay_points,
            limit=args.limit,
            deterministic=bool(args.deterministic),
        )
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    result_json = output_payload.get("meta", {}).get("output_path") if isinstance(output_payload, dict) else None
    if isinstance(result_json, str) and result_json:
        print(f"Result JSON: {result_json}")
    else:
        print(f"Result JSON: {output_path}")
    error_log = output_payload.get("meta", {}).get("error_log") if isinstance(output_payload, dict) else None
    if isinstance(error_log, str) and error_log:
        print(f"Error log: {error_log}")
    return 0

if __name__ == "__main__":
    main()
