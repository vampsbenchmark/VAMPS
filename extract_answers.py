import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from api_helpers import (
    DEFAULT_BASE_URLS,
    build_multimodal_message,
    call_api_with_retry,
    extract_reasoning,
    extract_response_text,
    extract_usage,
    load_b64_data_url,
    normalize_reasoning_level,
)


ANSWER_KEYS = ("answer", "final_answer", "choice", "option", "selected_option")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
LAST_STEP_TOKEN_FIELDS = (
    "last_step_prompt_tokens",
    "last_step_completion_tokens",
    "last_step_total_tokens",
)

API_STATS_FIELDS = (
    "api_called",
    "api_reasoning_level",
    "api_extracted_answer",
    "api_extraction_category",
    "api_extraction_rationale",
    "api_extraction_reasoning_output",
    "api_extraction_raw_output",
    "api_extraction_method",
    "api_extraction_error",
    "api_extraction_tokens",
    "api_extraction_raw_response",
    "api_manual_agrees",
)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def append_judge_debug_entry(debug_ctx: Optional[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    if not debug_ctx:
        return
    lock: Lock = debug_ctx["lock"]
    with lock:
        debug_ctx["entries"].append(_to_jsonable(entry))

# Scoped to one <<FINAL>> segment when json.loads fails (e.g. unescaped newlines in steps_summary).
_STRICT_FINAL_LOOSE_RES = (
    re.compile(
        r'["\']selected_option["\']\s*:\s*["\'](?P<v>[1-4]|N/A)["\']',
        flags=re.IGNORECASE,
    ),
    # JSON-ish numeric: "selected_option": 1  — (?!\d) avoids matching the leading digit of "12".
    re.compile(r'["\']selected_option["\']\s*:\s*(?P<v>[1-4])(?!\d)', flags=re.IGNORECASE),
    re.compile(r'["\']selected_option["\']\s*:\s*(?P<v>N/A)\b', flags=re.IGNORECASE),
)

ANSWER_REGEX_PATTERNS = (
    re.compile(r'["\']selected_option["\']\s*:\s*(?:"([^"\n]+)"|\'([^\'\n]+)\'|([A-Da-d1-9][A-Za-z0-9._-]*))', flags=re.IGNORECASE),
    re.compile(r'["\']answer["\']\s*:\s*(?:"([^"\n]+)"|\'([^\'\n]+)\'|([A-Da-d1-9][A-Za-z0-9._-]*))', flags=re.IGNORECASE),
    re.compile(r"\bfinal answer\s*(?:is|:)?\s*\(?([A-Da-d1-9][A-Za-z0-9._-]*)\)?", flags=re.IGNORECASE),
    re.compile(r"\bselected[_ ]option\s*(?:is|:|=)\s*\(?([A-Da-d1-9][A-Za-z0-9._-]*)\)?", flags=re.IGNORECASE),
    re.compile(r"\banswer\s*(?:is|:|=)\s*\(?([A-Da-d1-9][A-Za-z0-9._-]*)\)?", flags=re.IGNORECASE),
    re.compile(r"\boption\s*(?:is|:)?\s*\(?([A-Da-d1-9])\)?", flags=re.IGNORECASE),
)
JSON_HINT_TOKENS = tuple({key.lower() for key in ANSWER_KEYS} | {"errors", "category"})

EXTRACTION_SYSTEM_PROMPT = """\
Developer: You are a helpful but strict extractor and judge for multiple-choice model outputs.
Inputs include the question text, answer options (and possibly plot images), and the ORIGINAL model output (and possibly a trace of messages).

Your tasks:
1) Extract the option label the model selected (1, 2, 3, 4).
2) Classify the ORIGINAL model output into exactly one category.

Rules (follow strictly):
1. Do NOT solve or verify the question. Use the model output to determine what the model selected.
2. Prefer an explicit option label stated by the model (use the FINAL label if multiple are mentioned).
3. If no option label is stated but the model gives an answer value/text, map it to the matching option using the provided model response only (no additional reasoning).
4. If you cannot determine a single option, use "N/A".
5. category must be exactly one of: 
    - solved_analytically: response mainly uses algebra/calculus/symbolic or heavy computation.
    - visual_evidence_based: visual-cue-based solution is supported by screenshots or images with only light arithmetic on values directly visible from images or screenshots.
    - no_definitive_answer: response does not provide a clear final option selection.
    - other: none of the above clearly apply, explain in the 'rationale'.
6. rationale must be brief and must quote the exact substring from the model output that supports the extracted selection and category (or explain why none exists).
7. If the evidence is conflicting, incomplete, or ambiguous, do not infer; return "selected_option":"N/A" and choose "no_definitive_answer" unless the category is still clearly supported by the response.

Output format:
Return ONLY one compact JSON object with exactly these keys:
{"rationale":"<brief>","category":"<one-category>","selected_option":"<option-label-or-N/A>"}
No markdown, no code fences, no extra keys, no extra text beyond the required output.
"""

EXTRACTION_USER_CONTEXT_INSTRUCTION = (
    "Below is the ORIGINAL model output/context. Extract the selected option and classify it per the system instructions. "
    "Do not solve the question. Do not invent a label. "
)

EXTRACTION_RETRY_FEEDBACK_TEMPLATE = (
    "IMPORTANT: This is a retry. Previous failed attempt count: {failed_attempts}.\n"
    "Your previous output was:\n"
    "\"\"\"\n{previous_output}\n\"\"\"\n"
    "That output is invalid because I could not reliably extract all required fields.\n"
    "Return ONLY one compact JSON object with exactly these keys and no others:\n"
    '{{"rationale":"<brief>","category":"<one-category>","selected_option":"<label-or-N/A>"}}\n'
    "Constraints:\n"
    "- category must be one of: solved_analytically, visual_evidence_based, no_definitive_answer, other\n"
    "- rationale must be non-empty\n"
    "- selected_option must be one option label or N/A\n"
    "No markdown, no code fences, no preface, no trailing text."
)
 
BASE_BUCKET_KEYS = (
    "total_gt",
    "manually_extracted",
    "correct",
    "correct_filtered",
    "screenshot_gated",
    "api_called",
    "api_success",
    "api_failures",
    "api_manual_agrees",
    "api_manual_disagrees",
    "finish_reason_length",
)

COUNT_AVERAGE_FIELDS = (
    "tool_call_requests",
    "generated_screenshots",
)


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    text = text.strip().strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def extract_fenced_blocks(text: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)]


def extract_brace_objects(text: str) -> List[str]:
    objects: List[str] = []
    start = -1
    depth = 0
    in_str = False
    esc = False
    quote_char = ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote_char:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote_char = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start:i + 1])
                    start = -1
    return objects


def answer_from_dict(data: Dict[str, Any]) -> Optional[str]:
    lower_map = {str(k).lower(): v for k, v in data.items()}
    for key in ANSWER_KEYS:
        if key in lower_map:
            return normalize_answer(lower_map[key])
    return None


def extract_answer_regex(text: str) -> Optional[str]:
    # Return the last regex match found (prefer the most recent in the output)
    for pattern in ANSWER_REGEX_PATTERNS:
        try:
            matches = list(pattern.finditer(text))
        except re.error:
            continue
        if not matches:
            continue
        m = matches[-1]
        groups = m.groups()
        if groups:
            for value in groups:
                if value is not None:
                    return normalize_answer(value)
        return normalize_answer(m.group(0))
    return None


def iter_parse_candidates(response_text: str) -> Iterable[str]:
    # Prefer smaller structured snippets first; parsing the full response body is often the slow path.
    yield from extract_fenced_blocks(response_text)
    yield from extract_brace_objects(response_text)
    yield response_text


def should_try_parse(candidate: str) -> bool:
    stripped = candidate.strip()
    if not stripped or "{" not in stripped or "}" not in stripped:
        return False
    lowered = stripped.lower()
    return any(token in lowered for token in JSON_HINT_TOKENS)


def try_parse_candidate(candidate: str) -> Optional[Dict[str, Any]]:
    stripped = candidate.strip()
    if not should_try_parse(stripped):
        return None
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    try:
        value = ast.literal_eval(stripped)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return None


def extract_selected_option_from_final_segment_loose(segment: str) -> Optional[str]:
    """Parse strict JSON failed; pull ``selected_option`` only from this ``<<FINAL>>`` tail (1–4 or N/A).

    Accepts quoted strings, unquoted single-digit options, or unquoted N/A — still only within this segment.
    """
    candidates: List[Tuple[int, str]] = []
    for rx in _STRICT_FINAL_LOOSE_RES:
        for m in rx.finditer(segment):
            raw = m.group("v")
            val = normalize_answer(raw)
            if val is not None:
                candidates.append((m.end(), val))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def extract_best_strict_final_only(response_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """Only accept an answer from a ``<<FINAL>>`` block: strict JSON, else regex for ``selected_option`` in that block only.

    Does not scan the full transcript for options (no CoT regex). Use when CoT may mention options without
    a real final commit (e.g. responses cut off before ``<<FINAL>>``).
    """
    if not response_text:
        return None, None, "empty"
    final_parsed: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    last_final_segment: Optional[str] = None
    saw_final_tag = False
    for m in re.finditer(r"<<FINAL>>", response_text, flags=re.IGNORECASE):
        saw_final_tag = True
        start = m.end()
        next_m = re.search(r"<<FINAL>>", response_text[start:], flags=re.IGNORECASE)
        end = start + next_m.start() if next_m else len(response_text)
        sub = response_text[start:end].strip()
        last_final_segment = sub
        objs = extract_brace_objects(sub)
        parsed = None
        if objs:
            parsed = try_parse_candidate(objs[0])
        else:
            parsed = try_parse_candidate(sub)
        if parsed is not None:
            final_parsed = parsed
            final_answer = answer_from_dict(parsed)

    if final_answer is not None:
        return final_answer, final_parsed, "json"
    if last_final_segment:
        loose = extract_selected_option_from_final_segment_loose(last_final_segment)
        if loose is not None:
            return loose, final_parsed, "strict_final_relaxed"
    # If the model forgot the <<FINAL>> tag entirely but produced a fenced JSON object,
    # accept it as a last resort without scanning the whole transcript for option-like text.
    if not saw_final_tag:
        for block in extract_fenced_blocks(response_text):
            objs = extract_brace_objects(block)
            parsed = None
            if objs:
                parsed = try_parse_candidate(objs[0])
            else:
                parsed = try_parse_candidate(block)
            if parsed is not None:
                ans = answer_from_dict(parsed)
                if ans is not None:
                    return ans, parsed, "strict_final_fenced_json"
    return None, final_parsed, "none"


def extract_best(response_text: str, *, strict_final: bool = False) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    if not response_text:
        return None, None, "empty"
    if strict_final:
        return extract_best_strict_final_only(response_text)
    # First, prefer explicit <<FINAL>> blocks: search occurrences and try to parse JSON/objects
    final_parsed = None
    final_answer = None
    for m in re.finditer(r"<<FINAL>>", response_text, flags=re.IGNORECASE):
        start = m.end()
        # limit to next <<FINAL>> if present
        next_m = re.search(r"<<FINAL>>", response_text[start:], flags=re.IGNORECASE)
        end = start + next_m.start() if next_m else len(response_text)
        sub = response_text[start:end].strip()
        objs = extract_brace_objects(sub)
        parsed = None
        if objs:
            parsed = try_parse_candidate(objs[0])
        else:
            parsed = try_parse_candidate(sub)
        if parsed is not None:
            final_parsed = parsed
            final_answer = answer_from_dict(parsed)

    if final_parsed is not None:
        return final_answer, final_parsed, "json"

    # No <<FINAL>> results parsed; prefer the last successful parsed JSON-like snippet
    last_parsed = None
    last_answer = None
    for candidate in iter_parse_candidates(response_text):
        key = candidate.strip()
        if not key:
            continue
        parsed = try_parse_candidate(key)
        if parsed is not None:
            last_parsed = parsed
            last_answer = answer_from_dict(parsed)

    if last_parsed is not None:
        return last_answer, last_parsed, "json"

    # Fallback: regex extraction (already returns last match)
    answer = extract_answer_regex(response_text)
    if answer is not None:
        return answer, None, "regex"
    return None, None, "none"


def get_token_value(record: Dict[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clear_legacy_fields(record: Dict[str, Any]) -> None:
    record.pop("parsed", None)
    record.pop("extracted_answer", None)


def clear_api_fields(record: Dict[str, Any]) -> None:
    for field in API_STATS_FIELDS:
        record.pop(field, None)
    # Backward compatibility cleanup for old field names.
    record.pop("api_pred_agrees", None)


def sync_api_manual_agreement_fields(record: Dict[str, Any], agrees: Optional[bool]) -> None:
    """Whether local extract_best output matches the judge option (this run or embedded).

    ``api_pred_agrees`` is a legacy duplicate of ``api_manual_agrees``; keep them in sync.
    """
    if agrees is None:
        record.pop("api_manual_agrees", None)
        record.pop("api_pred_agrees", None)
    else:
        record["api_manual_agrees"] = agrees
        record["api_pred_agrees"] = agrees


def record_has_api_judge(record: Dict[str, Any]) -> bool:
    """True if this row was processed with API-assisted judge (category applies to filtered accuracy)."""
    if bool(record.get("api_called")):
        return True
    if record.get("api_extraction_method"):
        return True
    cat = record.get("api_extraction_category")
    if cat is not None and str(cat).strip():
        return True
    err = record.get("api_extraction_error")
    if err is not None and str(err).strip():
        return True
    return False


def api_state_changed(original_api_state: Dict[str, Any], record: Dict[str, Any]) -> bool:
    return any(original_api_state[field] != record.get(field) for field in API_STATS_FIELDS)


def build_summary(
    *,
    method: str,
    manually_extracted: bool,
    gt: Optional[str],
    prompt_tokens: Optional[float],
    completion_tokens: Optional[float],
    total_tokens: Optional[float],
    cumulative_prompt_tokens: Optional[float] = None,
    cumulative_completion_tokens: Optional[float] = None,
    cumulative_total_tokens: Optional[float] = None,
    last_step_prompt_tokens: Optional[float] = None,
    last_step_completion_tokens: Optional[float] = None,
    last_step_total_tokens: Optional[float] = None,
    correct: Optional[bool] = None,
    correct_filtered: bool = False,
    screenshot_gated: bool = False,
    api_called: bool = False,
    api_reasoning_level: Optional[str] = None,
    api_extracted_answer: Optional[str] = None,
    api_extraction_category: Optional[str] = None,
    api_extraction_method: Optional[str] = None,
    api_extraction_error: Optional[str] = None,
    api_extraction_tokens: Optional[int] = None,
    api_extraction_reasoning_output: Optional[Any] = None,
    api_extraction_raw_response: Optional[Dict[str, Any]] = None,
    api_extraction_raw_output: Optional[str] = None,
    api_manual_agrees: Optional[bool] = None,
    tool_call_requests: Optional[int] = None,
    generated_screenshots: Optional[int] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "method": method,
        "manually_extracted": manually_extracted,
        "gt": gt,
        "correct": correct,
        "correct_filtered": correct_filtered,
        "screenshot_gated": screenshot_gated,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cumulative_prompt_tokens": cumulative_prompt_tokens,
        "cumulative_completion_tokens": cumulative_completion_tokens,
        "cumulative_total_tokens": cumulative_total_tokens,
        "last_step_prompt_tokens": last_step_prompt_tokens,
        "last_step_completion_tokens": last_step_completion_tokens,
        "last_step_total_tokens": last_step_total_tokens,
        "api_called": api_called,
        "api_reasoning_level": api_reasoning_level,
        "api_extracted_answer": api_extracted_answer,
        "api_extraction_category": api_extraction_category,
        "api_extraction_method": api_extraction_method,
        "api_extraction_error": api_extraction_error,
        "api_extraction_tokens": api_extraction_tokens,
        "api_extraction_reasoning_output": api_extraction_reasoning_output,
        "api_extraction_raw_response": api_extraction_raw_response,
        "api_extraction_raw_output": api_extraction_raw_output,
        "category": api_extraction_category,
        "api_manual_agrees": api_manual_agrees,
        "tool_call_requests": tool_call_requests,
        "generated_screenshots": generated_screenshots,
        "finish_reason": finish_reason,
    }


def _extract_prompt_context_from_trace(trace: Any) -> Dict[str, Any]:
    """Best-effort extraction of question/options/images from initial prompt trace."""
    out: Dict[str, Any] = {
        "question_text": None,
        "question_images": [],
        "options": [],
    }
    if not isinstance(trace, list):
        return out

    initial_prompt = None
    for event in trace:
        if isinstance(event, dict) and event.get("event") == "initial_prompt":
            initial_prompt = event
            break
    if not isinstance(initial_prompt, dict):
        return out

    messages = initial_prompt.get("messages")
    if not isinstance(messages, list):
        return out

    user_message = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg
            break
    if not isinstance(user_message, dict):
        return out

    content = user_message.get("content")
    if not isinstance(content, list):
        return out

    in_options = False
    current_option: Optional[Dict[str, Any]] = None
    label_pattern = re.compile(r"^\s*([A-Za-z0-9]+)\.\s*$")

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()

        if ptype == "text":
            text = str(part.get("text") or "")
            if text.startswith("Question:"):
                out["question_text"] = text[len("Question:") :].strip()
                in_options = False
                continue
            if text.startswith("Summary of the rules:"):
                continue
            if text.strip() == "Options:":
                in_options = True
                continue

            # Handle compact text-only options: "Options:\n1. ...\n2. ..."
            if text.startswith("Options:\n"):
                in_options = True
                for raw_line in text.splitlines()[1:]:
                    line = raw_line.strip()
                    if not line:
                        continue
                    m = re.match(r"^([A-Za-z0-9]+)\.\s*(.*)$", line)
                    if not m:
                        continue
                    label = m.group(1)
                    opt_text = m.group(2).strip()
                    option_obj: Dict[str, Any] = {"label": label}
                    if opt_text:
                        option_obj["text"] = opt_text
                    out["options"].append(option_obj)
                continue

            if in_options:
                m = label_pattern.match(text)
                if m:
                    current_option = {"label": m.group(1)}
                    out["options"].append(current_option)
                elif current_option is not None:
                    maybe_text = text.strip()
                    if maybe_text:
                        current_option["text"] = maybe_text
            continue

        if ptype == "image":
            img_path = part.get("path")
            if not isinstance(img_path, str) or not img_path.strip():
                continue
            image_obj: Dict[str, Any] = {"path": img_path}
            mime = part.get("mime")
            if isinstance(mime, str) and mime.strip():
                image_obj["content_type"] = mime
            if in_options and current_option is not None:
                current_option["image"] = image_obj
            elif not in_options:
                out["question_images"].append(image_obj)

    return out


def _trace_to_multimodal_parts(trace: Any, base_dir: Path) -> List[Dict[str, Any]]:
    """Convert trace events to multimodal parts with consistent event labels."""
    if not isinstance(trace, list):
        return []

    out_parts: List[Dict[str, Any]] = []
    last_step: Any = object()
    for event in trace:
        if not isinstance(event, dict):
            continue

        pending_image_parts: List[Dict[str, Any]] = []

        step = event.get("step")
        if isinstance(step, int) and step != last_step:
            out_parts.append({"type": "text", "text": f"step: {step}"})
            last_step = step

        model_reasoning = event.get("model_reasoning")
        if isinstance(model_reasoning, str) and model_reasoning.strip():
            out_parts.append(
                {
                    "type": "text",
                    "text": f"model_reasoning: {model_reasoning.strip()}",
                }
            )

        model_text = event.get("model_text")
        if isinstance(model_text, str) and model_text.strip():
            out_parts.append(
                {
                    "type": "text",
                    "text": f"model_response: {model_text.strip()}",
                }
            )

        follow_up = event.get("follow_up_user_prompt")
        if isinstance(follow_up, dict):
            content = follow_up.get("content")
            if isinstance(content, list):
                follow_text_chunks: List[str] = []
                for piece in content:
                    if not isinstance(piece, dict):
                        continue
                    ptype = str(piece.get("type") or "").strip().lower()
                    if ptype == "text":
                        txt = str(piece.get("text") or "").strip()
                        if txt:
                            follow_text_chunks.append(txt)
                    elif ptype == "image":
                        img_path = piece.get("path")
                        if isinstance(img_path, str) and img_path.strip():
                            try:
                                data_url = load_b64_data_url(img_path, base_dir)
                                pending_image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                            except Exception as exc:
                                pending_image_parts.append(
                                    {
                                        "type": "text",
                                        "text": f"trace_image_load_error: {img_path}: {type(exc).__name__}: {exc}",
                                    }
                                )
                if follow_text_chunks:
                    out_parts.append(
                        {
                            "type": "text",
                            "text": "follow_up_prompt: " + "\n".join(follow_text_chunks),
                        }
                    )

        tool_result = event.get("tool_result")
        if isinstance(tool_result, dict):
            out_parts.append(
                {
                    "type": "text",
                    "text": "tool_result: " + json.dumps(tool_result, ensure_ascii=False),
                }
            )

        tool_error = event.get("tool_error")
        if isinstance(tool_error, str) and tool_error.strip():
            out_parts.append(
                {
                    "type": "text",
                    "text": f"tool_error: {tool_error.strip()}",
                }
            )

        if pending_image_parts:
            out_parts.extend(pending_image_parts)

    return out_parts


def build_extraction_conversation_text(record: Dict[str, Any], include_trace: bool) -> str:
    sections: List[str] = []

    response_text = record.get("final_response_text")
    if response_text is None:
        response_text = record.get("response_text")

    # When trace is enabled, avoid duplicating the same assistant text block.
    if not include_trace and isinstance(response_text, str) and response_text.strip():
        sections.append("Model's final response:\n" + response_text.strip())

    if not sections and not include_trace:
        sections.append("Model's final response:\n")

    return "\n\n".join(sections)


def parse_api_extraction_text(api_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not api_text:
        return None, None, None

    category: Optional[str] = None
    rationale: Optional[str] = None
    parsed = try_parse_candidate(api_text)
    if parsed is not None:
        selected_option = normalize_answer(
            parsed.get("selected_option")
            or parsed.get("answer")
            or answer_from_dict(parsed)
        )
        category = normalize_answer(
            parsed.get("category")
            or parsed.get("error_category")
            or parsed.get("classification")
            or parsed.get("label")
        )
        rationale = normalize_answer(
            parsed.get("rationale")
            or parsed.get("reason")
            or parsed.get("explanation")
            or parsed.get("steps_summary")
        )
        return selected_option, category, rationale

    selected_option, parsed_obj, _ = extract_best(api_text)
    selected_option_norm = normalize_answer(selected_option)
    if parsed_obj is not None and category is None:
        category = normalize_answer(
            parsed_obj.get("category")
            or parsed_obj.get("error_category")
            or parsed_obj.get("classification")
            or parsed_obj.get("label")
        )
        rationale = normalize_answer(
            parsed_obj.get("rationale")
            or parsed_obj.get("reason")
            or parsed_obj.get("explanation")
            or parsed_obj.get("steps_summary")
        )

    fallback_selected_option = (
        selected_option_norm if selected_option_norm is not None else normalize_answer(api_text)
    )
    return fallback_selected_option, category, rationale


def _append_attempt_log(
    attempts_log: List[Dict[str, Any]],
    *,
    attempt: int,
    error: Optional[str],
    raw: Optional[Dict[str, Any]],
    response_text: Optional[str],
    parsed_answer: Optional[str],
    parsed_category: Optional[str],
    parsed_rationale: Optional[str],
    reasoning_output: Optional[Any],
    accepted: bool,
    usage: Optional[Dict[str, Any]] = None,
) -> None:
    attempts_log.append(
        {
            "attempt": attempt,
            "error": error,
            "raw_response": raw,
            "response_text": response_text,
            "parsed_answer": parsed_answer,
            "parsed_category": parsed_category,
            "parsed_rationale": parsed_rationale,
            "reasoning_output": reasoning_output,
            "accepted": accepted,
            "usage": usage if usage is not None else (extract_usage(raw) if raw else {}),
        }
    )


def _append_judge_retry_feedback(
    messages: Any,
    *,
    failed_attempts: int,
    previous_output: str,
) -> None:
    if not isinstance(messages, list) or len(messages) < 2:
        return
    user_msg = messages[1]
    if not isinstance(user_msg, dict):
        return
    content = user_msg.get("content")
    if not isinstance(content, list):
        return

    # Keep this as plain text so it works across providers/message adapters.
    retry_text = EXTRACTION_RETRY_FEEDBACK_TEMPLATE.format(
        failed_attempts=failed_attempts,
        previous_output=previous_output,
    )
    content.append({"type": "text", "text": retry_text})


def run_api_extraction(
    record: Dict[str, Any],
    api_cfg: Dict[str, Any],
    debug_ctx: Optional[Dict[str, Any]] = None,
    debug_index: Optional[int] = None,
) -> Optional[str]:
    record["api_called"] = True
    reasoning_level = normalize_reasoning_level(api_cfg.get("reasoning_level"))
    record["api_reasoning_level"] = reasoning_level

    base_dir = Path(api_cfg.get("data_dir", "."))
    judge_parse_retries = max(0, int(api_cfg.get("judge_parse_retries", 1)))
    max_attempts = 1 + judge_parse_retries
    include_trace = bool(api_cfg.get("include_trace", True))

    question_text = record.get("question_text") or ""
    question_images = record.get("question_images")
    options = record.get("options")

    trace_context = _extract_prompt_context_from_trace(record.get("trace"))
    if not question_text and isinstance(trace_context.get("question_text"), str):
        question_text = trace_context.get("question_text") or ""
    if not question_images and trace_context.get("question_images"):
        question_images = trace_context.get("question_images")
    if not options and trace_context.get("options"):
        options = trace_context.get("options")

    # When trace is disabled, still provide question/options reconstructed from trace if needed.
    if not include_trace:
        if isinstance(trace_context.get("question_text"), str) and trace_context.get("question_text"):
            question_text = trace_context.get("question_text")
        if trace_context.get("question_images"):
            question_images = trace_context.get("question_images")
        if trace_context.get("options"):
            options = trace_context.get("options")

    extraction_context_text = build_extraction_conversation_text(
        record,
        include_trace=include_trace,
    )
    
    messages = build_multimodal_message(
        system_prompt=f"{EXTRACTION_SYSTEM_PROMPT}",
        question_text=question_text,
        question_images=question_images,
        visualization_images=None,
        options=options,
        include_options=True,
        base_dir=base_dir,
        user_instruction="",
    )

    if isinstance(messages, list) and len(messages) >= 2:
        user_msg = messages[1]
        if isinstance(user_msg, dict) and isinstance(user_msg.get("content"), list):
            instruction_text = EXTRACTION_USER_CONTEXT_INSTRUCTION
            if extraction_context_text.strip():
                instruction_text = f"{instruction_text}\n\n{extraction_context_text}"
            user_msg["content"].append({"type": "text", "text": instruction_text})
            if include_trace:
                user_msg["content"].extend(_trace_to_multimodal_parts(record.get("trace"), base_dir))

    api_pred: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    total_api_tokens = 0
    final_api_category: Optional[str] = None
    final_api_rationale: Optional[str] = None
    final_api_reasoning_output: Optional[Any] = None
    final_api_raw_output: Optional[str] = None
    final_api_text: Optional[str] = None
    accepted = False
    attempts_log: List[Dict[str, Any]] = []

    for attempt_idx in range(max_attempts):
        raw, error = call_api_with_retry(
            messages,
            api_cfg["provider"],
            api_cfg["base_url"],
            api_cfg["api_key"],
            api_cfg["model"],
            temperature=api_cfg.get("temperature", 0.0),
            top_p=api_cfg.get("top_p"),
            top_k=api_cfg.get("top_k"),
            min_p=api_cfg.get("min_p"),
            max_tokens=api_cfg.get("max_tokens", 16384),
            request_timeout=api_cfg.get("timeout"),
            seed=api_cfg.get("seed"),
            reasoning_level=reasoning_level,
            openrouter_provider_order=api_cfg.get("openrouter_provider_order"),
            openrouter_allow_fallbacks=api_cfg.get("openrouter_allow_fallbacks", False),
            max_retries=api_cfg.get("max_retries", 3),
            retry_base_sleep=api_cfg.get("retry_base_sleep", 1.0),
        )

        if not raw or error is not None:
            _append_attempt_log(
                attempts_log,
                attempt=attempt_idx + 1,
                error=error,
                raw=raw,
                response_text=None,
                parsed_answer=None,
                parsed_category=None,
                parsed_rationale=None,
                reasoning_output=None,
                accepted=False,
            )
            continue

        usage = extract_usage(raw)
        reasoning_output = extract_reasoning(raw)
        final_api_reasoning_output = reasoning_output
        if usage.get("total_tokens"):
            total_api_tokens += int(usage["total_tokens"])

        choice = (raw.get("choices") or [None])[0]
        final_api_text = extract_response_text(choice)
        if not final_api_text:
            _append_attempt_log(
                attempts_log,
                attempt=attempt_idx + 1,
                error=error,
                raw=raw,
                response_text=None,
                parsed_answer=None,
                parsed_category=None,
                parsed_rationale=None,
                reasoning_output=reasoning_output,
                accepted=False,
                usage=usage,
            )
            continue

        final_api_raw_output = final_api_text
        parsed_answer, parsed_category, parsed_rationale = parse_api_extraction_text(final_api_text)
        api_pred = parsed_answer
        final_api_category = parsed_category
        final_api_rationale = parsed_rationale
        # Accept when required fields are recoverable, even if the provider wraps JSON in extra text.
        accepted = bool(
            parsed_answer is not None
            and parsed_category is not None
            and parsed_rationale is not None
        )

        _append_attempt_log(
            attempts_log,
            attempt=attempt_idx + 1,
            error=error,
            raw=raw,
            response_text=final_api_text,
            parsed_answer=parsed_answer,
            parsed_category=parsed_category,
            parsed_rationale=parsed_rationale,
            reasoning_output=reasoning_output,
            accepted=accepted,
            usage=usage,
        )
        if accepted:
            break

        # Retry with an augmented prompt that includes the previous invalid output.
        failed_attempts = attempt_idx + 1
        if final_api_text is not None:
            _append_judge_retry_feedback(
                messages,
                failed_attempts=failed_attempts,
                previous_output=final_api_text,
            )

    if debug_ctx is not None:
        append_judge_debug_entry(
            debug_ctx,
            {
                "provider": api_cfg.get("provider"),
                "model": api_cfg.get("model"),
                "base_url": api_cfg.get("base_url"),
                "reasoning_level": reasoning_level,
                "qnum": record.get("qnum"),
                "id": record.get("id"),
                "variant": record.get("variant"),
                "debug_index": debug_index,
                "messages": messages,
                "attempts": attempts_log,
                "max_attempts": max_attempts,
                "accepted": accepted,
                "final_api_pred": api_pred,
                "final_api_category": final_api_category,
                "final_api_rationale": final_api_rationale,
                "final_api_reasoning_output": final_api_reasoning_output,
                "final_api_raw_output": final_api_raw_output,
            },
        )

    if raw and error is None:
        if final_api_reasoning_output is not None:
            record["api_extraction_reasoning_output"] = _to_jsonable(final_api_reasoning_output)
        else:
            record.pop("api_extraction_reasoning_output", None)

        if final_api_raw_output is not None:
            record["api_extraction_raw_output"] = final_api_raw_output
        else:
            record.pop("api_extraction_raw_output", None)

        if api_cfg.get("store_raw_response", False):
            # Optional: keep full provider payload for debugging/auditing.
            record["api_extraction_raw_response"] = raw
        else:
            record.pop("api_extraction_raw_response", None)

        if accepted:
            if final_api_category is not None:
                record["api_extraction_category"] = final_api_category
                record["api_extraction_rationale"] = final_api_rationale
            else:
                record.pop("api_extraction_category", None)
                record.pop("api_extraction_rationale", None)
            record["api_extraction_method"] = "api_success"
            record.pop("api_extraction_error", None)
        elif final_api_text:
            api_pred = "N/A"
            record.pop("api_extraction_category", None)
            record.pop("api_extraction_rationale", None)
            record["api_extraction_raw_output"] = final_api_raw_output
            record["api_extraction_method"] = "api_invalid_format"
            record["api_extraction_error"] = (
                "Extractor output was missing one or more required fields after "
                f"{max_attempts} attempt(s)."
            )
        else:
            record.pop("api_extraction_category", None)
            record.pop("api_extraction_rationale", None)
            record.pop("api_extraction_raw_output", None)
            record["api_extraction_method"] = "api_no_content"

        if total_api_tokens > 0:
            record["api_extraction_tokens"] = total_api_tokens
    else:
        record.pop("api_extraction_raw_response", None)
        record.pop("api_extraction_category", None)
        record.pop("api_extraction_rationale", None)
        record.pop("api_extraction_reasoning_output", None)
        record.pop("api_extraction_raw_output", None)
        record["api_extraction_method"] = "api_failed"
        record["api_extraction_error"] = error

    # Never write None: keeps a prior embedded judge answer if this run failed before returning an option.
    if api_pred is not None:
        record["api_extracted_answer"] = api_pred
    return api_pred


def _count_generated_screenshots(record: Dict[str, Any]) -> int:
    screenshots = record.get("screenshots")
    if isinstance(screenshots, list):
        return sum(1 for item in screenshots if isinstance(item, str) and item.strip())
    return 0


def _count_tool_call_requests(record: Dict[str, Any]) -> int:
    trace = record.get("trace")
    if not isinstance(trace, list):
        return 0

    total = 0
    for event in trace:
        if not isinstance(event, dict):
            continue
        tool_calls = event.get("tool_calls")
        if isinstance(tool_calls, list):
            total += sum(1 for item in tool_calls if isinstance(item, dict))
    return total


def apply_screenshot_coverage_gate(
    record: Dict[str, Any],
    pred: Optional[str],
    *,
    min_screenshots: int,
) -> Tuple[Optional[str], bool]:
    required_screenshots = max(0, int(min_screenshots))
    if required_screenshots <= 0:
        return pred, False
    generated_screenshots = _count_generated_screenshots(record)
    if generated_screenshots >= required_screenshots:
        return pred, False

    record["screenshot_gate_required_screenshots"] = required_screenshots
    record["screenshot_gate_generated_screenshots"] = generated_screenshots
    record["screenshot_gate_triggered"] = True
    return pred, True


def update_record(
    record: Dict[str, Any],
    api_cfg: Optional[Dict[str, Any]] = None,
    min_screenshots: int = 1,
    debug_ctx: Optional[Dict[str, Any]] = None,
    debug_index: Optional[int] = None,
    strict_final: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    original_extraction_method = record.get("extraction_method")
    original_correct = record.get("correct")
    original_pred = record.get("pred")
    original_api_state = {field: record.get(field) for field in API_STATS_FIELDS}

    had_parsed = "parsed" in record
    had_extracted_answer = "extracted_answer" in record

    response_text = record.get("final_response_text")
    if response_text is None:
        response_text = record.get("response_text")
    prompt_tokens = get_token_value(record, "prompt_tokens")
    completion_tokens = get_token_value(record, "completion_tokens")
    total_tokens = get_token_value(record, "total_tokens")
    cumulative_prompt_tokens = get_token_value(record, "cumulative_prompt_tokens")
    cumulative_completion_tokens = get_token_value(record, "cumulative_completion_tokens")
    cumulative_total_tokens = get_token_value(record, "cumulative_total_tokens")
    last_step_prompt_tokens = get_token_value(record, "last_step_prompt_tokens")
    last_step_completion_tokens = get_token_value(record, "last_step_completion_tokens")
    last_step_total_tokens = get_token_value(record, "last_step_total_tokens")
    gt = normalize_answer(record.get("gt"))
    tool_call_requests = _count_tool_call_requests(record)
    generated_screenshots = _count_generated_screenshots(record)

    early_method = None
    if not isinstance(response_text, str):
        early_method = "none"

    if early_method is not None:
        record["pred"] = None
        clear_legacy_fields(record)
        record["extraction_method"] = early_method
        record.pop("correct", None)
        sync_api_manual_agreement_fields(record, None)
        if api_cfg is not None:
            clear_api_fields(record)
        changed = (
            original_extraction_method != early_method
            or "correct" in record and original_correct is not None
            or original_pred is not None
            or had_parsed
            or had_extracted_answer
            or api_state_changed(original_api_state, record)
        )
        return build_summary(
            method="none",
            manually_extracted=False,
            gt=gt,
            correct=None,
            correct_filtered=False,
            screenshot_gated=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cumulative_prompt_tokens=cumulative_prompt_tokens,
            cumulative_completion_tokens=cumulative_completion_tokens,
            cumulative_total_tokens=cumulative_total_tokens,
            last_step_prompt_tokens=last_step_prompt_tokens,
            last_step_completion_tokens=last_step_completion_tokens,
            last_step_total_tokens=last_step_total_tokens,
            tool_call_requests=tool_call_requests,
            generated_screenshots=generated_screenshots,
            finish_reason=record.get("finish_reason"),
        ), changed

    answer, _, method = extract_best(response_text, strict_final=strict_final)
    local_pred = normalize_answer(answer)
    manually_extracted = local_pred is not None and method in {
        "json",
        "regex",
        "strict_final_relaxed",
        "strict_final_fenced_json",
    }

    api_pred = None
    api_called = False
    if api_cfg is not None:
        api_called = True
        api_pred = run_api_extraction(record, api_cfg, debug_ctx=debug_ctx, debug_index=debug_index)
    # Omit clear_api_fields: keep embedded API judge fields when not re-invoking (--api-extraction off).

    pred = local_pred
    final_method = method
    if api_called and api_pred is not None:
        if api_cfg.get("prefer_api", False): #or local_pred is None:
            pred = api_pred
            final_method = "api_selected"

    pred, screenshot_gated = apply_screenshot_coverage_gate(
        record,
        pred,
        min_screenshots=min_screenshots,
    )

    # Local extract_best vs judge option: prefer this run's api_pred; else embedded api_extracted_answer
    # (e.g. judge run failed but prior labels remain; or no --api-extraction and friend's file).
    if api_called and api_pred is not None:
        judge_answer = normalize_answer(api_pred)
    else:
        judge_answer = normalize_answer(record.get("api_extracted_answer"))
    if local_pred is not None and judge_answer is not None:
        sync_api_manual_agreement_fields(record, local_pred == judge_answer)
    else:
        sync_api_manual_agreement_fields(record, None)

    record["pred"] = pred
    clear_legacy_fields(record)
    record["extraction_method"] = final_method
    if strict_final:
        record["extraction_strict_final"] = True
    else:
        record.pop("extraction_strict_final", None)

    correct = None
    if gt is not None and pred is not None:
        correct = gt == pred
        record["correct"] = correct
    else:
        record.pop("correct", None)

    required_screenshots = max(0, int(min_screenshots))
    category = normalize_answer(record.get("api_extraction_category"))
    is_visual_evidence_based = bool(
        isinstance(category, str) and category.lower() == "visual_evidence_based"
    )
    meets_screenshot_requirement = generated_screenshots >= required_screenshots
    # Require visual-evidence category when this run invoked the judge OR embedded judge metadata remains.
    needs_visual_category = record_has_api_judge(record)
    passes_visual_category = (not needs_visual_category) or is_visual_evidence_based
    correct_filtered = bool(
        correct is True and passes_visual_category and meets_screenshot_requirement
    )

    changed = (
        original_extraction_method != final_method
        or original_correct != record.get("correct")
        or original_pred != pred
        or had_parsed
        or had_extracted_answer
        or api_state_changed(original_api_state, record)
    )

    return build_summary(
        method=final_method,
        manually_extracted=manually_extracted,
        gt=gt,
        correct=correct,
        correct_filtered=correct_filtered,
        screenshot_gated=screenshot_gated,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cumulative_prompt_tokens=cumulative_prompt_tokens,
        cumulative_completion_tokens=cumulative_completion_tokens,
        cumulative_total_tokens=cumulative_total_tokens,
        last_step_prompt_tokens=last_step_prompt_tokens,
        last_step_completion_tokens=last_step_completion_tokens,
        last_step_total_tokens=last_step_total_tokens,
        api_called=bool(api_called or record.get("api_called")),
        api_reasoning_level=record.get("api_reasoning_level"),
        api_extracted_answer=record.get("api_extracted_answer"),
        api_extraction_category=record.get("api_extraction_category"),
        api_extraction_method=record.get("api_extraction_method"),
        api_extraction_error=record.get("api_extraction_error"),
        api_extraction_tokens=record.get("api_extraction_tokens"),
        api_extraction_reasoning_output=record.get("api_extraction_reasoning_output"),
        api_extraction_raw_response=record.get("api_extraction_raw_response"),
        api_extraction_raw_output=record.get("api_extraction_raw_output"),
        api_manual_agrees=record.get("api_manual_agrees", record.get("api_pred_agrees")),
        tool_call_requests=tool_call_requests,
        generated_screenshots=generated_screenshots,
        finish_reason=record.get("finish_reason"),
    ), changed


def make_stats_bucket() -> Dict[str, Any]:
    return {
        "total_gt": 0,
        "manually_extracted": 0,
        "correct": 0,
        "correct_filtered": 0,
        "screenshot_gated": 0,
        "accuracy": 0.0,
        "accuracy_filtered": 0.0,
        "extraction_methods": {},
        "category_counts": {},
        "prompt_tokens_sum": 0.0,
        "completion_tokens_sum": 0.0,
        "total_tokens_sum": 0.0,
        "prompt_tokens_avg": None,
        "completion_tokens_avg": None,
        "total_tokens_avg": None,
        "prompt_tokens_max": None,
        "completion_tokens_max": None,
        "total_tokens_max": None,
        "prompt_tokens_count": 0,
        "completion_tokens_count": 0,
        "total_tokens_count": 0,
        "last_step_prompt_tokens_sum": 0.0,
        "last_step_completion_tokens_sum": 0.0,
        "last_step_total_tokens_sum": 0.0,
        "last_step_prompt_tokens_avg": None,
        "last_step_completion_tokens_avg": None,
        "last_step_total_tokens_avg": None,
        "last_step_prompt_tokens_max": None,
        "last_step_completion_tokens_max": None,
        "last_step_total_tokens_max": None,
        "last_step_prompt_tokens_count": 0,
        "last_step_completion_tokens_count": 0,
        "last_step_total_tokens_count": 0,
        "api_called": 0,
        "api_success": 0,
        "api_failures": 0,
        "api_manual_agrees": 0,
        "api_manual_disagrees": 0,
        "tool_call_requests_sum": 0.0,
        "generated_screenshots_sum": 0.0,
        "tool_call_requests_count": 0,
        "generated_screenshots_count": 0,
        "tool_call_requests_avg": None,
        "generated_screenshots_avg": None,
        "tool_call_requests_max": None,
        "generated_screenshots_max": None,
        "finish_reason_length": 0,
    }


def update_stats_bucket(bucket: Dict[str, Any], summary: Dict[str, Any]) -> None:
    if summary.get("manually_extracted"):
        bucket["manually_extracted"] += 1

    gt = summary.get("gt")
    correct = summary.get("correct")
    if gt is not None:
        bucket["total_gt"] += 1
        if correct:
            bucket["correct"] += 1
    if summary.get("correct_filtered") is True:
        bucket["correct_filtered"] += 1
    if summary.get("screenshot_gated") is True:
        bucket["screenshot_gated"] += 1
    method = summary.get("method") or "none"
    bucket["extraction_methods"][method] = bucket["extraction_methods"].get(method, 0) + 1
    category = normalize_answer(summary.get("category"))
    if category is not None:
        bucket["category_counts"][category] = bucket["category_counts"].get(category, 0) + 1

    # API stats
    if summary.get("api_called"):
        bucket["api_called"] += 1
        if summary.get("api_extraction_method") == "api_success":
            bucket["api_success"] += 1
        else:
            bucket["api_failures"] += 1
    if summary.get("api_manual_agrees") is True:
        bucket["api_manual_agrees"] += 1
    if summary.get("api_manual_agrees") is False:
        bucket["api_manual_disagrees"] += 1

    # Count finish_reason == 'length' (token limit reached)
    fr = summary.get("finish_reason")
    if isinstance(fr, str) and fr.lower() == "length":
        bucket["finish_reason_length"] += 1

    for field in TOKEN_FIELDS:
        value = summary.get(field)
        if value is None:
            continue
        bucket[f"{field}_sum"] += value
        bucket[f"{field}_count"] += 1
        current_max = bucket.get(f"{field}_max")
        bucket[f"{field}_max"] = value if current_max is None else max(current_max, value)
    for field in LAST_STEP_TOKEN_FIELDS:
        value = summary.get(field)
        if value is None:
            continue
        bucket[f"{field}_sum"] += value
        bucket[f"{field}_count"] += 1
        current_max = bucket.get(f"{field}_max")
        bucket[f"{field}_max"] = value if current_max is None else max(current_max, value)
    for field in COUNT_AVERAGE_FIELDS:
        value = summary.get(field)
        if value is None:
            continue
        bucket[f"{field}_sum"] += value
        bucket[f"{field}_count"] += 1
        current_max = bucket.get(f"{field}_max")
        bucket[f"{field}_max"] = value if current_max is None else max(current_max, value)


def finalize_stats_bucket(bucket: Dict[str, Any]) -> None:
    total = bucket.get("total_gt", 0)
    bucket["accuracy"] = (bucket["correct"] / total) if total else 0.0
    bucket["accuracy_filtered"] = (bucket["correct_filtered"] / total) if total else 0.0
    for field in TOKEN_FIELDS:
        count = bucket.get(f"{field}_count", 0)
        total = bucket.get(f"{field}_sum", 0.0)
        bucket[f"{field}_avg"] = (total / count) if count else None
        bucket.pop(f"{field}_count", None)
        bucket.pop(f"{field}_sum", None)
    for field in LAST_STEP_TOKEN_FIELDS:
        count = bucket.get(f"{field}_count", 0)
        total = bucket.get(f"{field}_sum", 0.0)
        bucket[f"{field}_avg"] = (total / count) if count else None
        bucket.pop(f"{field}_count", None)
        bucket.pop(f"{field}_sum", None)
    for field in COUNT_AVERAGE_FIELDS:
        count = bucket.get(f"{field}_count", 0)
        total = bucket.get(f"{field}_sum", 0.0)
        bucket[f"{field}_avg"] = (total / count) if count else None
        bucket.pop(f"{field}_count", None)
        bucket.pop(f"{field}_sum", None)


def merge_stats_bucket(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in BASE_BUCKET_KEYS:
        target[key] += source.get(key, 0)
    for method, count in source.get("extraction_methods", {}).items():
        target["extraction_methods"][method] = target["extraction_methods"].get(method, 0) + count
    for category, count in source.get("category_counts", {}).items():
        target["category_counts"][category] = target["category_counts"].get(category, 0) + count
    for field in TOKEN_FIELDS:
        target[f"{field}_sum"] += source.get(f"{field}_sum", 0.0)
        target[f"{field}_count"] += source.get(f"{field}_count", 0)
        source_max = source.get(f"{field}_max")
        target_max = target.get(f"{field}_max")
        if source_max is not None:
            target[f"{field}_max"] = source_max if target_max is None else max(target_max, source_max)
    for field in LAST_STEP_TOKEN_FIELDS:
        target[f"{field}_sum"] += source.get(f"{field}_sum", 0.0)
        target[f"{field}_count"] += source.get(f"{field}_count", 0)
        source_max = source.get(f"{field}_max")
        target_max = target.get(f"{field}_max")
        if source_max is not None:
            target[f"{field}_max"] = source_max if target_max is None else max(target_max, source_max)
    for field in COUNT_AVERAGE_FIELDS:
        target[f"{field}_sum"] += source.get(f"{field}_sum", 0.0)
        target[f"{field}_count"] += source.get(f"{field}_count", 0)
        source_max = source.get(f"{field}_max")
        target_max = target.get(f"{field}_max")
        if source_max is not None:
            target[f"{field}_max"] = source_max if target_max is None else max(target_max, source_max)


def default_output_json_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".json":
        return input_path.with_name(f"{input_path.stem}_extracted.json")
    return input_path.with_name(f"{input_path.name}_extracted.json")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    line = json.dumps(_to_jsonable(payload), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_progress_records(path: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not path.exists() or not path.is_file():
        return records

    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue

                index = row.get("index")
                record = row.get("record")
                if not isinstance(index, int) or not isinstance(record, dict):
                    continue
                records[index] = record
    except Exception:
        return {}

    return records


def summarize_existing_record(record: Dict[str, Any]) -> Dict[str, Any]:
    method = normalize_answer(record.get("extraction_method")) or "none"
    pred = normalize_answer(record.get("pred"))
    gt = normalize_answer(record.get("gt"))

    correct_value = record.get("correct")
    correct: Optional[bool]
    if isinstance(correct_value, bool):
        correct = correct_value
    elif gt is not None and pred is not None:
        correct = gt == pred
    else:
        correct = None

    generated_screenshots = _count_generated_screenshots(record)
    category = normalize_answer(record.get("api_extraction_category"))
    is_visual_evidence_based = bool(
        isinstance(category, str) and category.lower() == "visual_evidence_based"
    )
    screenshot_gated = bool(record.get("screenshot_gate_triggered"))
    required_screenshots = record.get("screenshot_gate_required_screenshots")
    if isinstance(required_screenshots, int):
        min_screenshots = max(0, required_screenshots)
    else:
        min_screenshots = 1
    meets_screenshot_requirement = generated_screenshots >= min_screenshots
    needs_visual_category = record_has_api_judge(record)
    passes_visual_category = (not needs_visual_category) or is_visual_evidence_based
    correct_filtered = bool(
        correct is True and meets_screenshot_requirement and passes_visual_category
    )

    return build_summary(
        method=method,
        manually_extracted=pred is not None
        and method in {"json", "regex", "strict_final_relaxed", "strict_final_fenced_json"},
        gt=gt,
        correct=correct,
        correct_filtered=correct_filtered,
        screenshot_gated=screenshot_gated,
        prompt_tokens=get_token_value(record, "prompt_tokens"),
        completion_tokens=get_token_value(record, "completion_tokens"),
        total_tokens=get_token_value(record, "total_tokens"),
        cumulative_prompt_tokens=get_token_value(record, "cumulative_prompt_tokens"),
        cumulative_completion_tokens=get_token_value(record, "cumulative_completion_tokens"),
        cumulative_total_tokens=get_token_value(record, "cumulative_total_tokens"),
        last_step_prompt_tokens=get_token_value(record, "last_step_prompt_tokens"),
        last_step_completion_tokens=get_token_value(record, "last_step_completion_tokens"),
        last_step_total_tokens=get_token_value(record, "last_step_total_tokens"),
        api_called=bool(record.get("api_called")),
        api_reasoning_level=record.get("api_reasoning_level"),
        api_extracted_answer=record.get("api_extracted_answer"),
        api_extraction_category=record.get("api_extraction_category"),
        api_extraction_method=record.get("api_extraction_method"),
        api_extraction_error=record.get("api_extraction_error"),
        api_extraction_tokens=record.get("api_extraction_tokens"),
        api_extraction_reasoning_output=record.get("api_extraction_reasoning_output"),
        api_extraction_raw_response=record.get("api_extraction_raw_response"),
        api_extraction_raw_output=record.get("api_extraction_raw_output"),
        api_manual_agrees=record.get("api_manual_agrees", record.get("api_pred_agrees")),
        tool_call_requests=_count_tool_call_requests(record),
        generated_screenshots=generated_screenshots,
        finish_reason=record.get("finish_reason"),
    )


def process_file(
    path: Path,
    output_path: Path,
    api_cfg: Optional[Dict[str, Any]] = None,
    min_screenshots: int = 1,
    parallel_workers: int = 1,
    debug_ctx: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    progress_jsonl_path: Optional[Path] = None,
    resume: bool = True,
    strict_final: bool = False,
) -> Tuple[int, int, Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0, make_stats_bucket()

    total = 0
    manually_extracted = 0
    changed_records = 0
    file_stats = make_stats_bucket()

    if not isinstance(data, dict):
        return 0, 0, make_stats_bucket()

    items: List[Dict[str, Any]]
    if isinstance(data.get("results"), list):
        items = [x for x in data["results"] if isinstance(x, dict)]
    else:
        items = [data]

    if limit is not None:
        max_items = max(0, int(limit))
        items = items[:max_items]

    resumed_indices = set()
    if resume and progress_jsonl_path is not None and progress_jsonl_path.exists():
        progress_records = load_progress_records(progress_jsonl_path)
        for index, resumed_record in progress_records.items():
            if 0 <= index < len(items):
                items[index].clear()
                items[index].update(resumed_record)
                resumed_indices.add(index)
        if resumed_indices:
            print(
                f"Resuming from progress log: restored {len(resumed_indices)} records from {progress_jsonl_path}",
                flush=True,
            )
        else:
            print(
                "Resume requested, but no resumable record snapshots were found in the progress log. "
                "Continuing from scratch for this run.",
                flush=True,
            )

    for index in sorted(resumed_indices):
        summary = summarize_existing_record(items[index])
        total += 1
        if summary["manually_extracted"]:
            manually_extracted += 1
        update_stats_bucket(file_stats, summary)

    pending_indices = [i for i in range(len(items)) if i not in resumed_indices]

    worker_count = max(1, int(parallel_workers or 1))
    if worker_count == 1 or len(pending_indices) <= 1:
        progress = tqdm(total=len(items), desc=f"Extracting {path.name}", unit="record", initial=total)
        for index in pending_indices:
            item = items[index]
            total += 1
            summary, changed = update_record(
                item,
                api_cfg=api_cfg,
                min_screenshots=min_screenshots,
                debug_ctx=debug_ctx,
                debug_index=index,
                strict_final=strict_final,
            )
            if changed:
                changed_records += 1
            if summary["manually_extracted"]:
                manually_extracted += 1
            update_stats_bucket(file_stats, summary)
            if progress_jsonl_path is not None:
                append_jsonl(
                    progress_jsonl_path,
                    {
                        "index": index,
                        "id": item.get("id") if isinstance(item, dict) else None,
                        "processed": total,
                        "total": len(items),
                        "changed": bool(changed),
                        "summary": summary,
                        "record": item,
                    },
                )
            progress.update(1)
        progress.close()
    else:
        progress = tqdm(total=len(items), desc=f"Extracting {path.name}", unit="record", initial=total)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    update_record,
                    items[index],
                    api_cfg,
                    min_screenshots,
                    debug_ctx,
                    index,
                    strict_final,
                ): (index, items[index])
                for index in pending_indices
            }
            for future in as_completed(futures):
                index, item = futures[future]
                summary, changed = future.result()
                total += 1
                if changed:
                    changed_records += 1
                if summary["manually_extracted"]:
                    manually_extracted += 1
                update_stats_bucket(file_stats, summary)
                if progress_jsonl_path is not None:
                    append_jsonl(
                        progress_jsonl_path,
                        {
                            "index": index,
                            "id": item.get("id") if isinstance(item, dict) else None,
                            "processed": total,
                            "total": len(items),
                            "changed": bool(changed),
                            "summary": summary,
                            "record": item,
                        },
                    )
                progress.update(1)
        progress.close()

    print(f"Writing output JSON to {output_path} ({changed_records} changed records)...", flush=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return total, manually_extracted, file_stats


def read_input_meta(input_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return {}
    relevant_keys = (
        "model",
        "api_provider",
        "reasoning_level",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "deterministic",
        "max_completion_tokens_per_question",
        "max_screenshots_per_question",
        "parallel_workers",
        "responses_dir",
        "input",
    )
    return {key: meta.get(key) for key in relevant_keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-process model result JSON files and extract answer from response_text.")
    ap.add_argument("input", help="Path to one aggregated JSON file (e.g., results_<model>.json).")
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path for transformed JSON. Default: <input_name>_extracted.json in the same directory.",
    )
    ap.add_argument("--stats-out", type=str, default="", help="Output path for stats JSON. Default: <input_dir>/extraction_stats.json")
    ap.add_argument(
        "--api-extraction",
        action="store_true",
        help=(
            "Enable API-assisted option extraction. When omitted, existing per-record judge fields "
            "(api_extraction_category, api_extracted_answer, ...) are preserved and used for filtered accuracy "
            "if already present."
        ),
    )
    ap.add_argument("--api-provider", choices=["llamacpp", "vllm", "openrouter"], default="openrouter", help="API provider for extraction.")
    ap.add_argument("--api-model", type=str, default="", help="[REQUIRED if --api-extraction] Model identifier for API extraction.")
    ap.add_argument("--api-key", type=str, default="", help="[REQUIRED if --api-extraction] API key (no env fallback).")
    ap.add_argument("--api-base-url", type=str, default="", help="Custom API base URL. Defaults by provider if omitted.")
    ap.add_argument("--api-temperature", type=float, default=0.0, help="Sampling temperature for API extraction.")
    ap.add_argument("--api-top-p", type=float, default=1.0, help="Nucleus sampling probability mass for API extraction.")
    ap.add_argument("--api-top-k", type=int, default=None, help="Top-k sampling cutoff for API extraction.")
    ap.add_argument("--api-min-p", type=float, default=None, help="Minimum probability threshold for API extraction.")
    ap.add_argument("--api-max-tokens", type=int, default=16384, help="Max tokens for API extraction response.")
    ap.add_argument(
        "--api-timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds. Default: unlimited.",
    )
    ap.add_argument("--api-seed", type=int, default=42, help="Seed for reproducibility.")
    ap.add_argument(
        "--api-reasoning-level",
        type=str,
        default="none",
        help=(
            "Reasoning level for API extraction requests. Supported levels: none, minimal, low, medium, high, xhigh. "
            "Aliases: off->none, on->high."
        ),
    )
    ap.add_argument("--api-max-retries", type=int, default=3, help="Max retry attempts for transient errors.")
    ap.add_argument("--api-retry-base-sleep", type=float, default=5.0, help="Base backoff sleep in seconds.")
    ap.add_argument(
        "--api-judge-parse-retries",
        type=int,
        default=2,
        help=(
            "Extra retries when judge output is present but invalid for required format. "
            "Total judge attempts = 1 + this value."
        ),
    )
    ap.add_argument(
        "--python-workers",
        "--parallel-workers",
        dest="parallel_workers",
        type=int,
        default=16,
        help="Number of worker threads for per-record processing. Use 1 for sequential processing.",
    )
    ap.add_argument("--api-prefer", action="store_true", help="Prefer API extraction when available (over local methods).")
    ap.add_argument(
        "--no-api-include-trace",
        dest="api_include_trace",
        action="store_false",
        default=True,
        help="Disable including record['trace'] in extractor API input context.",
    )
    ap.add_argument(
        "--api-openrouter-provider",
        type=str,
        default="",
        help="Comma-separated OpenRouter provider order, e.g. 'OpenAI,Anthropic'.",
    )
    ap.add_argument(
        "--api-openrouter-allow-fallbacks",
        action="store_true",
        help="Allow OpenRouter fallback providers when provider order is set.",
    )
    ap.add_argument(
        "--api-store-raw-response",
        action="store_true",
        default=False,
        help="Store exact raw extractor API payload in each record as api_extraction_raw_response. Disabled by default.",
    )
    ap.add_argument(
        "--api-debug-judge-io",
        action="store_true",
        default=False,
        help="Write a separate JSON file with raw judge inputs (messages) and raw judge outputs per processed record.",
    )
    ap.add_argument(
        "--api-debug-judge-io-path",
        type=str,
        default="",
        help="Output path for judge I/O debug JSON. Default: <input_dir>/judge_io_debug.json",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records from input for faster debugging.",
    )
    ap.add_argument(
        "--progress-jsonl",
        type=str,
        default="",
        help=(
            "Append one JSON object per processed record to this file for live monitoring. "
            "Default: <input_dir>/<input_stem>.progress.jsonl"
        ),
    )
    ap.add_argument(
        "--no-progress-jsonl",
        dest="progress_jsonl_enabled",
        action="store_false",
        default=True,
        help="Disable progress JSONL logging.",
    )
    ap.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing progress JSONL when available (default: enabled).",
    )
    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from scratch and overwrite progress JSONL.",
    )
    ap.add_argument(
        "--min-screenshots",
        type=int,
        default=1,
        help="Minimum generated screenshots required per record. Set 0 to disable this gate.",
    )
    ap.add_argument(
        "--strict-final",
        action="store_true",
        help=(
            "Strict local extraction: answers must come from <<FINAL>> blocks only (not regex over full CoT). "
            "Uses strict JSON when valid; if parsing fails (e.g. unescaped newlines in steps_summary), "
            "falls back to selected_option inside that same <<FINAL>> segment only "
            "(quoted 1–4/N/A, unquoted digit 1–4, or unquoted N/A). "
            "Records extraction_method strict_final_relaxed when the fallback matched."
        ),
    )
    args = ap.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_file}")
    if not input_file.is_file():
        raise ValueError(f"Input must be a JSON file, got directory: {input_file}")

    output_file = Path(args.output) if args.output else default_output_json_path(input_file)
    progress_jsonl_path: Optional[Path] = None
    if args.progress_jsonl_enabled:
        progress_jsonl_path = Path(args.progress_jsonl) if args.progress_jsonl else input_file.parent / f"{input_file.stem}.progress.jsonl"
        if progress_jsonl_path.exists() and not args.resume:
            progress_jsonl_path.unlink()
        if progress_jsonl_path.exists() and args.resume:
            print(f"Progress JSONL log: {progress_jsonl_path} (resume enabled)", flush=True)
        else:
            print(f"Progress JSONL log: {progress_jsonl_path}", flush=True)
    else:
        print("Progress JSONL log disabled (--no-progress-jsonl).", flush=True)

    aggregate_stats = make_stats_bucket()
    judge_debug_ctx: Optional[Dict[str, Any]] = None
    source_run_meta = read_input_meta(input_file)
    min_screenshots = max(0, int(args.min_screenshots))

    api_cfg: Optional[Dict[str, Any]] = None
    if args.api_extraction:
        if not args.api_model:
            raise ValueError("--api-model is required when --api-extraction is enabled.")
        if not args.api_key:
            raise ValueError("--api-key is required when --api-extraction is enabled (no env fallback).")

        requested_reasoning_level = (args.api_reasoning_level or "none").strip().lower() or "none"
        api_reasoning_level = normalize_reasoning_level(requested_reasoning_level)
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
        if requested_reasoning_level not in known_reasoning_values:
            print(
                "Ignoring unsupported --api-reasoning-level "
                f"'{args.api_reasoning_level}'; expected one of "
                "none/minimal/low/medium/high/xhigh (or on/off alias). "
                "Falling back to none.",
                flush=True,
            )

        base_url = (args.api_base_url or DEFAULT_BASE_URLS.get(args.api_provider, "")).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        api_cfg = {
            "provider": args.api_provider,
            "model": args.api_model,
            "base_url": base_url,
            "api_key": args.api_key,
            "temperature": args.api_temperature,
            "top_p": args.api_top_p,
            "top_k": args.api_top_k,
            "min_p": args.api_min_p,
            "max_tokens": args.api_max_tokens,
            "timeout": args.api_timeout,
            "seed": args.api_seed,
            "reasoning_level": api_reasoning_level,
            "openrouter_provider_order": [
                part.strip() for part in (args.api_openrouter_provider or "").split(",") if part.strip()
            ],
            "openrouter_allow_fallbacks": bool(args.api_openrouter_allow_fallbacks),
            "max_retries": args.api_max_retries,
            "retry_base_sleep": args.api_retry_base_sleep,
            "judge_parse_retries": args.api_judge_parse_retries,
            "prefer_api": bool(args.api_prefer),
            "include_trace": bool(args.api_include_trace),
            "store_raw_response": bool(args.api_store_raw_response),
            "data_dir": input_file.parent,
        }
        print(
            "API extraction enabled: "
            f"provider={args.api_provider}, model={args.api_model}, prefer={args.api_prefer}, "
            f"reasoning_level={api_reasoning_level}",
            flush=True,
        )
    elif args.api_debug_judge_io:
        raise ValueError("--api-debug-judge-io requires --api-extraction.")

    if args.api_debug_judge_io:
        judge_debug_ctx = {"entries": [], "lock": Lock()}
        print("Judge I/O debug capture enabled.", flush=True)

    if args.strict_final:
        print("Strict <<FINAL>>-only extraction enabled (--strict-final).", flush=True)
    print(f"Loading and processing {input_file.name}...", flush=True)
    total, extracted, file_stats = process_file(
        input_file,
        output_file,
        api_cfg=api_cfg,
        min_screenshots=min_screenshots,
        parallel_workers=args.parallel_workers,
        debug_ctx=judge_debug_ctx,
        limit=args.limit,
        progress_jsonl_path=progress_jsonl_path,
        resume=bool(args.resume),
        strict_final=bool(args.strict_final),
    )
    merge_stats_bucket(aggregate_stats, file_stats)

    finalize_stats_bucket(aggregate_stats)

    stats_output = {
        "input": str(input_file),
        "output": str(output_file),
        "source_run_meta": source_run_meta or None,
        "api_extraction_enabled": bool(args.api_extraction),
        "api_provider": args.api_provider if args.api_extraction else None,
        "api_model": args.api_model if args.api_extraction else None,
        "api_reasoning_level": api_cfg.get("reasoning_level") if args.api_extraction and api_cfg else None,
        "api_top_p": api_cfg.get("top_p") if args.api_extraction and api_cfg else None,
        "api_top_k": api_cfg.get("top_k") if args.api_extraction and api_cfg else None,
        "api_min_p": api_cfg.get("min_p") if args.api_extraction and api_cfg else None,
        "prefer_api": bool(args.api_prefer) if args.api_extraction else None,
        "api_judge_parse_retries": int(args.api_judge_parse_retries) if args.api_extraction else None,
        "api_include_trace": bool(args.api_include_trace) if args.api_extraction else None,
        "api_store_raw_response": bool(args.api_store_raw_response) if args.api_extraction else None,
        "api_debug_judge_io": bool(args.api_debug_judge_io) if args.api_extraction else None,
        "min_screenshots": min_screenshots,
        "strict_final_extraction": bool(args.strict_final),
        "limit": int(args.limit) if args.limit is not None else None,
        **aggregate_stats,
    }
    default_stats_path = input_file.parent / "extraction_stats.json"
    stats_out_path = Path(args.stats_out) if args.stats_out else default_stats_path
    print(f"Writing stats to {stats_out_path}...", flush=True)
    stats_out_path.write_text(json.dumps(stats_output, indent=2, ensure_ascii=False), encoding="utf-8")

    if judge_debug_ctx is not None:
        default_judge_debug_path = input_file.parent / "judge_io_debug.json"
        judge_debug_out_path = Path(args.api_debug_judge_io_path) if args.api_debug_judge_io_path else default_judge_debug_path
        sorted_entries = sorted(
            judge_debug_ctx["entries"],
            key=lambda entry: (
                int(entry.get("debug_index")) if isinstance(entry, dict) and isinstance(entry.get("debug_index"), int) else 10**12,
                str(entry.get("id") if isinstance(entry, dict) else ""),
            ),
        )
        judge_debug_payload = {
            "input": str(input_file),
            "provider": args.api_provider,
            "model": args.api_model,
            "records": sorted_entries,
        }
        print(f"Writing judge I/O debug to {judge_debug_out_path}...", flush=True)
        judge_debug_out_path.write_text(
            json.dumps(judge_debug_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        f"Records={total} extracted={extracted}",
        flush=True,
    )
    print(f"Wrote output JSON to {output_file}", flush=True)
    print(f"Wrote stats to {stats_out_path}")
    if judge_debug_ctx is not None:
        print(f"Wrote judge I/O debug to {judge_debug_out_path}", flush=True)


if __name__ == "__main__":
    main()
