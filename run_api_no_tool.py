import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from api_helpers import (
    DEFAULT_BASE_URLS as API_HELPERS_DEFAULT_BASE_URLS,
    call_api_with_retry as api_call_with_retry,
    extract_reasoning as api_extract_reasoning,
    extract_response_text as api_extract_response_text,
    extract_usage as api_extract_usage,
    normalize_reasoning_level as api_normalize_reasoning_level,
)


DEFAULT_BASE_URLS = dict(API_HELPERS_DEFAULT_BASE_URLS)


def normalize_reasoning_level(reasoning_level: Optional[str]) -> str:
    return api_normalize_reasoning_level(reasoning_level)


def load_b64_data_url(img_path: str, base_dir: Path) -> str:
    p = (base_dir / img_path).resolve()
    data = p.read_bytes()
    ext = p.suffix.lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def count_input_images(q: Dict[str, Any], include_visualization: bool) -> int:
    count = len(q.get("question_images") or [])
    if include_visualization:
        count += len(q.get("visualization_images") or [])
    count += sum(1 for o in (q.get("options") or []) if o.get("image"))
    return count


def build_prompt(
    q: Dict[str, Any],
    include_visualization: bool,
    include_options: bool,
    base_dir: Path,
    prompt_text: str,
    prompt_role: str,
) -> List[Dict[str, Any]]:
    instructions = prompt_text

    content = [{"type": "text", "text": f"Question: {q.get('question_text')}"}]

    if not include_options:
        content.append({"type": "text", "text": "Options are hidden. Provide the final answer directly (no option label)."})

    if q.get("question_images"):
        for img in q["question_images"]:
            content.append({"type": "image_url", "image_url": {"url": load_b64_data_url(img["path"], base_dir)}})

    if include_visualization and q.get("visualization_images"):
        for img in q["visualization_images"]:
            content.append({"type": "image_url", "image_url": {"url": load_b64_data_url(img["path"], base_dir)}})

    options = q.get("options") or []

    if include_options and options:
        has_option_images = any(o.get("image") for o in options)
        if has_option_images:
            content.append({"type": "text", "text": "Options:"})
            for o in options:
                label = str(o.get("label") or "").strip()
                content.append({"type": "text", "text": f"{label}."})
                if o.get("image"):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": load_b64_data_url(o["image"]["path"], base_dir)},
                        }
                    )
        else:
            opts_text = "\n".join([f"{o.get('label')}. {o.get('text') or ''}".strip() for o in options])
            content.append({"type": "text", "text": f"Options:\n{opts_text}"})


    if prompt_role == "system":
        return [
            {"role": "system", "content": [{"type": "text", "text": instructions}]},
            {"role": "user", "content": content},
        ]
    return [{"role": "user", "content": [{"type": "text", "text": instructions}] + content}]


def get_default_api_key(api_provider: str) -> str:
    default_keys = {
        "llamacpp": "llama",
        "openrouter": "",
        "vllm": "vllm",
    }
    return default_keys.get(api_provider, "")


def sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "model"


def ensure_json_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    suffix = f"{path.suffix}.json" if path.suffix else ".json"
    return path.with_suffix(suffix)


def format_run_config_suffix(max_tokens: int, reasoning_level: Optional[str]) -> str:
    safe_reasoning = sanitize_filename_component((reasoning_level or "none").strip().lower() or "none")
    return f"mt{max(0, int(max_tokens))}_reasoning_{safe_reasoning}"


def resolve_output_paths(
    raw_out: str,
    model: str,
    max_tokens: int,
    reasoning_level: Optional[str],
) -> Tuple[Path, Path]:
    safe_model = sanitize_filename_component(model)
    out_path = Path(raw_out)
    if not out_path.is_absolute() and out_path.parent == Path(".") and out_path.name:
        Path("results").mkdir(parents=True, exist_ok=True)
        out_path = Path("results") / out_path

    looks_like_dir = raw_out.endswith(("/", "\\")) or (not out_path.suffix and not out_path.exists())
    if looks_like_dir:
        out_path = out_path / f"results_{safe_model}.json"
    else:
        out_path = ensure_json_suffix(out_path)
        out_path = out_path.with_name(f"{out_path.stem}_{safe_model}{out_path.suffix}")

    run_config_suffix = format_run_config_suffix(max_tokens, reasoning_level)
    model_dir = out_path.parent / f"{safe_model}_{run_config_suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_out_path = ensure_json_suffix(model_dir / out_path.name)
    per_response_dir = model_dir / "responses"
    per_response_dir.mkdir(parents=True, exist_ok=True)
    return final_out_path, per_response_dir


def load_existing_results(out_path: Path) -> Dict[tuple, Dict[str, Any]]:
    existing_results_by_key: Dict[tuple, Dict[str, Any]] = {}
    if not out_path.exists():
        return existing_results_by_key
    try:
        existing_output = json.loads(out_path.read_text())
    except Exception:
        return existing_results_by_key

    for item in existing_output.get("results", []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("id")), str(item.get("variant")))
        existing_results_by_key[key] = item
    return existing_results_by_key


def build_result_entry(
    q: Dict[str, Any],
    variant: str,
    num_input_images: int,
    has_image_options: bool,
    args: Any,
    messages: Optional[List[Dict[str, Any]]],
    usage_info: Dict[str, Optional[int]],
    response_text: Optional[str],
    reasoning: Optional[Any],
    finish_reason: Optional[str],
    openrouter_provider_order: List[str],
    raw_response: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    result_entry = {
        "id": q.get("id"),
        "qnum": q.get("qnum"),
        "difficulty": q.get("difficulty"),
        "variant": variant,
        "num_input_images": num_input_images,
        "has_image_options": has_image_options,
        "model": args.model,
        "api_provider": args.api_provider,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "response_text": response_text,
        "finish_reason": finish_reason,
        "gt": str(q.get("answer")) if q.get("answer") is not None else None,
        "prompt_tokens": usage_info["prompt_tokens"],
        "completion_tokens": usage_info["completion_tokens"],
        "total_tokens": usage_info["total_tokens"],
        "openrouter_provider_order": openrouter_provider_order if args.api_provider == "openrouter" and openrouter_provider_order else None,
        "openrouter_allow_fallbacks": bool(args.openrouter_allow_fallbacks) if args.api_provider == "openrouter" and openrouter_provider_order else None,
    }
    # if reasoning is not None:
    result_entry["reasoning"] = reasoning
    if args.save_input_messages:
        result_entry["input_messages"] = messages
    if getattr(args, "save_raw_response", False):
        result_entry["raw_response"] = raw_response
    if error is not None:
        result_entry["error"] = error
    return result_entry


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name to use for API calls.")
    ap.add_argument("--api-provider", choices=["llamacpp", "vllm", "openrouter"], default="openrouter")
    ap.add_argument("--base-url", default="", help="Custom API base URL. If empty, provider default is used.")
    ap.add_argument("--api-key", default="", help="API key. If empty, provider-specific env var is used.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability mass.")
    ap.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    ap.add_argument("--min-p", type=float, default=None, help="Minimum probability threshold sampling.")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional generation seed for providers that support it (recommended for reproducibility).",
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=5, help="Max retry attempts for transient errors.")
    ap.add_argument("--retry-base-sleep", type=float, default=5.0, help="Base backoff seconds before jitter.")
    ap.add_argument(
        "--python-workers",
        "--parallel-requests",
        dest="python_workers",
        type=int,
        default=1,
        help="Number of concurrent Python worker threads for API requests. Use 1 for sequential processing.",
    )
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Per-request read timeout in seconds. By default, requests wait indefinitely.",
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prompt", default="PROMPT.md")
    ap.add_argument("--prompt-role", choices=["system", "user"], default="system")
    ap.add_argument("--subset-start", type=int, default=0)
    ap.add_argument("--subset-count", type=int, default=0)
    ap.add_argument(
        "--include-visualization",
        action="store_true",
        help="Include visualization images in the prompt when available.",
    )
    ap.add_argument(
        "--openrouter-provider",
        type=str,
        default="",
        help="Comma-separated OpenRouter provider order, e.g. 'OpenAI,Anthropic'",
    )
    ap.add_argument(
        "--openrouter-allow-fallbacks",
        action="store_true",
        help="Allow OpenRouter fallback providers when provider order is set.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if results already exist.",
    )
    ap.add_argument(
        "--reasoning-level",
        type=str,
        default="none",
        help=(
            "Reasoning level to use (none/low/medium/high). "
            "A single canonical value is used and mapped to provider-specific request fields internally."
        ),
    )

    ap.add_argument(
        "--save-raw-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store the full raw API response payload in output files for debugging. Disabled by default.",
    )

    ap.add_argument(
        "--question-only",
        action="store_true",
        help="If set, omit options from the prompt and expect the model to answer directly without choosing a label.",
    )
    ap.add_argument(
        "--save-input-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store input_messages, including base64 image payloads, in output files. Enabled by default.",
    )

    args = ap.parse_args()
    args.python_workers = max(1, int(args.python_workers))

    base_url = (args.base_url or DEFAULT_BASE_URLS[args.api_provider]).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    api_key = args.api_key or get_default_api_key(args.api_provider)
    if args.api_provider == "openrouter" and not api_key:
        raise ValueError(f"API key is required for provider '{args.api_provider}'.")
    if args.api_provider == "vllm" and not api_key:
        api_key = "vllm"
    if args.api_provider == "llamacpp" and not api_key:
        api_key = "llama"

    openrouter_provider_order = [p.strip() for p in args.openrouter_provider.split(",") if p.strip()]
    requested_reasoning_level = (args.reasoning_level or "none").strip().lower() or "none"
    reasoning_level = normalize_reasoning_level(requested_reasoning_level)
    if reasoning_level == "none" and requested_reasoning_level not in {
        "none", "minimal", "low", "medium", "high", "xhigh",
        "off", "disable", "disabled", "no", "on", "enable", "enabled",
        "min", "x-high", "very-high", "very_high",
    }:
        print(
            "Ignoring unsupported reasoning level "
            f"'{args.reasoning_level}'; expected one of none/minimal/low/medium/high/xhigh (or on/off alias). Falling back to none."
        )

    if args.api_provider not in {"openrouter", "vllm", "llamacpp"}:
        print(
            f"Ignoring --reasoning-level for api_provider={args.api_provider}; "
            "supported for openrouter, vllm, and llamacpp only."
        )
        reasoning_level = None

    if args.api_provider == "vllm" and reasoning_level != "none":
        print(
            "Note: vLLM reasoning output requires the server to be launched with a compatible "
            "--reasoning-parser (for example: --reasoning-parser qwen3)."
        )

    data_path = Path(args.data)
    base_dir = data_path.parent
    prompt_text = Path(args.prompt).read_text().strip()
    out_path, per_response_dir = resolve_output_paths(
        args.out,
        args.model,
        args.max_tokens,
        reasoning_level,
    )

    data = json.loads(data_path.read_text())
    if args.subset_count > 0:
        data = data[args.subset_start:args.subset_start + args.subset_count]
    elif args.subset_start > 0:
        data = data[args.subset_start:]

    results: List[Dict[str, Any]] = []

    existing_results_by_key = load_existing_results(out_path)

    def build_output_payload() -> Dict[str, Any]:
        return {
            "meta": {
                "model": args.model,
                "api_provider": args.api_provider,
                "base_url": base_url,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "prompt": str(Path(args.prompt)),
                "data": str(data_path),
                "subset_start": args.subset_start,
                "subset_count": args.subset_count,
                "max_retries": args.max_retries,
                "retry_base_sleep": args.retry_base_sleep,
                "python_workers": args.python_workers,
                "parallel_requests": args.python_workers,
                "request_timeout": args.request_timeout,
                "save_input_messages": args.save_input_messages,
                "save_raw_response": args.save_raw_response,
                "reasoning_level": reasoning_level,
                "openrouter_provider_order": openrouter_provider_order or None,
                "openrouter_allow_fallbacks": bool(args.openrouter_allow_fallbacks),
            },
            "results": results,
        }

    variants = ["with_visualization"] if args.include_visualization else ["no_visualization"]

    def process_one(q: Dict[str, Any], variant: str) -> Dict[str, Any]:
        include_vis = variant == "with_visualization"
        include_opts = not args.question_only
        record_id = str(q.get("id"))
        record_key = (record_id, variant)
        file_stub = f"{q.get('id') or q.get('qnum') or 'unknown'}_{variant}"
        per_response_path = per_response_dir / f"{sanitize_filename_component(str(file_stub))}.json"

        existing_record = None
        if per_response_path.exists():
            try:
                loaded = json.loads(per_response_path.read_text())
                if isinstance(loaded, dict):
                    existing_record = loaded
            except Exception:
                existing_record = None
        elif record_key in existing_results_by_key:
            existing_record = existing_results_by_key[record_key]

        if not args.force and isinstance(existing_record, dict) and isinstance(existing_record.get("response_text"), str):
            print(f"{q.get('id')} [{variant}] skipped (already exists)")
            return existing_record

        messages = build_prompt(q, include_vis, include_opts, base_dir, prompt_text, args.prompt_role)
        num_input_images = count_input_images(q, include_vis)
        has_image_options = any(o.get("image") for o in (q.get("options") or []))
        max_retries = max(1, args.max_retries)
        raw, last_error = api_call_with_retry(
            messages=messages,
            api_provider=args.api_provider,
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            seed=args.seed,
            reasoning_level=reasoning_level,
            openrouter_provider_order=openrouter_provider_order,
            openrouter_allow_fallbacks=args.openrouter_allow_fallbacks,
            max_retries=max_retries,
            retry_base_sleep=args.retry_base_sleep,
        )

        # Validate response structure even when raw is not None.
        choices = raw.get("choices") if raw is not None else None
        usage_info = api_extract_usage(raw)
        if raw is None or not choices or not isinstance(choices[0].get("message"), dict):
            if raw is not None and last_error is None:
                last_error = f"Unexpected response structure: {json.dumps(raw)[:500]}"
            print(f"{q.get('id')} [{variant}] request failed after {max_retries} attempts: {last_error}")
            result_entry = build_result_entry(
                q=q,
                variant=variant,
                num_input_images=num_input_images,
                has_image_options=has_image_options,
                args=args,
                messages=messages,
                usage_info=usage_info,
                response_text=None,
                reasoning=None,
                finish_reason=None,
                openrouter_provider_order=openrouter_provider_order,
                raw_response=raw,
                error=last_error,
            )
            write_json(per_response_path, result_entry)
            return result_entry

        choice0 = choices[0]
        text = api_extract_response_text(choice0)
        reasoning = api_extract_reasoning(raw)

        # If content is null but reasoning is present (e.g., vLLM with reasoning),
        # use reasoning as the response text.
        if text is None and reasoning is not None:
            if isinstance(reasoning, str):
                text = reasoning
            else:
                text = json.dumps(reasoning, ensure_ascii=False)

        # Require either content or reasoning to be present.
        if text is None:
            if raw is not None and last_error is None:
                last_error = f"Response has neither content nor reasoning: {json.dumps(raw)[:500]}"
            print(f"{q.get('id')} [{variant}] request failed after {max_retries} attempts: {last_error}")
            result_entry = build_result_entry(
                q=q,
                variant=variant,
                num_input_images=num_input_images,
                has_image_options=has_image_options,
                args=args,
                messages=messages,
                usage_info=usage_info,
                response_text=None,
                reasoning=reasoning,
                finish_reason=None,
                openrouter_provider_order=openrouter_provider_order,
                raw_response=raw,
                error=last_error,
            )
            write_json(per_response_path, result_entry)
            return result_entry

        finish_reason = choice0.get("finish_reason")
        print(f"{q.get('id')} [{variant}] finish_reason={finish_reason}")
        result_entry = build_result_entry(
            q=q,
            variant=variant,
            num_input_images=num_input_images,
            has_image_options=has_image_options,
            args=args,
            messages=messages,
            usage_info=usage_info,
            response_text=text,
            reasoning=reasoning,
            finish_reason=finish_reason,
            openrouter_provider_order=openrouter_provider_order,
            raw_response=raw,
        )
        write_json(per_response_path, result_entry)
        return result_entry

    jobs: List[Tuple[int, Dict[str, Any], str]] = []
    idx = 0
    for q in data:
        for variant in variants:
            jobs.append((idx, q, variant))
            idx += 1

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    if args.python_workers == 1:
        for i, q, variant in tqdm(jobs, desc="Requests"):
            results_by_idx[i] = process_one(q, variant)
    else:
        with ThreadPoolExecutor(max_workers=args.python_workers) as executor:
            futures = {executor.submit(process_one, q, variant): (i, q, variant) for i, q, variant in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Requests"):
                i, q, variant = futures[fut]
                try:
                    results_by_idx[i] = fut.result()
                except Exception as e:
                    err_msg = f"Unhandled worker error: {type(e).__name__}: {e}"
                    print(f"{q.get('id')} [{variant}] {err_msg}")
                    include_vis = variant == "with_visualization"
                    include_opts = not args.question_only
                    messages = build_prompt(q, include_vis, include_opts, base_dir, prompt_text, args.prompt_role)
                    usage_info = api_extract_usage(None)
                    has_image_options = any(o.get("image") for o in (q.get("options") or []))
                    fallback = build_result_entry(
                        q=q,
                        variant=variant,
                        num_input_images=count_input_images(q, include_vis),
                        has_image_options=has_image_options,
                        args=args,
                        messages=messages,
                        usage_info=usage_info,
                        response_text=None,
                        reasoning=None,
                        finish_reason=None,
                        openrouter_provider_order=openrouter_provider_order,
                        raw_response=None,
                        error=err_msg,
                    )
                    results_by_idx[i] = fallback

    results.extend(results_by_idx[i] for i in range(len(jobs)) if i in results_by_idx)

    output = build_output_payload()
    write_json(out_path, output)
    print(f"Wrote {len(results)} results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
