#!/usr/bin/env python3
"""
Shared API communication utilities extracted from run_api_no_tool.py.
Provides image handling, multimodal prompt building, and API calls.
"""

import base64
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_BASE_URLS = {
    "llamacpp": "http://localhost:8080",
    "vllm": "http://localhost:8000",
    "openrouter": "https://openrouter.ai/api/v1",
}


def normalize_reasoning_level(reasoning_level: Optional[str]) -> str:
    level = (reasoning_level or "none").strip().lower()
    aliases = {
        "off": "none",
        "disable": "none",
        "disabled": "none",
        "no": "none",
        "on": "high",
        "enable": "high",
        "enabled": "high",
        "min": "minimal",
        "x-high": "xhigh",
        "very-high": "xhigh",
        "very_high": "xhigh",
    }
    level = aliases.get(level, level)
    if level not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        return "none"
    return level


def _llamacpp_thinking_budget_for_level(reasoning_level: str, max_tokens: Optional[int]) -> Optional[int]:
    """Map abstract effort levels to llama.cpp's thinking_budget_tokens."""
    if reasoning_level == "none":
        return None

    ratio_by_level = {
        "minimal": 0.10,
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
        "xhigh": 0.90,
    }
    fallback_budget = {
        "minimal": 128,
        "low": 256,
        "medium": 512,
        "high": 1024,
        "xhigh": 2048,
    }

    if max_tokens is None or max_tokens <= 0:
        return fallback_budget.get(reasoning_level, fallback_budget["medium"])

    requested = max(1, int(max_tokens * ratio_by_level.get(reasoning_level, 0.50)))
    # Keep a response tail budget so we don't spend the full output on thinking.
    reserve_for_answer = max(32, int(max_tokens * 0.15))
    hard_cap = max(1, max_tokens - reserve_for_answer)
    return min(requested, hard_cap)


def _resolve_image_path(img_path: str, base_dir: Path) -> Path:
    raw = str(img_path or "").strip()
    if not raw:
        raise FileNotFoundError("Empty image path")

    candidate = Path(raw)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        repo_dir = Path(__file__).resolve().parent
        workspace_dir = repo_dir.parent
        normalized = raw.replace("\\", "/")

        candidates.extend(
            [
                base_dir / candidate,
                Path.cwd() / candidate,
                repo_dir / candidate,
                workspace_dir / candidate,
            ]
        )

        # If the path embeds the repo folder re-anchor it at the workspace directory.
        repo_token = f"{repo_dir.name}/"
        token_idx = normalized.find(repo_token)
        if token_idx >= 0:
            candidates.append(workspace_dir / Path(normalized[token_idx:]))

        # Common dataset-relative shortcut.
        if normalized.startswith("dataset/"):
            candidates.append(repo_dir / Path(normalized))

    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Image path not found: {img_path}")


def load_b64_data_url(img_path: str, base_dir: Path) -> str:
    """Load image and return as base64 data URL."""
    p = _resolve_image_path(img_path, base_dir)
    data = p.read_bytes()
    ext = p.suffix.lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_multimodal_message(
    system_prompt: str,
    question_text: str,
    question_images: Optional[List[Dict[str, Any]]],
    visualization_images: Optional[List[Dict[str, Any]]],
    options: Optional[List[Dict[str, Any]]],
    include_options: bool,
    base_dir: Path,
    user_instruction: str = "",
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible message list with text and images.
    
    Args:
        system_prompt: system role content
        question_text: question string
        question_images: list of {"path": "..."}
        visualization_images: list of {"path": "..."}
        options: list of option dicts with "label", "text", and optional "image"
        include_options: whether to include options in message
        base_dir: base directory for resolving image paths
        user_instruction: extra user instruction text
    
    Returns:
        OpenAI-compatible message list
    """
    user_content: List[Dict[str, Any]] = []
    
    if user_instruction:
        user_content.append({"type": "text", "text": user_instruction})
    
    if question_text:
        user_content.append({"type": "text", "text": f"Question: {question_text}"})
    
    # Add question images
    if question_images:
        for img in question_images:
            if isinstance(img, dict) and "path" in img:
                url = load_b64_data_url(img["path"], base_dir)
                user_content.append({"type": "image_url", "image_url": {"url": url}})
    
    # Add visualization images
    if visualization_images:
        for img in visualization_images:
            if isinstance(img, dict) and "path" in img:
                url = load_b64_data_url(img["path"], base_dir)
                user_content.append({"type": "image_url", "image_url": {"url": url}})
    
    # Add options
    if include_options and options:
        has_option_images = any(o.get("image") for o in options if isinstance(o, dict))
        if has_option_images:
            user_content.append({"type": "text", "text": "Options:"})
            for opt in options:
                if not isinstance(opt, dict):
                    continue
                label = str(opt.get("label") or "").strip()
                if label:
                    user_content.append({"type": "text", "text": f"{label}."})
                if opt.get("image") and isinstance(opt["image"], dict) and "path" in opt["image"]:
                    url = load_b64_data_url(opt["image"]["path"], base_dir)
                    user_content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            opts_lines = []
            for opt in options:
                if not isinstance(opt, dict):
                    continue
                label = str(opt.get("label") or "").strip()
                text = str(opt.get("text") or "").strip()
                if label and text:
                    opts_lines.append(f"{label}. {text}")
                elif label:
                    opts_lines.append(f"{label}.")
            if opts_lines:
                user_content.append({"type": "text", "text": "Options:\n" + "\n".join(opts_lines)})
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def call_api(
    messages: List[Dict[str, Any]],
    api_provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    request_timeout: Optional[float] = None,
    seed: Optional[int] = None,
    reasoning_level: Optional[str] = None,
    openrouter_provider_order: Optional[List[str]] = None,
    openrouter_allow_fallbacks: bool = False,
) -> Dict[str, Any]:
    """
    Call OpenAI-compatible chat API.
    
    Args:
        messages: OpenAI-compatible message list
        api_provider: provider hint used for provider-specific payload fields
        base_url: API chat-completions base URL
        api_key: API key
        model: model identifier
        temperature: sampling temperature
        max_tokens: max tokens in response
        request_timeout: per-request timeout in seconds
        seed: optional seed for reproducibility
    
    Returns:
        Raw API response dict
    
    Raises:
        requests.HTTPError: on HTTP errors
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if max_tokens is not None and max_tokens >= 0:
        payload["max_tokens"] = max_tokens

    if seed is not None and api_provider in {"vllm", "openrouter", "llamacpp"}:
        payload["seed"] = seed

    normalized_reasoning_level = normalize_reasoning_level(reasoning_level)

    if api_provider == "openrouter":
        if normalized_reasoning_level != "none":
            payload["reasoning"] = {
                "effort": normalized_reasoning_level,
                "exclude": False,
                "enabled": True,
            }
        else:
            payload["reasoning"] = {"effort": "none"}
        if openrouter_provider_order:
            payload["provider"] = {
                "order": openrouter_provider_order,
                "allow_fallbacks": bool(openrouter_allow_fallbacks),
            }

    if api_provider == "vllm":
        effort = {"minimal": "low", "xhigh": "high"}.get(
            normalized_reasoning_level,
            normalized_reasoning_level,
        )
        payload["reasoning_effort"] = effort
        payload["include_reasoning"] = effort != "none"
        payload["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": effort != "none",
            },
        }

    if api_provider == "llamacpp":
        enable_thinking = normalized_reasoning_level != "none"
        payload["reasoning_format"] = "auto" if enable_thinking else "none"
        payload["chat_template_kwargs"] = {
            "enable_thinking": enable_thinking,
        }
        # if enable_thinking:
            # llama.cpp supports per-request thinking budgets via thinking_budget_tokens.
            # budget = _llamacpp_thinking_budget_for_level(normalized_reasoning_level, max_tokens)
            # if budget is not None:
                # payload["thinking_budget_tokens"] = budget
    
    request_kwargs: Dict[str, Any] = {"headers": headers, "json": payload}
    if request_timeout is not None and request_timeout > 0:
        request_kwargs["timeout"] = (10.0, max(1.0, request_timeout))
    
    resp = requests.post(url, **request_kwargs)
    resp.raise_for_status()
    return resp.json()


def extract_usage(raw: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """Extract token usage from API response."""
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def extract_reasoning(raw: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None

    # Some providers return reasoning directly at the choice level.
    for key in ("reasoning", "reasoning_content", "reasoning_text"):
        if choice.get(key) is not None:
            return choice.get(key)

    message = choice.get("message")
    if isinstance(message, dict):
        for key in ("reasoning", "reasoning_content", "reasoning_text", "thinking"):
            if message.get(key) is not None:
                return message.get(key)

        content = message.get("content")
        if isinstance(content, list):
            reasoning_parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type", "")).lower() in {
                    "reasoning",
                    "reasoning_text",
                    "thinking",
                    "summary_text",
                }:
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        reasoning_parts.append(text.strip())
            if reasoning_parts:
                return "\n".join(reasoning_parts)

    return None


def extract_response_text(choice: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract content text from API response choice."""
    if not isinstance(choice, dict):
        return None
    
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    
    return None


def is_retryable_error(msg: str) -> bool:
    """Check if error message indicates a retryable error."""
    msg_lower = msg.lower()
    return any(
        pattern in msg_lower
        for pattern in [
            "429",
            "resource_exhausted",
            "rate",
            "quota",
            "retry",
            "timeout",
            "timed out",
            "temporarily",
            "connection reset",
            "connection aborted",
            "retry later",
            "please try again",
            "502",
            "503",
            "504",
            "server error",
        ]
    )


def parse_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Parse Retry-After header or error message for retry delay."""
    retry_after = None
    response_obj = getattr(exc, "response", None)
    if response_obj is not None and hasattr(response_obj, "headers"):
        ra_val = response_obj.headers.get("Retry-After")
        if ra_val:
            try:
                retry_after = float(ra_val)
            except ValueError:
                pass
    
    msg = str(exc)
    patterns = [
        r"retry\s+in\s+(\d+(?:\.\d+)?)s",
        r"retrydelay['\"]?[:=]\s*['\"]?(\d+(?:\.\d+)?)(?:s|sec|seconds)?",
        r"retry[- ]after[: ](\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, msg, flags=re.IGNORECASE)
        if m:
            try:
                retry_after = float(m.group(1))
                break
            except ValueError:
                continue
    
    return retry_after


def call_api_with_retry(
    messages: List[Dict[str, Any]],
    api_provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    request_timeout: Optional[float] = None,
    seed: Optional[int] = None,
    reasoning_level: Optional[str] = None,
    openrouter_provider_order: Optional[List[str]] = None,
    openrouter_allow_fallbacks: bool = False,
    max_retries: int = 5,
    retry_base_sleep: float = 5.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call API with exponential backoff retry logic.
    
    Returns:
        (response_dict or None, error_message or None)
    """
    last_error: Optional[str] = None
    
    for attempt in range(max_retries):
        try:
            raw = call_api(
                messages,
                api_provider,
                base_url,
                api_key,
                model,
                temperature,
                top_p,
                top_k,
                min_p,
                max_tokens,
                request_timeout,
                seed,
                reasoning_level,
                openrouter_provider_order,
                openrouter_allow_fallbacks,
            )
            return raw, None
        except Exception as exc:
            msg = str(exc)
            last_error = f"{type(exc).__name__}: {msg}"
            
            if is_retryable_error(msg) and attempt < max_retries - 1:
                hinted = parse_retry_after_seconds(exc)
                if hinted is not None:
                    wait_s = min(300, max(0.5, hinted))
                else:
                    base = max(0.5, retry_base_sleep)
                    wait_s = min(120, base * (2 ** attempt))
                wait_s += random.uniform(0, 1)
                print(
                    f"Transient API error; retrying in {wait_s:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries}). Error: {msg}"
                )
                time.sleep(wait_s)
                continue
            break
    
    return None, last_error
