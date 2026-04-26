"""Shared LLM adapter for OpenAI-compatible providers.

Prefers Groq when `GROQ_API_KEY` is present and falls back to OpenAI when
`OPENAI_API_KEY` is present. The rest of the codebase should call the helpers
here instead of constructing clients directly.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

_client: OpenAI | None = None
_provider: str | None = None


def has_llm_credentials() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def get_provider_name() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return "Groq"
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI"
    raise ValueError("No LLM API key configured. Set GROQ_API_KEY or OPENAI_API_KEY.")


def _resolve_model(preferred_openai_model: str | None = None) -> str:
    provider = get_provider_name()
    if provider == "Groq":
        return os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return preferred_openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def get_client() -> OpenAI:
    global _client, _provider

    provider = get_provider_name()
    if _client is not None and _provider == provider:
        return _client

    if provider == "Groq":
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=GROQ_BASE_URL,
        )
    else:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    _provider = provider
    return _client


def chat_json(
    system: str,
    user: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    preferred_openai_model: str = "gpt-4o-mini",
) -> dict:
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=_resolve_model(preferred_openai_model),
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _parse_json(response.choices[0].message.content or "{}")
    except Exception:
        if get_provider_name() != "Groq" or not os.environ.get("OPENAI_API_KEY"):
            raise
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=preferred_openai_model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _parse_json(response.choices[0].message.content or "{}")


def chat_text(
    system: str,
    user: str,
    max_tokens: int = 4000,
    temperature: float = 0.5,
    preferred_openai_model: str = "gpt-4o-mini",
) -> str:
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=_resolve_model(preferred_openai_model),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        if get_provider_name() != "Groq" or not os.environ.get("OPENAI_API_KEY"):
            raise
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=preferred_openai_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text.strip())
