"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load simple KEY=VALUE lines from .env without requiring python-dotenv."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_path = parent / ".env"
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def _env_key(name: str) -> str | None:
    value = os.getenv(name)
    if not value or "PASTE_YOUR" in value:
        return None
    return value


def _max_tokens(default: int = 512) -> int:
    value = os.getenv("LLM_MAX_TOKENS")
    if not value:
        return default
    return int(value)


def get_llm(model: str | None = None, temperature: float = 0.0) -> object:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. OPENAI_API_KEY → ChatOpenAI
    3. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    _load_dotenv()

    openrouter_key = _env_key("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            api_key=openrouter_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            max_tokens=_max_tokens(),
        )

    gemini_key = _env_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=gemini_key,
            temperature=temperature,
        )

    openai_key = _env_key("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=temperature,
            max_tokens=_max_tokens(),
        )

    anthropic_key = _env_key("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            api_key=anthropic_key,
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set OPENROUTER_API_KEY, GEMINI_API_KEY, "
        "OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
