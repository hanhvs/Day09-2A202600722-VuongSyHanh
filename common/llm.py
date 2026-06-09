"""Shared LLM factory for all agents.

Provider is selected via the LLM_PROVIDER env var:

    openrouter  → OpenRouter (default, OpenAI-compatible aggregator)
    openai      → OpenAI direct
    anthropic   → Anthropic Claude direct
    gemini      → Google Gemini
    custom      → Any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, ...)

Each provider reads its own API key, model, and (where applicable) base URL.
Imports are lazy so a missing optional dependency only errors when that
provider is actually selected.
"""

import os

from langchain_core.language_models import BaseChatModel


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a chat model based on the LLM_PROVIDER env var.

    Args:
        temperature: Optional sampling temperature (0.0 = deterministic,
                     1.0+ = more creative). If None, provider default is used.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    extra = {"temperature": temperature} if temperature is not None else {}

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            **extra,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_api_key=api_key,
            **extra,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required for LLM_PROVIDER=anthropic. "
                "Install it with: uv add langchain-anthropic"
            ) from e

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            api_key=api_key,
            **extra,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for LLM_PROVIDER=gemini. "
                "Install it with: uv add langchain-google-genai"
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=api_key,
            **extra,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("CUSTOM_LLM_BASE_URL")
        if not base_url:
            raise ValueError("CUSTOM_LLM_BASE_URL is not set")
        return ChatOpenAI(
            model=os.getenv("CUSTOM_LLM_MODEL", "gpt-3.5-turbo"),
            openai_api_key=os.getenv("CUSTOM_LLM_API_KEY", "not-needed"),
            openai_api_base=base_url,
            **extra,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        "Expected one of: openrouter, openai, anthropic, gemini, custom"
    )
