"""Regression tests for Gemini 3.8 Flash registration and thinking modes."""

from unittest.mock import Mock

import pytest
from google.genai import types

from providers.gemini import GeminiModelProvider
from providers.openrouter import OpenRouterProvider
from tools.models import ToolModelCategory


@pytest.mark.parametrize(
    ("provider_class", "prefix"),
    [(GeminiModelProvider, ""), (OpenRouterProvider, "google/")],
)
def test_flash_aliases_and_versioned_models(provider_class, prefix):
    """Route general Flash aliases to 3.8 and retain explicit 2.5 access."""
    provider = provider_class("test-key")
    canonical_name = f"{prefix}gemini-3.8-flash"

    for alias in ["flash", "gemini-flash", "flash3.8", "gemini3.8", "FLASH"]:
        assert provider.validate_model_name(alias)
        assert provider.get_capabilities(alias).model_name == canonical_name

    capabilities = provider.get_capabilities(canonical_name)
    assert capabilities.context_window == 1_048_576
    assert capabilities.max_output_tokens == 65_536
    assert capabilities.supports_extended_thinking
    assert capabilities.supports_images
    assert capabilities.supports_function_calling
    assert capabilities.supports_json_mode
    assert canonical_name in provider.list_models(respect_restrictions=False)

    for alias in ["flash2.5", "gemini-flash-2.5", f"{prefix}gemini-2.5-flash"]:
        assert provider.get_capabilities(alias).model_name == f"{prefix}gemini-2.5-flash"


@pytest.mark.parametrize("category", [ToolModelCategory.FAST_RESPONSE, ToolModelCategory.BALANCED])
@pytest.mark.parametrize(
    ("allowed_models", "expected_model"),
    [
        (["gemini-3.8-flash", "gemini-flash-2.5"], "gemini-3.8-flash"),
        (["gemini-flash", "flash2.5"], "gemini-3.8-flash"),
        (["flash2.5"], "gemini-2.5-flash"),
    ],
)
def test_flash_preference_resolves_aliases(category, allowed_models, expected_model):
    """Select canonical Flash versions while respecting the allowed model list."""
    provider = GeminiModelProvider("test-key")
    assert provider.get_preferred_model(category, allowed_models) == expected_model


@pytest.fixture
def gemini_provider():
    """Mock the network boundary while exercising real Gemini request configuration."""
    provider = GeminiModelProvider("test-key")
    provider._client = Mock()
    provider._client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Generated content")]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    return provider


@pytest.mark.parametrize("model_name", ["gemini-3.8-flash", "flash"])
@pytest.mark.parametrize(
    ("thinking_mode", "expected_level"),
    [("minimal", "LOW"), ("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH"), ("max", "HIGH")],
)
def test_flash_thinking_levels(gemini_provider, model_name, thinking_mode, expected_level):
    """Send supported thinking levels without numeric budgets for Gemini 3.8 Flash."""
    response = gemini_provider.generate_content("Test prompt", model_name, thinking_mode=thinking_mode)

    request = gemini_provider.client.models.generate_content.call_args.kwargs
    assert request["model"] == "gemini-3.8-flash"
    assert request["config"].thinking_config.thinking_level == expected_level
    assert request["config"].thinking_config.thinking_budget is None
    assert response.model_name == "gemini-3.8-flash"
    assert response.content == "Generated content"


@pytest.mark.parametrize(
    ("model_name", "max_thinking_tokens", "budget_fraction"),
    [
        ("gemini-2.5-flash", 24_576, 0.33),
        ("gemini-3-pro-preview", 32_768, 0.67),
        ("gemini-3.1-pro-preview", 32_768, 0.67),
    ],
)
def test_existing_gemini_thinking_budgets(gemini_provider, model_name, max_thinking_tokens, budget_fraction):
    """Preserve legacy numeric budgets and the Pro medium-to-high workaround."""
    gemini_provider.generate_content("Test prompt", model_name)

    thinking_config = gemini_provider.client.models.generate_content.call_args.kwargs["config"].thinking_config
    assert thinking_config.thinking_budget == int(max_thinking_tokens * budget_fraction)
    assert thinking_config.thinking_level is None
