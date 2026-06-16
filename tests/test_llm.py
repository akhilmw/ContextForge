import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from contextforge.llm import FakeLLM, GeminiLLM


def make_gemini_llm(client, model="gemini-2.5-flash"):
    llm = object.__new__(GeminiLLM)
    llm.api_key = "test-api-key"
    llm.client = client
    llm.model = model
    return llm


def test_fake_llm_returns_configured_response():
    llm = FakeLLM("grounded answer")

    answer = llm.generate("Use these sources to answer the question.")

    assert answer == "grounded answer"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_fake_llm_rejects_empty_prompt(prompt):
    llm = FakeLLM("grounded answer")

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        llm.generate(prompt)


def test_gemini_llm_generates_text():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="Gemini answer",
    )
    llm = make_gemini_llm(client)

    answer = llm.generate("Answer using only the provided context.")

    assert answer == "Gemini answer"
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert call.kwargs["contents"] == "Answer using only the provided context."


@pytest.mark.parametrize("prompt", ["", "   "])
def test_gemini_llm_rejects_empty_prompt_without_calling_api(prompt):
    client = Mock()
    llm = make_gemini_llm(client)

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        llm.generate(prompt)

    client.models.generate_content.assert_not_called()


def test_gemini_llm_rejects_empty_response():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="")
    llm = make_gemini_llm(client)

    with pytest.raises(ValueError, match="Gemini returned an empty response"):
        llm.generate("Answer the question.")


def test_gemini_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Could not fetch the API Key"):
        GeminiLLM()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1",
    reason="Set RUN_GEMINI_INTEGRATION=1 to call the live Gemini API",
)
def test_gemini_llm_live_api():
    llm = GeminiLLM()

    answer = llm.generate("Reply with one short sentence about Python.")

    assert answer.strip()
