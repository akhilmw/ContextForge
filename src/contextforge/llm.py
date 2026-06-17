"""Generate final answers from grounded prompts.

The LLM protocol keeps the ask pipeline independent from a specific provider.
FakeLLM supports deterministic tests, while GeminiLLM and OpenAILLM call real
generation APIs for end-to-end usage.
"""

import os
from typing import Protocol

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()


class LLM(Protocol):
    """Operations required from any answer-generation provider."""

    def generate(self, prompt: str) -> str:
        ...


class FakeLLM:
    """Deterministic offline LLM used by tests and local pipeline checks."""

    def __init__(self, response: str):
        """Store the exact response returned for every valid prompt."""
        self.response = response

    def generate(self, prompt: str) -> str:
        """Return the configured response after validating the prompt."""
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        return self.response


class GeminiLLM:
    """Gemini-backed LLM provider for real answer generation."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        """Create a Gemini client using the API key from the environment."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Could not fetch the API Key")
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """Send one grounded prompt to Gemini and return the generated text."""
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response")
        return response.text


class OpenAILLM:
    """OpenAI-backed LLM provider for real answer generation."""

    def __init__(self, model: str = "gpt-5.4-mini"):
        """Create an OpenAI client using the API key from the environment."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Could not fetch the OpenAI API Key")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """Send one grounded prompt to OpenAI and return the generated text."""
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        if not response.output_text:
            raise ValueError("OpenAI returned an empty response")
        return response.output_text
