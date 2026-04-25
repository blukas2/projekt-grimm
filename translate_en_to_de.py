"""Standalone English-to-German translation with Google's Gemma 3 27B model."""

import argparse
import sys
from pathlib import Path
from typing import Dict

MODEL = "gemma-3-27b-it"
ENV_FILE = Path(__file__).with_name(".env")
TRANSLATION_INSTRUCTIONS = """You translate English text into natural German.

Rules:
- Return only the German translation.
- Preserve meaning, tone, line breaks, and formatting.
- Do not add explanations, notes, or quotation marks unless the source includes them.
- If the input contains names, numbers, or code, preserve them exactly when appropriate.
"""


class GemmaTranslator:
    """Translate English text into German using Google GenAI."""

    def __init__(self, api_key: str, model: str):
        genai_client, genai_types = load_google_genai()
        self._client = genai_client.Client(api_key=api_key)
        self._model = model
        self._config = genai_types.GenerateContentConfig(temperature=0.2)

    def translate(self, source_text: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=build_translation_prompt(source_text),
            config=self._config,
        )
        return response.text.strip()


def main() -> int:
    """Run the translation command line interface."""
    arguments = parse_arguments()
    source_text = read_source_text(arguments)
    api_key = read_api_key()
    translator = GemmaTranslator(api_key=api_key, model=arguments.model)
    print(translator.translate(source_text))
    return 0


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate English text into German with Gemma 3 27B.",
    )
    parser.add_argument("text", nargs="*", help="English text to translate")
    parser.add_argument("--model", default=MODEL, help="Google model name")
    return parser.parse_args()


def read_source_text(arguments: argparse.Namespace) -> str:
    """Read source text from args or stdin."""
    if arguments.text:
        return " ".join(arguments.text).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide text as an argument or pipe it through stdin.")


def read_api_key() -> str:
    """Read the API key from the repository root .env file."""
    env_values = read_environment_file()
    api_key = env_values.get("GEMINI_API_KEY") or env_values.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY in the repository root .env file.")


def read_environment_file() -> Dict[str, str]:
    """Read key-value pairs from the repository root .env file."""
    if not ENV_FILE.exists():
        raise SystemExit("Create a .env file in the repository root with GEMINI_API_KEY=your_api_key_here")
    try:
        from dotenv import dotenv_values
    except ImportError as exc:
        raise SystemExit("Install the project dependencies with: pip install -r requirements.txt") from exc
    return dict(dotenv_values(ENV_FILE))


def build_translation_prompt(source_text: str) -> str:
    """Build a plain text prompt compatible with Gemma models."""
    return f"{TRANSLATION_INSTRUCTIONS}\n\nEnglish text:\n{source_text}\n\nGerman translation:"


def load_google_genai():
    """Import Google GenAI modules with a clear install message on failure."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Install the project dependencies with: pip install -r requirements.txt") from exc
    return genai, types


if __name__ == "__main__":
    raise SystemExit(main())