"""Agent implementations."""

from .german_teacher import GermanTeacherAgent
from .translator import (
	DEFAULT_TRANSLATOR_MODEL,
	GemmaTranslator,
	MissingApiKeyError,
	TranslationError,
	TranslationResult,
)