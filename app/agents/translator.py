import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types


DEFAULT_TRANSLATOR_MODEL = "gemma-3-27b-it"
LOGGER = logging.getLogger("projekt_grimm.translator")
SINGLE_WORD_PATTERN = re.compile(r"^[A-Za-zÄÖÜäöüß'-]+$")

TranslationDirection = Literal["en_to_de", "de_to_en"]
ResultType = Literal["plain", "lexical"]
LexicalType = Literal["noun", "verb", "other"]


class TranslationError(RuntimeError):
    """Base class for translator failures."""


class MissingApiKeyError(TranslationError):
    """Raised when the translator API key is not configured."""


class ModelUnavailableError(TranslationError):
    """Raised when the configured translation model cannot be used."""


class MalformedTranslationError(TranslationError):
    """Raised when the model returns an invalid translator payload."""


@dataclass(frozen=True)
class GermanNounData:
    """Grammatical data for a German noun."""

    lemma: str
    article: str
    plural: str


@dataclass(frozen=True)
class GermanVerbData:
    """Core inflection data for a German verb."""

    infinitive: str
    third_person_singular: str
    simple_past: str
    past_participle: str
    auxiliary: str


@dataclass(frozen=True)
class LexicalTranslation:
    """One relevant lexical translation for a single-word lookup."""

    translated_text: str
    lexical_type: LexicalType
    display_text: str
    noun: GermanNounData | None = None
    verb: GermanVerbData | None = None


@dataclass(frozen=True)
class TranslationResult:
    """Structured translation result used by the UI and CLI."""

    result_type: ResultType
    direction: TranslationDirection
    source_text: str
    translated_text: str
    display_text: str
    lexical_type: LexicalType | None = None
    noun: GermanNounData | None = None
    verb: GermanVerbData | None = None
    lexical_translations: tuple[LexicalTranslation, ...] = ()

    @classmethod
    def plain(
        cls,
        direction: TranslationDirection,
        source_text: str,
        translated_text: str,
    ) -> "TranslationResult":
        return cls(
            result_type="plain",
            direction=direction,
            source_text=source_text,
            translated_text=translated_text,
            display_text=translated_text,
        )

    @classmethod
    def lexical(
        cls,
        direction: TranslationDirection,
        source_text: str,
        lexical_translations: list[LexicalTranslation],
    ) -> "TranslationResult":
        primary_translation = first_lexical_translation(lexical_translations)
        display_text = join_lexical_display_texts(lexical_translations)
        return cls(
            result_type="lexical",
            direction=direction,
            source_text=source_text,
            translated_text=primary_translation.translated_text,
            display_text=display_text,
            lexical_type=primary_translation.lexical_type,
            noun=primary_translation.noun,
            verb=primary_translation.verb,
            lexical_translations=tuple(lexical_translations),
        )


class GemmaTranslator:
    """Translate between English and German with Gemma 3 27B."""

    def __init__(self, api_key: str, model: str = DEFAULT_TRANSLATOR_MODEL):
        if not api_key:
            LOGGER.error("Translator initialization failed: missing API key")
            raise MissingApiKeyError("Set GEMINI_API_KEY or GOOGLE_API_KEY before starting the app.")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._config = types.GenerateContentConfig(temperature=0.2)
        LOGGER.info("Gemma translator initialized")

    def translate(
        self,
        source_text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        cleaned_text = normalize_source_text(source_text)
        is_single_word = should_enrich_translation(cleaned_text)
        prompt = build_translation_prompt(cleaned_text, direction, is_single_word)
        response_text = self._generate_text(prompt)
        return parse_translation_response(cleaned_text, direction, is_single_word, response_text)

    def _generate_text(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config,
            )
        except Exception as exc:
            raise build_generation_error(self._model, exc) from exc
        return extract_response_text(response)


def normalize_source_text(source_text: str) -> str:
    """Normalize user input before classification and prompting."""
    cleaned_text = source_text.strip()
    if not cleaned_text:
        raise TranslationError("Enter text before requesting a translation.")
    return cleaned_text


def should_enrich_translation(source_text: str) -> bool:
    """Treat a single lexical token as a dictionary-style lookup."""
    return bool(SINGLE_WORD_PATTERN.fullmatch(source_text))


def build_translation_prompt(
    source_text: str,
    direction: TranslationDirection,
    is_single_word: bool,
) -> str:
    """Build a prompt kept inside the request body for Gemma."""
    return build_single_word_prompt(source_text, direction) if is_single_word else build_plain_prompt(source_text, direction)


def build_plain_prompt(source_text: str, direction: TranslationDirection) -> str:
    """Create a plain translation prompt for phrases and sentences."""
    source_label, target_label = direction_labels(direction)
    return (
        f"You translate {source_label} text into natural {target_label}.\n\n"
        "Rules:\n"
        f"- Return only the {target_label} translation.\n"
        "- Preserve meaning, tone, line breaks, and formatting.\n"
        "- Do not add explanations, notes, or quotation marks unless the source includes them.\n"
        "- If the input contains names, numbers, or code, preserve them exactly when appropriate.\n\n"
        f"{source_label.title()} text:\n{source_text}\n\n"
        f"{target_label.title()} translation:"
    )


def build_single_word_prompt(source_text: str, direction: TranslationDirection) -> str:
    """Create a JSON-only lexical lookup prompt for single words."""
    source_label, target_label = direction_labels(direction)
    return (
        f"You translate one {source_label} word into {target_label}.\n"
        "Return compact JSON only. Do not use markdown fences.\n"
        "Use this schema exactly:\n"
        '{"translations":[{"translated_text":"...","lexical_type":"noun|verb|other","german_noun":{"lemma":"...","article":"der|die|das","plural":"..."}|null,'
        '"german_verb":{"infinitive":"...","third_person_singular":"...","simple_past":"...","past_participle":"...","auxiliary":"haben|sein"}|null}]}\n\n'
        "Rules:\n"
        f"- Translate the source word from {source_label} into {target_label}.\n"
        "- Return all clearly relevant senses for the same source word when they differ by part of speech or common meaning.\n"
        "- Keep the list short, usually one to three translations, ordered by usefulness.\n"
        "- Set lexical_type to noun, verb, or other.\n"
        "- Fill german_noun only when the German side is a noun.\n"
        "- Fill german_verb only when the German side is a verb.\n"
        "- When translating German into English, keep the German grammatical data from the source word.\n"
        "- translated_text must contain only the translated word or short lexical equivalent.\n"
        "- Do not duplicate equivalent translations.\n"
        "- german_verb.auxiliary must be the correct German Perfekt auxiliary lemma for the verb.\n"
        "- german_verb.past_participle must contain only the participle, never the auxiliary.\n"
        'Examples: gehen -> {"infinitive":"gehen","third_person_singular":"geht","simple_past":"ging","past_participle":"gegangen","auxiliary":"sein"}.\n'
        'Examples: machen -> {"infinitive":"machen","third_person_singular":"macht","simple_past":"machte","past_participle":"gemacht","auxiliary":"haben"}.\n'
        "- Use null for irrelevant objects.\n\n"
        f"Source word: {source_text}"
    )


def direction_labels(direction: TranslationDirection) -> tuple[str, str]:
    """Return human-readable source and target language labels."""
    if direction == "en_to_de":
        return "english", "german"
    return "german", "english"


def build_generation_error(model: str, exc: Exception) -> TranslationError:
    """Map low-level GenAI failures to app-facing translator errors."""
    message = str(exc).lower()
    if "not found" in message or "unsupported" in message or "unavailable" in message:
        LOGGER.exception("Translator model unavailable: %s", model)
        return ModelUnavailableError(f"The translation model '{model}' is unavailable.")
    LOGGER.exception("Translator request failed")
    return TranslationError("The translation request failed. Check the logs for details.")


def extract_response_text(response) -> str:
    """Read text from the model response or fail clearly."""
    response_text = getattr(response, "text", "")
    if response_text and response_text.strip():
        return response_text.strip()
    LOGGER.error("Translator returned an empty response")
    raise MalformedTranslationError("The translation model returned an empty response.")


def parse_translation_response(
    source_text: str,
    direction: TranslationDirection,
    is_single_word: bool,
    response_text: str,
) -> TranslationResult:
    """Normalize either plain text or lexical JSON into the app contract."""
    if not is_single_word:
        return TranslationResult.plain(direction, source_text, response_text.strip())
    payload = parse_json_payload(response_text)
    return build_lexical_result(source_text, direction, payload)


def parse_json_payload(response_text: str) -> dict:
    """Parse a JSON payload and surface malformed model output cleanly."""
    candidate = strip_code_fence(response_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        LOGGER.error("Translator returned malformed JSON: %s", response_text)
        raise MalformedTranslationError("The translation model returned malformed JSON.") from exc
    if isinstance(payload, dict):
        return payload
    raise MalformedTranslationError("The translation model returned an invalid lexical payload.")


def strip_code_fence(response_text: str) -> str:
    """Allow tolerant parsing if the model wraps JSON in fences."""
    stripped_text = response_text.strip()
    if not stripped_text.startswith("```"):
        return stripped_text
    lines = stripped_text.splitlines()
    body = lines[1:-1] if len(lines) >= 3 else []
    return "\n".join(body).strip()


def build_lexical_result(
    source_text: str,
    direction: TranslationDirection,
    payload: dict,
) -> TranslationResult:
    """Construct a structured lexical result from model JSON."""
    lexical_translations = build_lexical_translations(direction, payload)
    return TranslationResult.lexical(direction, source_text, lexical_translations)


def first_lexical_translation(
    lexical_translations: list[LexicalTranslation],
) -> LexicalTranslation:
    """Return the primary lexical translation or fail when none exist."""
    if lexical_translations:
        return lexical_translations[0]
    raise MalformedTranslationError("The translation payload did not include any lexical translations.")


def join_lexical_display_texts(
    lexical_translations: list[LexicalTranslation],
) -> str:
    """Join lexical display blocks into a single fallback display string."""
    return "\n\n".join(entry.display_text for entry in lexical_translations)


def build_lexical_translations(
    direction: TranslationDirection,
    payload: dict,
) -> list[LexicalTranslation]:
    """Build one or more lexical translations from the model payload."""
    translations_payload = payload.get("translations")
    if isinstance(translations_payload, list):
        return parse_translation_list(direction, translations_payload)
    return [build_lexical_entry(direction, payload)]


def parse_translation_list(
    direction: TranslationDirection,
    translations_payload: list[object],
) -> list[LexicalTranslation]:
    """Parse the model's lexical translations array."""
    lexical_translations = [
        build_lexical_entry(direction, expect_mapping(entry, "translations"))
        for entry in translations_payload
    ]
    if lexical_translations:
        return lexical_translations
    raise MalformedTranslationError("The translation payload contains an empty translations list.")


def build_lexical_entry(
    direction: TranslationDirection,
    payload: dict,
) -> LexicalTranslation:
    """Build one lexical translation entry from model JSON."""
    translated_text = required_string(payload, "translated_text")
    lexical_type = required_lexical_type(payload)
    noun = build_noun_data(payload.get("german_noun"), lexical_type)
    verb = build_verb_data(payload.get("german_verb"), lexical_type)
    display_text = format_display_text(direction, translated_text, lexical_type, noun, verb)
    return LexicalTranslation(translated_text, lexical_type, display_text, noun, verb)


def required_string(payload: dict, key: str) -> str:
    """Read a required non-empty string field from a model payload."""
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise MalformedTranslationError(f"The translation payload is missing '{key}'.")


def required_lexical_type(payload: dict) -> LexicalType:
    """Validate the lexical type emitted by the model."""
    lexical_type = required_string(payload, "lexical_type")
    if lexical_type in {"noun", "verb", "other"}:
        return lexical_type
    raise MalformedTranslationError("The translation payload contains an unsupported lexical type.")


def build_noun_data(payload: object, lexical_type: LexicalType) -> GermanNounData | None:
    """Build noun metadata when the German side is nominal."""
    if lexical_type != "noun":
        return None
    noun_payload = expect_mapping(payload, "german_noun")
    return GermanNounData(
        lemma=required_string(noun_payload, "lemma"),
        article=required_string(noun_payload, "article"),
        plural=required_string(noun_payload, "plural"),
    )


def build_verb_data(payload: object, lexical_type: LexicalType) -> GermanVerbData | None:
    """Build verb metadata when the German side is verbal."""
    if lexical_type != "verb":
        return None
    verb_payload = expect_mapping(payload, "german_verb")
    return GermanVerbData(
        infinitive=required_string(verb_payload, "infinitive"),
        third_person_singular=required_string(verb_payload, "third_person_singular"),
        simple_past=required_string(verb_payload, "simple_past"),
        past_participle=required_string(verb_payload, "past_participle"),
        auxiliary=required_string(verb_payload, "auxiliary"),
    )


def expect_mapping(payload: object, key: str) -> dict:
    """Require a dictionary object for structured lexical metadata."""
    if isinstance(payload, dict):
        return payload
    raise MalformedTranslationError(f"The translation payload is missing '{key}'.")


def format_display_text(
    direction: TranslationDirection,
    translated_text: str,
    lexical_type: LexicalType,
    noun: GermanNounData | None,
    verb: GermanVerbData | None,
) -> str:
    """Render stable display text for lexical lookups."""
    if lexical_type == "noun" and noun:
        return format_noun_display(direction, translated_text, noun)
    if lexical_type == "verb" and verb:
        return format_verb_display(direction, translated_text, verb)
    return translated_text


def format_noun_display(
    direction: TranslationDirection,
    translated_text: str,
    noun: GermanNounData,
) -> str:
    """Render noun output with article and plural."""
    german_noun = f"{noun.article} {noun.lemma}"
    lines = [translated_text]
    if direction == "en_to_de":
        lines[0] = german_noun
    else:
        lines.append(f"German noun: {german_noun}")
    lines.append(f"Plural: die {noun.plural}")
    return "\n".join(lines)


def format_verb_display(
    direction: TranslationDirection,
    translated_text: str,
    verb: GermanVerbData,
) -> str:
    """Render verb output with required key forms."""
    lines = [translated_text]
    if direction == "en_to_de":
        lines[0] = verb.infinitive
    else:
        lines.append(f"German verb: {verb.infinitive}")
    lines.append(f"3rd person singular: {verb.third_person_singular}")
    lines.append(f"Simple past: {verb.simple_past}")
    lines.append(f"Past participle: {perfect_auxiliary(verb.auxiliary)} {verb.past_participle}")
    return "\n".join(lines)


def perfect_auxiliary(auxiliary: str) -> str:
    """Convert an auxiliary lemma into the displayed perfect-tense helper."""
    normalized = auxiliary.strip().lower()
    if normalized == "haben":
        return "hat"
    if normalized == "sein":
        return "ist"
    return auxiliary.strip()