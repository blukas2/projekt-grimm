import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from google import genai
from google.genai import types


DEFAULT_TRANSLATOR_MODEL = "gemma-4-26b-a4b-it"
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

    german_root: str
    english_translation: str
    translated_text: str
    lexical_type: LexicalType
    display_text: str
    noun: GermanNounData | None = None
    verb: GermanVerbData | None = None


@dataclass(frozen=True)
class VocabularyEntry:
    """Normalized vocabulary row derived from a translation result."""

    german_root: str
    english_translation: str
    other_forms: str


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
    vocabulary_entries: tuple[VocabularyEntry, ...] = ()

    @classmethod
    def plain(
        cls,
        direction: TranslationDirection,
        source_text: str,
        translated_text: str,
        vocabulary_entries: list[VocabularyEntry],
    ) -> "TranslationResult":
        return cls(
            result_type="plain",
            direction=direction,
            source_text=source_text,
            translated_text=translated_text,
            display_text=translated_text,
            vocabulary_entries=tuple(deduplicate_vocabulary_entries(vocabulary_entries)),
        )

    @classmethod
    def lexical(
        cls,
        direction: TranslationDirection,
        source_text: str,
        lexical_translations: list[LexicalTranslation],
        vocabulary_entries: list[VocabularyEntry],
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
            vocabulary_entries=tuple(deduplicate_vocabulary_entries(vocabulary_entries)),
        )


class GemmaTranslator:
    """Translate between English and German with Gemma 4 26B."""

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

    def normalize_vocabulary_entries(
        self,
        entries: Sequence[VocabularyEntry],
        batch_size: int = 25,
    ) -> list[VocabularyEntry]:
        """Normalize stored English vocabulary in batched translator requests."""
        if batch_size < 1:
            raise TranslationError("Vocabulary normalization batch size must be at least 1.")
        normalized_entries: list[VocabularyEntry] = []
        for batch in chunk_vocabulary_entries(entries, batch_size):
            prompt = build_vocabulary_normalization_prompt(batch)
            response_text = self._generate_text(prompt)
            normalized_entries.extend(parse_normalized_vocabulary_response(batch, response_text))
        return normalized_entries

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
    if is_single_word:
        return build_single_word_prompt(source_text, direction)
    return build_sentence_prompt(source_text, direction)


def build_sentence_prompt(source_text: str, direction: TranslationDirection) -> str:
    """Create a JSON prompt for sentence translation plus vocabulary extraction."""
    source_label, target_label = direction_labels(direction)
    return (
        f"You translate {source_label} text into natural {target_label}.\n\n"
        "Return compact JSON only. Do not use markdown fences.\n"
        "Use this schema exactly:\n"
        '{"translated_text":"...","vocabulary":[{"german_root":"...","english_translation":"...","lexical_type":"noun|verb|other","german_noun":{"lemma":"...","article":"der|die|das","plural":"..."}|null,'
        '"german_verb":{"infinitive":"...","third_person_singular":"...","simple_past":"...","past_participle":"...","auxiliary":"haben|sein"}|null}]}\n\n'
        "Rules:\n"
        f"- translated_text must contain only the final {target_label} translation.\n"
        "- Preserve meaning, tone, line breaks, and formatting.\n"
        "- Do not add explanations, notes, or quotation marks unless the source includes them.\n"
        "- If the input contains names, numbers, or code, preserve them exactly when appropriate.\n\n"
        "Vocabulary rules:\n"
        "- Extract at most 3 important words or short terms from the meaning of the text.\n"
        "- Always return vocabulary in German root form, regardless of translation direction.\n"
        "- For nouns, german_root must be article plus singular lemma, and german_noun must be filled.\n"
        "- For verbs, german_root must be the infinitive, and german_verb must be filled.\n"
        "- For other lexical items, german_root must be the normalized German base form and both grammar objects must be null.\n"
        "- english_translation must be the normalized English base form or canonical English term for the German root.\n"
        "- Prefer meaningful content words over function words, except when a negation like nicht is central to the sentence meaning.\n"
        "- Avoid duplicates.\n\n"
        f"{source_label.title()} text:\n{source_text}"
    )


def build_single_word_prompt(source_text: str, direction: TranslationDirection) -> str:
    """Create a JSON-only lexical lookup prompt for single words."""
    source_label, target_label = direction_labels(direction)
    return (
        f"You translate one {source_label} word into {target_label}.\n"
        "Return compact JSON only. Do not use markdown fences.\n"
        "Use this schema exactly:\n"
        '{"translations":[{"german_root":"...","english_translation":"...","translated_text":"...","lexical_type":"noun|verb|other","german_noun":{"lemma":"...","article":"der|die|das","plural":"..."}|null,'
        '"german_verb":{"infinitive":"...","third_person_singular":"...","simple_past":"...","past_participle":"...","auxiliary":"haben|sein"}|null}]}\n\n'
        "Rules:\n"
        f"- Translate the source word from {source_label} into {target_label}.\n"
        "- Return all clearly relevant senses for the same source word when they differ by part of speech or common meaning.\n"
        "- Keep the list short, usually one to three translations, ordered by usefulness.\n"
        "- german_root must always be the normalized German root form for the German side of the pair.\n"
        "- english_translation must always be the normalized English base form or canonical English term for the English side of the pair.\n"
        "- For nouns, german_root must be article plus singular lemma.\n"
        "- For verbs, german_root must be the infinitive.\n"
        "- For other items, german_root must be the normalized German dictionary form.\n"
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


def build_vocabulary_normalization_prompt(entries: Sequence[VocabularyEntry]) -> str:
    """Create a JSON prompt that normalizes stored English vocabulary in batch."""
    input_payload = {
        "entries": [
            {
                "index": index,
                "german_root": entry.german_root,
                "english_translation": entry.english_translation,
                "other_forms": entry.other_forms,
            }
            for index, entry in enumerate(entries)
        ]
    }
    return (
        "You normalize English vocabulary for a German vocabulary list.\n"
        "Return compact JSON only. Do not use markdown fences.\n"
        "Use this schema exactly:\n"
        '{"entries":[{"index":0,"english_translation":"..."}]}\n\n'
        "Rules:\n"
        "- Return the same number of entries and preserve each input index exactly once.\n"
        "- english_translation must be the normalized English base form or canonical English term for the given German root.\n"
        "- Use standard English casing for the meaning, such as lowercase for common words and normal capitalization for proper nouns or named calendar terms.\n"
        "- Keep the gloss short, usually one word or a short term.\n"
        "- Use other_forms only to preserve the intended sense, not to invent a new meaning.\n\n"
        f"Input JSON:\n{json.dumps(input_payload, ensure_ascii=True)}"
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
    payload = parse_json_payload(response_text)
    if not is_single_word:
        return build_sentence_result(source_text, direction, payload)
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
    vocabulary_entries = build_lexical_vocabulary_entries(lexical_translations)
    return TranslationResult.lexical(
        direction,
        source_text,
        lexical_translations,
        vocabulary_entries,
    )


def build_sentence_result(
    source_text: str,
    direction: TranslationDirection,
    payload: dict,
) -> TranslationResult:
    """Construct a sentence translation result with extracted vocabulary."""
    translated_text = required_string(payload, "translated_text")
    vocabulary_entries = build_sentence_vocabulary_entries(payload.get("vocabulary"))
    return TranslationResult.plain(
        direction,
        source_text,
        translated_text,
        vocabulary_entries,
    )


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
    german_root = required_string(payload, "german_root")
    english_translation = required_string(payload, "english_translation")
    translated_text = required_string(payload, "translated_text")
    lexical_type = required_lexical_type(payload)
    noun = build_noun_data(payload.get("german_noun"), lexical_type)
    verb = build_verb_data(payload.get("german_verb"), lexical_type)
    display_text = format_display_text(direction, translated_text, lexical_type, noun, verb)
    return LexicalTranslation(
        german_root,
        english_translation,
        translated_text,
        lexical_type,
        display_text,
        noun,
        verb,
    )


def build_sentence_vocabulary_entries(payload: object) -> list[VocabularyEntry]:
    """Parse sentence vocabulary entries from the model payload."""
    if payload is None:
        return []
    entries_payload = expect_list(payload, "vocabulary")
    entries = [build_sentence_vocabulary_entry(item) for item in entries_payload]
    return deduplicate_vocabulary_entries(entries)[:3]


def build_sentence_vocabulary_entry(payload: object) -> VocabularyEntry:
    """Build one vocabulary entry from sentence extraction JSON."""
    entry_payload = expect_mapping(payload, "vocabulary")
    lexical_type = required_lexical_type(entry_payload)
    noun = build_noun_data(entry_payload.get("german_noun"), lexical_type)
    verb = build_verb_data(entry_payload.get("german_verb"), lexical_type)
    german_root = resolve_german_root(entry_payload, lexical_type, noun, verb)
    english_translation = required_string(entry_payload, "english_translation")
    other_forms = format_other_forms(lexical_type, noun, verb)
    return VocabularyEntry(german_root, english_translation, other_forms)


def build_lexical_vocabulary_entries(
    lexical_translations: list[LexicalTranslation],
) -> list[VocabularyEntry]:
    """Convert lexical translations into normalized vocabulary entries."""
    entries = [build_lexical_vocabulary_entry(entry) for entry in lexical_translations]
    return deduplicate_vocabulary_entries(entries)


def build_lexical_vocabulary_entry(
    lexical_translation: LexicalTranslation,
) -> VocabularyEntry:
    """Build a stored vocabulary row from one lexical translation."""
    other_forms = format_other_forms(
        lexical_translation.lexical_type,
        lexical_translation.noun,
        lexical_translation.verb,
    )
    return VocabularyEntry(
        lexical_translation.german_root,
        lexical_translation.english_translation,
        other_forms,
    )


def parse_normalized_vocabulary_response(
    source_entries: Sequence[VocabularyEntry],
    response_text: str,
) -> list[VocabularyEntry]:
    """Parse one batched normalization response into updated vocabulary entries."""
    payload = parse_json_payload(response_text)
    entries_payload = expect_list(payload.get("entries"), "entries")
    normalized_entries = build_normalized_vocabulary_entries(source_entries, entries_payload)
    if len(normalized_entries) != len(source_entries):
        raise MalformedTranslationError("The normalization payload returned the wrong number of entries.")
    return normalized_entries


def build_normalized_vocabulary_entries(
    source_entries: Sequence[VocabularyEntry],
    entries_payload: list[object],
) -> list[VocabularyEntry]:
    """Build normalized vocabulary entries while preserving input order."""
    normalized_entries: list[VocabularyEntry | None] = [None] * len(source_entries)
    for entry_payload in entries_payload:
        normalized_entry = build_normalized_vocabulary_entry(source_entries, entry_payload)
        index = required_index(expect_mapping(entry_payload, "entries"), len(source_entries))
        if normalized_entries[index] is not None:
            raise MalformedTranslationError("The normalization payload duplicated an entry index.")
        normalized_entries[index] = normalized_entry
    if any(entry is None for entry in normalized_entries):
        raise MalformedTranslationError("The normalization payload is missing an entry index.")
    return [entry for entry in normalized_entries if entry is not None]


def build_normalized_vocabulary_entry(
    source_entries: Sequence[VocabularyEntry],
    payload: object,
) -> VocabularyEntry:
    """Build one normalized vocabulary entry from model JSON."""
    entry_payload = expect_mapping(payload, "entries")
    index = required_index(entry_payload, len(source_entries))
    source_entry = source_entries[index]
    return VocabularyEntry(
        source_entry.german_root,
        required_string(entry_payload, "english_translation"),
        source_entry.other_forms,
    )


def chunk_vocabulary_entries(
    entries: Sequence[VocabularyEntry],
    batch_size: int,
) -> list[Sequence[VocabularyEntry]]:
    """Split vocabulary entries into fixed-size request batches."""
    return [entries[index:index + batch_size] for index in range(0, len(entries), batch_size)]


def resolve_german_root(
    payload: dict,
    lexical_type: LexicalType,
    noun: GermanNounData | None,
    verb: GermanVerbData | None,
) -> str:
    """Choose the normalized German root for storage."""
    if lexical_type == "noun" and noun:
        return f"{noun.article} {noun.lemma}"
    if lexical_type == "verb" and verb:
        return verb.infinitive
    return required_string(payload, "german_root")


def format_other_forms(
    lexical_type: LexicalType,
    noun: GermanNounData | None,
    verb: GermanVerbData | None,
) -> str:
    """Serialize non-root German forms for vocabulary storage."""
    if lexical_type == "noun" and noun:
        return f"Plural: die {noun.plural}"
    if lexical_type == "verb" and verb:
        return format_verb_forms(verb)
    return ""


def format_verb_forms(verb: GermanVerbData) -> str:
    """Serialize key German verb forms onto one line."""
    return (
        f"3rd person singular: {verb.third_person_singular}; "
        f"Simple past: {verb.simple_past}; "
        f"Past participle: {perfect_auxiliary(verb.auxiliary)} {verb.past_participle}"
    )


def deduplicate_vocabulary_entries(entries: list[VocabularyEntry]) -> list[VocabularyEntry]:
    """Keep the first vocabulary row for each German-English pair."""
    unique_entries: list[VocabularyEntry] = []
    seen_keys: set[tuple[str, str]] = set()
    for entry in entries:
        key = vocabulary_key(entry)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_entries.append(entry)
    return unique_entries


def vocabulary_key(entry: VocabularyEntry) -> tuple[str, str]:
    """Return a normalized uniqueness key for one vocabulary row."""
    return normalize_for_key(entry.german_root), normalize_for_key(entry.english_translation)


def normalize_for_key(value: str) -> str:
    """Normalize text for stable deduplication."""
    collapsed = " ".join(value.split())
    return collapsed.casefold()


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


def expect_list(payload: object, key: str) -> list[object]:
    """Require a list object for structured lexical metadata."""
    if isinstance(payload, list):
        return payload
    raise MalformedTranslationError(f"The translation payload is missing '{key}'.")


def required_index(payload: dict, entry_count: int) -> int:
    """Read a required entry index from a normalization payload."""
    value = payload.get("index")
    if isinstance(value, int) and 0 <= value < entry_count:
        return value
    raise MalformedTranslationError("The normalization payload contains an invalid entry index.")


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