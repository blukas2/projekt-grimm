"""Deterministic vocabulary-practice generation and validation."""

from dataclasses import dataclass
from random import Random
from uuid import uuid4

from .vocabulary import VocabularyRepository, VocabularyRow
from .workbook import (
    WorkbookAnswer,
    WorkbookAssistanceMode,
    WorkbookPracticeType,
    WorkbookTask,
    WorkbookTaskFamily,
    WorkbookValidationResult,
    WorkbookVocabularyItem,
    WorkbookVocabularyResult,
)


ARTICLE_CHOICES = ("der", "die", "das")
NOUN_ARTICLES = set(ARTICLE_CHOICES)
MIN_TASK_WORDS = 5
MAX_TASK_WORDS = 10
TYPE_CAPS = {
    WorkbookPracticeType.PAIR_TRANSLATION: 3,
    WorkbookPracticeType.PAIR_ARTICLE: 3,
    WorkbookPracticeType.ALTERNATIVE_FORMS: 3,
    WorkbookPracticeType.GERMAN_TO_ENGLISH: 4,
    WorkbookPracticeType.ENGLISH_TO_GERMAN: 5,
}


@dataclass(frozen=True)
class PracticeWord:
    """Parsed vocabulary row used by the practice generator."""

    row: VocabularyRow
    lexical_kind: str
    lemma: str
    article: str
    plural_form: str
    simple_past: str
    perfect_form: str

    def supports_article(self) -> bool:
        return self.lexical_kind == "noun" and bool(self.article and self.lemma)

    def supports_alternative_forms(self) -> bool:
        return self.supports_noun_forms() or self.supports_verb_forms()

    def supports_noun_forms(self) -> bool:
        return self.lexical_kind == "noun" and bool(self.plural_form)

    def supports_verb_forms(self) -> bool:
        return self.lexical_kind == "verb" and bool(self.simple_past and self.perfect_form)


class VocabularyPracticeService:
    """Create and validate vocabulary practice from stored vocabulary."""

    def __init__(
        self,
        repository: VocabularyRepository,
        rng: Random | None = None,
        strong_word_probability: float = 0.3,
    ):
        self._repository = repository
        self._rng = rng or Random()
        self._strong_word_probability = strong_word_probability

    def create_task(self, request_text: str) -> WorkbookTask:
        words = self._load_words()
        count = self._task_word_count(words)
        selected = self._select_words(words, count)
        items = self._build_items(selected)
        return self._build_task(request_text, items)

    def validate_task(
        self,
        task: WorkbookTask,
        answers: tuple[WorkbookAnswer, ...],
    ) -> WorkbookValidationResult:
        results = self._build_results(task, answers)
        self._repository.update_strengths(self._strength_updates(task, results))
        return self._build_validation(task, results)

    def _load_words(self) -> list[PracticeWord]:
        rows = self._repository.load_rows()
        return [parse_practice_word(row) for row in rows]

    def _task_word_count(self, words: list[PracticeWord]) -> int:
        if len(words) < MIN_TASK_WORDS:
            raise ValueError("Mindestens 5 Woerter im Wortschatz sind fuer das Training noetig.")
        return min(MAX_TASK_WORDS, len(words))

    def _select_words(self, words: list[PracticeWord], count: int) -> list[PracticeWord]:
        remaining = list(words)
        selected = self._reserve_strong_word(remaining)
        while len(selected) < count and remaining:
            chosen = self._pick_weighted_word(remaining)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _reserve_strong_word(self, words: list[PracticeWord]) -> list[PracticeWord]:
        if not self._should_reserve_strong_word(words):
            return []
        chosen = self._pick_weighted_word([word for word in words if word.row.strength == 5])
        words.remove(chosen)
        return [chosen]

    def _should_reserve_strong_word(self, words: list[PracticeWord]) -> bool:
        has_strong = any(word.row.strength == 5 for word in words)
        return has_strong and self._rng.random() < self._strong_word_probability

    def _pick_weighted_word(self, words: list[PracticeWord]) -> PracticeWord:
        weights = [self._word_weight(word) for word in words]
        return self._rng.choices(words, weights=weights, k=1)[0]

    def _word_weight(self, word: PracticeWord) -> float:
        if word.row.strength == 5:
            return 0.35
        return float((6 - word.row.strength) ** 2)

    def _build_items(self, words: list[PracticeWord]) -> tuple[WorkbookVocabularyItem, ...]:
        items = [self._build_item(word, words) for word in words]
        return tuple(items)

    def _build_item(
        self,
        word: PracticeWord,
        words: list[PracticeWord],
    ) -> WorkbookVocabularyItem:
        practice_type = self._choose_practice_type(word)
        return build_vocabulary_item(word, practice_type, words, self._rng)

    def _choose_practice_type(self, word: PracticeWord) -> WorkbookPracticeType:
        eligible = practice_types_for_word(word)
        progressive = [item for item in eligible if TYPE_CAPS[item] > word.row.strength]
        options = progressive or eligible
        weights = [practice_weight(word.row.strength, item) for item in options]
        return self._rng.choices(options, weights=weights, k=1)[0]

    def _build_task(
        self,
        request_text: str,
        items: tuple[WorkbookVocabularyItem, ...],
    ) -> WorkbookTask:
        return WorkbookTask(
            task_id=str(uuid4()),
            family=WorkbookTaskFamily.VOCABULARY_PRACTICE,
            title="Wortschatztraining",
            instructions=build_task_instructions(items),
            prompt_seed=request_text.strip(),
            assistance_mode=WorkbookAssistanceMode.OPEN,
            word_bank=(),
            paragraphs=(),
            blanks=(),
            vocabulary_items=items,
        )

    def _build_results(
        self,
        task: WorkbookTask,
        answers: tuple[WorkbookAnswer, ...],
    ) -> tuple[WorkbookVocabularyResult, ...]:
        answer_map = {answer.blank_id: answer.answer_text for answer in answers}
        return tuple(self._build_result(item, answer_map) for item in task.vocabulary_items)

    def _build_result(
        self,
        item: WorkbookVocabularyItem,
        answer_map: dict[str, str],
    ) -> WorkbookVocabularyResult:
        submitted = answer_map.get(item.item_id, "")
        is_correct = is_expected_answer(submitted, item.accepted_answers)
        strength_after = next_strength(item, is_correct)
        return WorkbookVocabularyResult(
            item_id=item.item_id,
            submitted_answer=submitted,
            is_correct=is_correct,
            correction=item.correct_answer,
            explanation=result_explanation(item, is_correct),
            strength_before=item.strength_before,
            strength_after=strength_after,
        )

    def _strength_updates(
        self,
        task: WorkbookTask,
        results: tuple[WorkbookVocabularyResult, ...],
    ) -> dict[tuple[str, str], int]:
        updates: dict[tuple[str, str], int] = {}
        for result in results:
            item = task.vocabulary_item_for(result.item_id)
            if item is None:
                continue
            updates[normalize_key(item.german_root, item.english_translation)] = result.strength_after
        return updates

    def _build_validation(
        self,
        task: WorkbookTask,
        results: tuple[WorkbookVocabularyResult, ...],
    ) -> WorkbookValidationResult:
        grade = build_grade(results)
        return WorkbookValidationResult(
            task_id=task.task_id,
            grade=grade,
            summary=build_summary(results),
            chat_explanation=build_chat_explanation(results, grade),
            blank_results=(),
            vocabulary_results=results,
        )


def parse_practice_word(row: VocabularyRow) -> PracticeWord:
    article, lemma = parse_article_and_lemma(row.german_root)
    plural_form = parse_noun_plural(row.other_forms)
    simple_past, perfect_form = parse_verb_forms(row.other_forms)
    lexical_kind = detect_lexical_kind(article, plural_form, simple_past, perfect_form)
    return PracticeWord(row, lexical_kind, lemma, article, plural_form, simple_past, perfect_form)


def parse_article_and_lemma(german_root: str) -> tuple[str, str]:
    parts = " ".join(german_root.split()).split(" ", 1)
    if len(parts) == 2 and parts[0].casefold() in NOUN_ARTICLES:
        return parts[0], parts[1]
    return "", german_root.strip()


def parse_noun_plural(other_forms: str) -> str:
    prefix = "Plural:"
    if not other_forms.startswith(prefix):
        return ""
    return other_forms[len(prefix) :].strip()


def parse_verb_forms(other_forms: str) -> tuple[str, str]:
    parts = parse_other_form_parts(other_forms)
    return parts.get("simple past", ""), parts.get("past participle", "")


def parse_other_form_parts(other_forms: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for chunk in other_forms.split(";"):
        label, _, value = chunk.partition(":")
        if not value:
            continue
        parts[label.strip().casefold()] = value.strip()
    return parts


def detect_lexical_kind(
    article: str,
    plural_form: str,
    simple_past: str,
    perfect_form: str,
) -> str:
    if article and plural_form:
        return "noun"
    if simple_past and perfect_form:
        return "verb"
    return "other"


def practice_types_for_word(word: PracticeWord) -> list[WorkbookPracticeType]:
    types = [WorkbookPracticeType.PAIR_TRANSLATION]
    if word.supports_article():
        types.append(WorkbookPracticeType.PAIR_ARTICLE)
    if word.supports_alternative_forms():
        types.append(WorkbookPracticeType.ALTERNATIVE_FORMS)
    types.append(WorkbookPracticeType.GERMAN_TO_ENGLISH)
    types.append(WorkbookPracticeType.ENGLISH_TO_GERMAN)
    return types


def practice_weight(strength: int, practice_type: WorkbookPracticeType) -> float:
    if strength <= 2:
        return low_strength_weight(practice_type)
    if strength == 3:
        return mid_strength_weight(practice_type)
    if strength == 4:
        return upper_strength_weight(practice_type)
    return mastered_weight(practice_type)


def low_strength_weight(practice_type: WorkbookPracticeType) -> float:
    weights = {
        WorkbookPracticeType.PAIR_TRANSLATION: 9.0,
        WorkbookPracticeType.PAIR_ARTICLE: 8.0,
        WorkbookPracticeType.ALTERNATIVE_FORMS: 7.0,
        WorkbookPracticeType.GERMAN_TO_ENGLISH: 2.0,
        WorkbookPracticeType.ENGLISH_TO_GERMAN: 1.0,
    }
    return weights[practice_type]


def mid_strength_weight(practice_type: WorkbookPracticeType) -> float:
    weights = {
        WorkbookPracticeType.PAIR_TRANSLATION: 1.0,
        WorkbookPracticeType.PAIR_ARTICLE: 1.0,
        WorkbookPracticeType.ALTERNATIVE_FORMS: 1.0,
        WorkbookPracticeType.GERMAN_TO_ENGLISH: 8.0,
        WorkbookPracticeType.ENGLISH_TO_GERMAN: 4.0,
    }
    return weights[practice_type]


def upper_strength_weight(practice_type: WorkbookPracticeType) -> float:
    weights = {
        WorkbookPracticeType.PAIR_TRANSLATION: 1.0,
        WorkbookPracticeType.PAIR_ARTICLE: 1.0,
        WorkbookPracticeType.ALTERNATIVE_FORMS: 1.0,
        WorkbookPracticeType.GERMAN_TO_ENGLISH: 1.0,
        WorkbookPracticeType.ENGLISH_TO_GERMAN: 9.0,
    }
    return weights[practice_type]


def mastered_weight(practice_type: WorkbookPracticeType) -> float:
    weights = {
        WorkbookPracticeType.PAIR_TRANSLATION: 1.0,
        WorkbookPracticeType.PAIR_ARTICLE: 1.0,
        WorkbookPracticeType.ALTERNATIVE_FORMS: 1.0,
        WorkbookPracticeType.GERMAN_TO_ENGLISH: 2.0,
        WorkbookPracticeType.ENGLISH_TO_GERMAN: 10.0,
    }
    return weights[practice_type]


def build_vocabulary_item(
    word: PracticeWord,
    practice_type: WorkbookPracticeType,
    words: list[PracticeWord],
    rng: Random,
) -> WorkbookVocabularyItem:
    item_id = str(uuid4())
    prompt = build_prompt(word, practice_type)
    correct_answer = build_correct_answer(word, practice_type)
    accepted_answers = build_accepted_answers(word, practice_type)
    choice_options = build_choice_options(word, practice_type, words, rng)
    return WorkbookVocabularyItem(
        item_id=item_id,
        practice_type=practice_type,
        prompt=prompt,
        correct_answer=correct_answer,
        accepted_answers=accepted_answers,
        choice_options=choice_options,
        german_root=word.row.german_root,
        english_translation=word.row.english_translation,
        other_forms=word.row.other_forms,
        strength_before=word.row.strength,
        strength_cap=TYPE_CAPS[practice_type],
    )


def build_prompt(word: PracticeWord, practice_type: WorkbookPracticeType) -> str:
    if practice_type == WorkbookPracticeType.PAIR_TRANSLATION:
        return f"Waehle die passende englische Uebersetzung fuer: {word.row.german_root}"
    if practice_type == WorkbookPracticeType.PAIR_ARTICLE:
        return f"Waehle den richtigen Artikel fuer: {word.lemma}"
    if practice_type == WorkbookPracticeType.ALTERNATIVE_FORMS:
        return alternative_form_prompt(word)
    if practice_type == WorkbookPracticeType.GERMAN_TO_ENGLISH:
        return f"Uebersetze ins Englische: {word.row.german_root}"
    return f"Uebersetze ins Deutsche: {word.row.english_translation}"


def alternative_form_prompt(word: PracticeWord) -> str:
    if word.supports_noun_forms():
        return f"Nenne den Plural von: {word.row.german_root}"
    return f"Nenne Praeteritum und Perfekt von: {word.row.german_root}"


def build_correct_answer(word: PracticeWord, practice_type: WorkbookPracticeType) -> str:
    if practice_type == WorkbookPracticeType.PAIR_ARTICLE:
        return word.article
    if practice_type == WorkbookPracticeType.ALTERNATIVE_FORMS:
        return alternative_form_answer(word)
    if practice_type == WorkbookPracticeType.ENGLISH_TO_GERMAN:
        return word.row.german_root
    return word.row.english_translation


def alternative_form_answer(word: PracticeWord) -> str:
    if word.supports_noun_forms():
        return word.plural_form
    return f"{word.simple_past}; {word.perfect_form}"


def build_accepted_answers(
    word: PracticeWord,
    practice_type: WorkbookPracticeType,
) -> tuple[str, ...]:
    if practice_type == WorkbookPracticeType.PAIR_ARTICLE:
        return (word.article,)
    if practice_type == WorkbookPracticeType.ALTERNATIVE_FORMS:
        return alternative_form_answers(word)
    if practice_type == WorkbookPracticeType.ENGLISH_TO_GERMAN:
        return german_translation_answers(word)
    return (word.row.english_translation,)


def alternative_form_answers(word: PracticeWord) -> tuple[str, ...]:
    if word.supports_noun_forms():
        return noun_plural_answers(word)
    return verb_form_answers(word)


def noun_plural_answers(word: PracticeWord) -> tuple[str, ...]:
    plural_without_article = remove_article(word.plural_form)
    return tuple(item for item in (word.plural_form, plural_without_article) if item)


def verb_form_answers(word: PracticeWord) -> tuple[str, ...]:
    joined = f"{word.simple_past}; {word.perfect_form}"
    comma_joined = f"{word.simple_past}, {word.perfect_form}"
    slash_joined = f"{word.simple_past} / {word.perfect_form}"
    return (joined, comma_joined, slash_joined)


def german_translation_answers(word: PracticeWord) -> tuple[str, ...]:
    if word.lexical_kind != "noun":
        return (word.row.german_root,)
    return (word.row.german_root, remove_article(word.row.german_root))


def build_choice_options(
    word: PracticeWord,
    practice_type: WorkbookPracticeType,
    words: list[PracticeWord],
    rng: Random,
) -> tuple[str, ...]:
    if practice_type == WorkbookPracticeType.PAIR_ARTICLE:
        return ARTICLE_CHOICES
    if practice_type != WorkbookPracticeType.PAIR_TRANSLATION:
        return ()
    return translation_choices(word, words, rng)


def translation_choices(
    word: PracticeWord,
    words: list[PracticeWord],
    rng: Random,
) -> tuple[str, ...]:
    distractors = [item.row.english_translation for item in words if item != word]
    rng.shuffle(distractors)
    options = distractors[:3] + [word.row.english_translation]
    rng.shuffle(options)
    return tuple(deduplicate_strings(options))


def deduplicate_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value in unique:
            continue
        unique.append(value)
    return unique


def build_task_instructions(items: tuple[WorkbookVocabularyItem, ...]) -> str:
    return (
        f"Bearbeite {len(items)} Wortschatzfragen. "
        "Richtige Antworten staerken ein Wort, falsche Antworten schwaechen es."
    )


def is_expected_answer(submitted: str, accepted_answers: tuple[str, ...]) -> bool:
    normalized = normalize_answer(submitted)
    accepted = {normalize_answer(answer) for answer in accepted_answers}
    return normalized in accepted


def normalize_answer(value: str) -> str:
    collapsed = " ".join(value.strip().split())
    return collapsed.casefold().replace(" ;", ";").replace(" ,", ",")


def next_strength(item: WorkbookVocabularyItem, is_correct: bool) -> int:
    if is_correct:
        return min(5, item.strength_cap, item.strength_before + 1)
    return max(1, item.strength_before - 1)


def result_explanation(item: WorkbookVocabularyItem, is_correct: bool) -> str:
    if is_correct:
        return "Richtig beantwortet."
    return f"Richtige Loesung: {item.correct_answer}"


def build_grade(results: tuple[WorkbookVocabularyResult, ...]) -> int:
    ratio = correct_ratio(results)
    if ratio >= 0.9:
        return 1
    if ratio >= 0.75:
        return 2
    if ratio >= 0.6:
        return 3
    if ratio >= 0.4:
        return 4
    return 5


def correct_ratio(results: tuple[WorkbookVocabularyResult, ...]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for result in results if result.is_correct)
    return correct / len(results)


def build_summary(results: tuple[WorkbookVocabularyResult, ...]) -> str:
    correct = sum(1 for result in results if result.is_correct)
    total = len(results)
    changed = sum(1 for result in results if result.strength_after != result.strength_before)
    return f"{correct} von {total} Antworten waren richtig. {changed} Woerter haben ihre Staerke veraendert."


def build_chat_explanation(
    results: tuple[WorkbookVocabularyResult, ...],
    grade: int,
) -> str:
    correct = sum(1 for result in results if result.is_correct)
    total = len(results)
    return f"Du hast {correct} von {total} richtig geloest. Deine Note ist {grade}."


def normalize_key(german_root: str, english_translation: str) -> tuple[str, str]:
    return normalize_answer(german_root), normalize_answer(english_translation)


def remove_article(value: str) -> str:
    parts = value.split(" ", 1)
    if len(parts) == 2 and parts[0].casefold() in NOUN_ARTICLES:
        return parts[1]
    return value