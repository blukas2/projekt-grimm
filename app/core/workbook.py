"""Workbook domain models shared by the UI and agent layers."""

from dataclasses import dataclass
from enum import Enum


class WorkbookTaskFamily(str, Enum):
    """Supported workbook task families."""

    FILL_IN_BLANK = "fill_in_blank"
    GUIDED_WRITING = "guided_writing"
    READING_COMPREHENSION = "reading_comprehension"
    VOCABULARY_PRACTICE = "vocabulary_practice"


class WorkbookAssistanceMode(str, Enum):
    """Supported assistance modes for workbook tasks."""

    AUTO = "auto"
    BRACKET_HINT = "bracket_hint"
    WORD_BANK = "word_bank"
    OPEN = "open"


class WorkbookSegmentKind(str, Enum):
    """Render segment kinds for workbook paragraphs."""

    TEXT = "text"
    BLANK = "blank"


class WorkbookEventKind(str, Enum):
    """In-memory workbook events emitted by the agent."""

    TASK_GENERATED = "task_generated"
    TASK_VALIDATED = "task_validated"


class WorkbookPracticeType(str, Enum):
    """Supported vocabulary-practice subtypes."""

    PAIR_TRANSLATION = "pair_translation"
    PAIR_ARTICLE = "pair_article"
    ALTERNATIVE_FORMS = "alternative_forms"
    GERMAN_TO_ENGLISH = "german_to_english"
    ENGLISH_TO_GERMAN = "english_to_german"


@dataclass(frozen=True)
class WorkbookSegment:
    """One renderable piece inside a workbook paragraph."""

    kind: WorkbookSegmentKind
    text: str = ""
    blank_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "blank_id": self.blank_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookSegment":
        return cls(
            kind=WorkbookSegmentKind(payload.get("kind") or WorkbookSegmentKind.TEXT.value),
            text=str(payload.get("text") or ""),
            blank_id=str(payload.get("blank_id") or ""),
        )


@dataclass(frozen=True)
class WorkbookParagraph:
    """One paragraph inside a workbook task."""

    segments: tuple[WorkbookSegment, ...]

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"segments": [segment.to_dict() for segment in self.segments]}

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookParagraph":
        segments = payload.get("segments") or []
        return cls(segments=tuple(WorkbookSegment.from_dict(item) for item in segments))


@dataclass(frozen=True)
class WorkbookBlank:
    """Definition for one fill-in-the-blank slot."""

    blank_id: str
    prompt: str
    correct_answer: str
    accepted_answers: tuple[str, ...]
    bracket_hint: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "blank_id": self.blank_id,
            "prompt": self.prompt,
            "correct_answer": self.correct_answer,
            "accepted_answers": list(self.accepted_answers),
            "bracket_hint": self.bracket_hint,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookBlank":
        accepted_answers = payload.get("accepted_answers") or []
        return cls(
            blank_id=str(payload.get("blank_id") or ""),
            prompt=str(payload.get("prompt") or ""),
            correct_answer=str(payload.get("correct_answer") or ""),
            accepted_answers=tuple(str(item) for item in accepted_answers),
            bracket_hint=str(payload.get("bracket_hint") or ""),
        )


@dataclass(frozen=True)
class WorkbookQuestion:
    """Definition for one reading-comprehension question."""

    question_id: str
    prompt: str
    correct_answer: str
    accepted_answers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "prompt": self.prompt,
            "correct_answer": self.correct_answer,
            "accepted_answers": list(self.accepted_answers),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookQuestion":
        accepted_answers = payload.get("accepted_answers") or []
        return cls(
            question_id=str(payload.get("question_id") or ""),
            prompt=str(payload.get("prompt") or ""),
            correct_answer=str(payload.get("correct_answer") or ""),
            accepted_answers=tuple(str(item) for item in accepted_answers),
        )


@dataclass(frozen=True)
class WorkbookVocabularyItem:
    """Definition for one vocabulary-practice prompt."""

    item_id: str
    practice_type: WorkbookPracticeType
    prompt: str
    correct_answer: str
    accepted_answers: tuple[str, ...]
    choice_options: tuple[str, ...]
    german_root: str
    english_translation: str
    other_forms: str
    strength_before: int
    strength_cap: int

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "practice_type": self.practice_type.value,
            "prompt": self.prompt,
            "correct_answer": self.correct_answer,
            "accepted_answers": list(self.accepted_answers),
            "choice_options": list(self.choice_options),
            "german_root": self.german_root,
            "english_translation": self.english_translation,
            "other_forms": self.other_forms,
            "strength_before": self.strength_before,
            "strength_cap": self.strength_cap,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookVocabularyItem":
        accepted_answers = payload.get("accepted_answers") or []
        choice_options = payload.get("choice_options") or []
        return cls(
            item_id=str(payload.get("item_id") or ""),
            practice_type=WorkbookPracticeType(
                payload.get("practice_type")
                or WorkbookPracticeType.PAIR_TRANSLATION.value
            ),
            prompt=str(payload.get("prompt") or ""),
            correct_answer=str(payload.get("correct_answer") or ""),
            accepted_answers=tuple(str(item) for item in accepted_answers),
            choice_options=tuple(str(item) for item in choice_options),
            german_root=str(payload.get("german_root") or ""),
            english_translation=str(payload.get("english_translation") or ""),
            other_forms=str(payload.get("other_forms") or ""),
            strength_before=max(1, min(5, int(payload.get("strength_before") or 1))),
            strength_cap=max(1, min(5, int(payload.get("strength_cap") or 5))),
        )


@dataclass(frozen=True)
class WorkbookTask:
    """Structured workbook task rendered inside the Arbeitsbuch area."""

    task_id: str
    family: WorkbookTaskFamily
    title: str
    instructions: str
    prompt_seed: str
    assistance_mode: WorkbookAssistanceMode
    word_bank: tuple[str, ...]
    paragraphs: tuple[WorkbookParagraph, ...]
    blanks: tuple[WorkbookBlank, ...]
    writing_prompt: str = ""
    guidance_points: tuple[str, ...] = ()
    reading_text: str = ""
    questions: tuple[WorkbookQuestion, ...] = ()
    vocabulary_items: tuple[WorkbookVocabularyItem, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family.value,
            "title": self.title,
            "instructions": self.instructions,
            "prompt_seed": self.prompt_seed,
            "assistance_mode": self.assistance_mode.value,
            "word_bank": list(self.word_bank),
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "blanks": [blank.to_dict() for blank in self.blanks],
            "writing_prompt": self.writing_prompt,
            "guidance_points": list(self.guidance_points),
            "reading_text": self.reading_text,
            "questions": [question.to_dict() for question in self.questions],
            "vocabulary_items": [item.to_dict() for item in self.vocabulary_items],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookTask":
        paragraphs = payload.get("paragraphs") or []
        blanks = payload.get("blanks") or []
        guidance_points = payload.get("guidance_points") or []
        questions = payload.get("questions") or []
        word_bank = payload.get("word_bank") or []
        vocabulary_items = payload.get("vocabulary_items") or []
        return cls(
            task_id=str(payload.get("task_id") or ""),
            family=WorkbookTaskFamily(payload.get("family") or WorkbookTaskFamily.FILL_IN_BLANK.value),
            title=str(payload.get("title") or ""),
            instructions=str(payload.get("instructions") or ""),
            prompt_seed=str(payload.get("prompt_seed") or ""),
            assistance_mode=WorkbookAssistanceMode(
                payload.get("assistance_mode") or WorkbookAssistanceMode.OPEN.value
            ),
            word_bank=tuple(str(item) for item in word_bank),
            paragraphs=tuple(WorkbookParagraph.from_dict(item) for item in paragraphs),
            blanks=tuple(WorkbookBlank.from_dict(item) for item in blanks),
            writing_prompt=str(payload.get("writing_prompt") or ""),
            guidance_points=tuple(str(item) for item in guidance_points),
            reading_text=str(payload.get("reading_text") or ""),
            questions=tuple(WorkbookQuestion.from_dict(item) for item in questions),
            vocabulary_items=tuple(
                WorkbookVocabularyItem.from_dict(item) for item in vocabulary_items
            ),
        )

    def blank_for(self, blank_id: str) -> WorkbookBlank | None:
        for blank in self.blanks:
            if blank.blank_id == blank_id:
                return blank
        return None

    def question_for(self, question_id: str) -> WorkbookQuestion | None:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None

    def vocabulary_item_for(self, item_id: str) -> WorkbookVocabularyItem | None:
        for item in self.vocabulary_items:
            if item.item_id == item_id:
                return item
        return None

    def answer_ids(self) -> tuple[str, ...]:
        if self.family == WorkbookTaskFamily.GUIDED_WRITING:
            return ("writing_response",)
        if self.family == WorkbookTaskFamily.READING_COMPREHENSION:
            return tuple(question.question_id for question in self.questions)
        if self.family == WorkbookTaskFamily.VOCABULARY_PRACTICE:
            return tuple(item.item_id for item in self.vocabulary_items)
        return tuple(blank.blank_id for blank in self.blanks)


@dataclass(frozen=True)
class WorkbookAnswer:
    """One user answer for a workbook task input."""

    blank_id: str
    answer_text: str

    def to_dict(self) -> dict[str, str]:
        return {"blank_id": self.blank_id, "answer_text": self.answer_text}

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookAnswer":
        return cls(
            blank_id=str(payload.get("blank_id") or ""),
            answer_text=str(payload.get("answer_text") or ""),
        )


@dataclass(frozen=True)
class WorkbookBlankResult:
    """Validation outcome for one blank."""

    blank_id: str
    submitted_answer: str
    is_correct: bool
    correction: str
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blank_id": self.blank_id,
            "submitted_answer": self.submitted_answer,
            "is_correct": self.is_correct,
            "correction": self.correction,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookBlankResult":
        return cls(
            blank_id=str(payload.get("blank_id") or ""),
            submitted_answer=str(payload.get("submitted_answer") or ""),
            is_correct=bool(payload.get("is_correct")),
            correction=str(payload.get("correction") or ""),
            explanation=str(payload.get("explanation") or ""),
        )


@dataclass(frozen=True)
class WorkbookQuestionResult:
    """Validation outcome for one reading question."""

    question_id: str
    submitted_answer: str
    is_correct: bool
    correction: str
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "submitted_answer": self.submitted_answer,
            "is_correct": self.is_correct,
            "correction": self.correction,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookQuestionResult":
        return cls(
            question_id=str(payload.get("question_id") or ""),
            submitted_answer=str(payload.get("submitted_answer") or ""),
            is_correct=bool(payload.get("is_correct")),
            correction=str(payload.get("correction") or ""),
            explanation=str(payload.get("explanation") or ""),
        )


@dataclass(frozen=True)
class WorkbookWritingResult:
    """Validation outcome for one guided-writing submission."""

    answer_id: str
    submitted_answer: str
    corrected_answer: str
    explanation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "answer_id": self.answer_id,
            "submitted_answer": self.submitted_answer,
            "corrected_answer": self.corrected_answer,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookWritingResult":
        return cls(
            answer_id=str(payload.get("answer_id") or ""),
            submitted_answer=str(payload.get("submitted_answer") or ""),
            corrected_answer=str(payload.get("corrected_answer") or ""),
            explanation=str(payload.get("explanation") or ""),
        )


@dataclass(frozen=True)
class WorkbookVocabularyResult:
    """Validation outcome for one vocabulary item."""

    item_id: str
    submitted_answer: str
    is_correct: bool
    correction: str
    explanation: str
    strength_before: int
    strength_after: int

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "submitted_answer": self.submitted_answer,
            "is_correct": self.is_correct,
            "correction": self.correction,
            "explanation": self.explanation,
            "strength_before": self.strength_before,
            "strength_after": self.strength_after,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookVocabularyResult":
        return cls(
            item_id=str(payload.get("item_id") or ""),
            submitted_answer=str(payload.get("submitted_answer") or ""),
            is_correct=bool(payload.get("is_correct")),
            correction=str(payload.get("correction") or ""),
            explanation=str(payload.get("explanation") or ""),
            strength_before=max(1, min(5, int(payload.get("strength_before") or 1))),
            strength_after=max(1, min(5, int(payload.get("strength_after") or 1))),
        )


@dataclass(frozen=True)
class WorkbookValidationResult:
    """Full validation payload for a submitted workbook task."""

    task_id: str
    grade: int
    summary: str
    chat_explanation: str
    blank_results: tuple[WorkbookBlankResult, ...]
    writing_result: WorkbookWritingResult | None = None
    question_results: tuple[WorkbookQuestionResult, ...] = ()
    vocabulary_results: tuple[WorkbookVocabularyResult, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "grade": self.grade,
            "summary": self.summary,
            "chat_explanation": self.chat_explanation,
            "blank_results": [result.to_dict() for result in self.blank_results],
            "writing_result": self.writing_result.to_dict() if self.writing_result else None,
            "question_results": [result.to_dict() for result in self.question_results],
            "vocabulary_results": [result.to_dict() for result in self.vocabulary_results],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookValidationResult":
        blank_results = payload.get("blank_results") or []
        question_results = payload.get("question_results") or []
        vocabulary_results = payload.get("vocabulary_results") or []
        writing_result = payload.get("writing_result")
        return cls(
            task_id=str(payload.get("task_id") or ""),
            grade=max(1, min(5, int(payload.get("grade") or 5))),
            summary=str(payload.get("summary") or ""),
            chat_explanation=str(payload.get("chat_explanation") or ""),
            blank_results=tuple(WorkbookBlankResult.from_dict(item) for item in blank_results),
            writing_result=(
                WorkbookWritingResult.from_dict(writing_result)
                if isinstance(writing_result, dict)
                else None
            ),
            question_results=tuple(
                WorkbookQuestionResult.from_dict(item) for item in question_results
            ),
            vocabulary_results=tuple(
                WorkbookVocabularyResult.from_dict(item) for item in vocabulary_results
            ),
        )

    def result_for(self, blank_id: str) -> WorkbookBlankResult | None:
        for result in self.blank_results:
            if result.blank_id == blank_id:
                return result
        return None

    def question_result_for(self, question_id: str) -> WorkbookQuestionResult | None:
        for result in self.question_results:
            if result.question_id == question_id:
                return result
        return None

    def vocabulary_result_for(self, item_id: str) -> WorkbookVocabularyResult | None:
        for result in self.vocabulary_results:
            if result.item_id == item_id:
                return result
        return None

    def writing_response(self) -> WorkbookWritingResult | None:
        return self.writing_result


@dataclass(frozen=True)
class WorkbookEvent:
    """One workbook event emitted by the teacher agent."""

    kind: WorkbookEventKind
    task: WorkbookTask | None = None
    validation: WorkbookValidationResult | None = None
