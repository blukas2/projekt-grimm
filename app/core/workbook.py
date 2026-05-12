"""Workbook domain models shared by the UI and agent layers."""

from dataclasses import dataclass
from enum import Enum


class WorkbookTaskFamily(str, Enum):
    """Supported workbook task families."""

    FILL_IN_BLANK = "fill_in_blank"


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
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookTask":
        paragraphs = payload.get("paragraphs") or []
        blanks = payload.get("blanks") or []
        word_bank = payload.get("word_bank") or []
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
        )

    def blank_for(self, blank_id: str) -> WorkbookBlank | None:
        for blank in self.blanks:
            if blank.blank_id == blank_id:
                return blank
        return None


@dataclass(frozen=True)
class WorkbookAnswer:
    """One user answer for a workbook blank."""

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
class WorkbookValidationResult:
    """Full validation payload for a submitted workbook task."""

    task_id: str
    grade: int
    summary: str
    chat_explanation: str
    blank_results: tuple[WorkbookBlankResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "grade": self.grade,
            "summary": self.summary,
            "chat_explanation": self.chat_explanation,
            "blank_results": [result.to_dict() for result in self.blank_results],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkbookValidationResult":
        blank_results = payload.get("blank_results") or []
        return cls(
            task_id=str(payload.get("task_id") or ""),
            grade=max(1, min(5, int(payload.get("grade") or 5))),
            summary=str(payload.get("summary") or ""),
            chat_explanation=str(payload.get("chat_explanation") or ""),
            blank_results=tuple(WorkbookBlankResult.from_dict(item) for item in blank_results),
        )

    def result_for(self, blank_id: str) -> WorkbookBlankResult | None:
        for result in self.blank_results:
            if result.blank_id == blank_id:
                return result
        return None


@dataclass(frozen=True)
class WorkbookEvent:
    """One workbook event emitted by the teacher agent."""

    kind: WorkbookEventKind
    task: WorkbookTask | None = None
    validation: WorkbookValidationResult | None = None
