"""Workbook generation and validation tools for the German teacher agent."""

import json
import logging
from uuid import uuid4

from google import genai
from google.genai import types

from app.core.workbook import (
    WorkbookAnswer,
    WorkbookAssistanceMode,
    WorkbookBlank,
    WorkbookBlankResult,
    WorkbookEvent,
    WorkbookEventKind,
    WorkbookParagraph,
    WorkbookQuestion,
    WorkbookQuestionResult,
    WorkbookSegment,
    WorkbookSegmentKind,
    WorkbookTask,
    WorkbookTaskFamily,
    WorkbookValidationResult,
    WorkbookWritingResult,
)
from app.core.vocabulary import VocabularyRepository
from app.core.vocabulary_practice import VocabularyPracticeService


LOGGER = logging.getLogger("projekt_grimm.workbook")
WORKBOOK_MODEL = "gemini-3-flash-preview"

FILL_IN_BLANK_GENERATION_SCHEMA = {
    "type": "object",
    "required": [
        "title",
        "instructions",
        "assistance_mode",
        "word_bank",
        "paragraphs",
        "blanks",
    ],
    "properties": {
        "title": {"type": "string"},
        "instructions": {"type": "string"},
        "assistance_mode": {
            "type": "string",
            "enum": ["bracket_hint", "word_bank", "open"],
        },
        "word_bank": {"type": "array", "items": {"type": "string"}},
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["segments"],
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["kind", "text", "blank_id"],
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": ["text", "blank"],
                                },
                                "text": {"type": "string"},
                                "blank_id": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
        "blanks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "blank_id",
                    "prompt",
                    "correct_answer",
                    "accepted_answers",
                    "bracket_hint",
                ],
                "properties": {
                    "blank_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "accepted_answers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "bracket_hint": {"type": "string"},
                },
            },
        },
    },
}

READING_GENERATION_SCHEMA = {
    "type": "object",
    "required": ["title", "instructions", "reading_text", "questions"],
    "properties": {
        "title": {"type": "string"},
        "instructions": {"type": "string"},
        "reading_text": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "question_id",
                    "prompt",
                    "correct_answer",
                    "accepted_answers",
                ],
                "properties": {
                    "question_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "accepted_answers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}

WRITING_GENERATION_SCHEMA = {
    "type": "object",
    "required": ["title", "instructions", "writing_prompt", "guidance_points"],
    "properties": {
        "title": {"type": "string"},
        "instructions": {"type": "string"},
        "writing_prompt": {"type": "string"},
        "guidance_points": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

FILL_IN_BLANK_VALIDATION_SCHEMA = {
    "type": "object",
    "required": ["grade", "summary", "chat_explanation", "blank_results"],
    "properties": {
        "grade": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"},
        "chat_explanation": {"type": "string"},
        "blank_results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "blank_id",
                    "submitted_answer",
                    "is_correct",
                    "correction",
                    "explanation",
                ],
                "properties": {
                    "blank_id": {"type": "string"},
                    "submitted_answer": {"type": "string"},
                    "is_correct": {"type": "boolean"},
                    "correction": {"type": "string"},
                    "explanation": {"type": "string"},
                },
            },
        },
    },
}

READING_VALIDATION_SCHEMA = {
    "type": "object",
    "required": ["grade", "summary", "chat_explanation", "question_results"],
    "properties": {
        "grade": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"},
        "chat_explanation": {"type": "string"},
        "question_results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "question_id",
                    "submitted_answer",
                    "is_correct",
                    "correction",
                    "explanation",
                ],
                "properties": {
                    "question_id": {"type": "string"},
                    "submitted_answer": {"type": "string"},
                    "is_correct": {"type": "boolean"},
                    "correction": {"type": "string"},
                    "explanation": {"type": "string"},
                },
            },
        },
    },
}

WRITING_VALIDATION_SCHEMA = {
    "type": "object",
    "required": ["grade", "summary", "chat_explanation", "writing_result"],
    "properties": {
        "grade": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"},
        "chat_explanation": {"type": "string"},
        "writing_result": {
            "type": "object",
            "required": [
                "answer_id",
                "submitted_answer",
                "corrected_answer",
                "explanation",
            ],
            "properties": {
                "answer_id": {"type": "string"},
                "submitted_answer": {"type": "string"},
                "corrected_answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
        },
    },
}

GENERATION_SCHEMAS = {
    WorkbookTaskFamily.FILL_IN_BLANK.value: FILL_IN_BLANK_GENERATION_SCHEMA,
    WorkbookTaskFamily.GUIDED_WRITING.value: WRITING_GENERATION_SCHEMA,
    WorkbookTaskFamily.READING_COMPREHENSION.value: READING_GENERATION_SCHEMA,
}

VALIDATION_SCHEMAS = {
    WorkbookTaskFamily.FILL_IN_BLANK.value: FILL_IN_BLANK_VALIDATION_SCHEMA,
    WorkbookTaskFamily.GUIDED_WRITING.value: WRITING_VALIDATION_SCHEMA,
    WorkbookTaskFamily.READING_COMPREHENSION.value: READING_VALIDATION_SCHEMA,
}


class WorkbookToolService:
    """Generate and validate workbook tasks with structured Gemini calls."""

    def __init__(
        self,
        api_key: str,
        vocabulary_repository: VocabularyRepository,
        model: str = WORKBOOK_MODEL,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._events: list[WorkbookEvent] = []
        self._practice_service = VocabularyPracticeService(vocabulary_repository)

    def consume_events(self) -> tuple[WorkbookEvent, ...]:
        """Return and clear all queued workbook events."""
        events = tuple(self._events)
        self._events.clear()
        return events

    def create_workbook_task(
        self,
        request_text: str,
        assistance_mode: str = WorkbookAssistanceMode.AUTO.value,
        task_family: str = WorkbookTaskFamily.FILL_IN_BLANK.value,
    ) -> dict[str, object]:
        """Create a workbook task for the current lesson request.

        Args:
            request_text: The student's natural-language task request.
            assistance_mode: One of auto, bracket_hint, word_bank, or open.
            task_family: The workbook task family to generate.

        Returns:
            A compact status payload for the chat model after the task is created.
        """
        task = self._generate_task(request_text, assistance_mode, task_family)
        self._events.append(WorkbookEvent(kind=WorkbookEventKind.TASK_GENERATED, task=task))
        return build_creation_status(task)

    def validate_workbook_task(self, task_json: str, answers_json: str) -> dict[str, object]:
        """Validate one submitted workbook task.

        Args:
            task_json: A serialized workbook task JSON payload.
            answers_json: A serialized list of workbook answers.

        Returns:
            A compact status payload plus chat-ready explanation text.
        """
        task = WorkbookTask.from_dict(json.loads(task_json))
        answers = parse_answers_payload(json.loads(answers_json))
        result = self._validate_task(task, answers)
        self._events.append(
            WorkbookEvent(kind=WorkbookEventKind.TASK_VALIDATED, validation=result)
        )
        return build_validation_status(result)

    def _generate_task(
        self,
        request_text: str,
        assistance_mode: str,
        task_family: str,
    ) -> WorkbookTask:
        ensure_supported_family(task_family)
        if task_family == WorkbookTaskFamily.VOCABULARY_PRACTICE.value:
            return self._practice_service.create_task(request_text)
        response = self._client.models.generate_content(
            model=self._model,
            contents=build_generation_prompt(request_text, assistance_mode, task_family),
            config=build_generation_config(task_family),
        )
        payload = parse_response_payload(response)
        task = hydrate_task_payload(payload, request_text, task_family)
        LOGGER.info("Workbook task generated")
        return task

    def _validate_task(
        self,
        task: WorkbookTask,
        answers: tuple[WorkbookAnswer, ...],
    ) -> WorkbookValidationResult:
        if task.family == WorkbookTaskFamily.VOCABULARY_PRACTICE:
            return self._practice_service.validate_task(task, answers)
        response = self._client.models.generate_content(
            model=self._model,
            contents=build_validation_prompt(task, answers),
            config=build_validation_config(task.family.value),
        )
        payload = parse_response_payload(response)
        result = hydrate_validation_payload(task, answers, payload)
        LOGGER.info("Workbook task validated")
        return result


def build_generation_config(task_family: str) -> types.GenerateContentConfig:
    """Build the structured-output config for task generation."""
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GENERATION_SCHEMAS[task_family],
        temperature=0.2,
    )


def build_validation_config(task_family: str) -> types.GenerateContentConfig:
    """Build the structured-output config for task validation."""
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VALIDATION_SCHEMAS[task_family],
        temperature=0.2,
    )


def ensure_supported_family(task_family: str):
    """Reject task families that are not implemented yet."""
    if task_family in GENERATION_SCHEMAS:
        return
    if task_family == WorkbookTaskFamily.VOCABULARY_PRACTICE.value:
        return
    raise ValueError(f"Unsupported workbook task family: {task_family}")


def build_generation_prompt(
    request_text: str,
    assistance_mode: str,
    task_family: str,
) -> str:
    """Build the structured prompt for workbook generation."""
    if task_family == WorkbookTaskFamily.GUIDED_WRITING.value:
        return build_writing_generation_prompt(request_text)
    if task_family == WorkbookTaskFamily.READING_COMPREHENSION.value:
        return build_reading_generation_prompt(request_text)
    return build_fill_in_blank_generation_prompt(request_text, assistance_mode)


def build_fill_in_blank_generation_prompt(request_text: str, assistance_mode: str) -> str:
    """Build the structured prompt for fill-in-the-blank generation."""
    return (
        "Du erstellst eine Deutsch-Aufgabe fuer das Arbeitsbuch eines B1/B2-Lerners.\n"
        "Erzeuge genau eine Aufgabe vom Typ fill-in-the-blank.\n"
        "Der gesamte Aufgabentext und alle Erklaerungen muessen auf Deutsch sein.\n"
        "Nutze eine natuerliche, zusammenhaengende Textaufgabe mit 3 bis 6 Luecken.\n"
        "Gib nur JSON gemaess dem Schema zurueck.\n"
        "segment.kind ist immer text oder blank.\n"
        "Bei text-Segmenten enthaelt text den sichtbaren Text und blank_id ist leer.\n"
        "Bei blank-Segmenten ist text leer und blank_id verweist auf ein Objekt in blanks.\n"
        "accepted_answers muss moegliche richtige Varianten enthalten.\n"
        "Wenn assistance_mode bracket_hint ist, muss bracket_hint pro Luecke gesetzt werden.\n"
        "Wenn assistance_mode word_bank ist, muss word_bank alle benoetigten Woerter enthalten.\n"
        "Wenn assistance_mode open ist, lasse bracket_hint leer und word_bank leer.\n"
        "Wenn assistance_mode auto ist, waehle passend zur Anfrage zwischen bracket_hint, word_bank oder open.\n"
        f"Gewuenschter assistance_mode: {assistance_mode}\n"
        f"Anfrage des Lerners: {request_text}"
    )


def build_reading_generation_prompt(request_text: str) -> str:
    """Build the structured prompt for reading-comprehension generation."""
    return (
        "Du erstellst eine Deutsch-Aufgabe fuer das Arbeitsbuch eines B1/B2-Lerners.\n"
        "Erzeuge genau eine Aufgabe vom Typ reading_comprehension.\n"
        "Der gesamte Aufgabentext und alle Erklaerungen muessen auf Deutsch sein.\n"
        "Erstelle einen natuerlichen Lesetext mit etwa 120 bis 180 Woertern.\n"
        "Erstelle danach 3 bis 5 Verstaendnisfragen zum Text.\n"
        "Jede Frage braucht question_id, prompt, correct_answer und accepted_answers.\n"
        "accepted_answers muss sinnvolle richtige Varianten enthalten.\n"
        "Die Fragen sollen den Inhalt des Textes pruefen, nicht isolierte Grammatik.\n"
        "Gib nur JSON gemaess dem Schema zurueck.\n"
        f"Anfrage des Lerners: {request_text}"
    )


def build_writing_generation_prompt(request_text: str) -> str:
    """Build the structured prompt for guided-writing generation."""
    return (
        "Du erstellst eine Schreibaufgabe fuer das Arbeitsbuch eines B1/B2-Lerners.\n"
        "Erzeuge genau eine Aufgabe vom Typ guided_writing.\n"
        "Der gesamte Aufgabentext und alle Erklaerungen muessen auf Deutsch sein.\n"
        "Erstelle eine klare Schreibaufgabe mit einem Hauptprompt und 3 bis 5 Leitpunkten.\n"
        "Die Leitpunkte sollen Thema, Umfang oder sprachliche Ziele konkretisieren.\n"
        "Die Aufgabe soll eine zusammenhaengende freie Antwort des Lerners ausloesen.\n"
        "Gib nur JSON gemaess dem Schema zurueck.\n"
        f"Anfrage des Lerners: {request_text}"
    )


def build_validation_prompt(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
) -> str:
    """Build the structured prompt for workbook validation."""
    if task.family == WorkbookTaskFamily.GUIDED_WRITING:
        return build_writing_validation_prompt(task, answers)
    if task.family == WorkbookTaskFamily.READING_COMPREHENSION:
        return build_reading_validation_prompt(task, answers)
    return build_fill_in_blank_validation_prompt(task, answers)


def build_fill_in_blank_validation_prompt(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
) -> str:
    """Build the structured prompt for fill-in-the-blank validation."""
    return (
        "Du validierst eine ausgefuellte Deutsch-Aufgabe fuer einen B1/B2-Lerner.\n"
        "Bewerte nur diese Aufgabe, nicht die allgemeine Unterhaltung.\n"
        "Gib eine Note von 1 bis 5. 1 ist am besten, 5 am schlechtesten.\n"
        "blank_results muss fuer jede Luecke genau einen Eintrag enthalten.\n"
        "Wenn eine Antwort falsch ist, gib in correction die beste Korrektur an.\n"
        "Wenn eine Antwort richtig ist, darf correction gleich der richtigen Antwort sein.\n"
        "chat_explanation ist eine kurze Lehrerklaerung fuer das Chatfenster.\n"
        f"Aufgabe JSON: {json.dumps(task.to_dict(), ensure_ascii=True)}\n"
        f"Antworten JSON: {json.dumps([answer.to_dict() for answer in answers], ensure_ascii=True)}"
    )


def build_reading_validation_prompt(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
) -> str:
    """Build the structured prompt for reading-comprehension validation."""
    return (
        "Du validierst eine ausgefuellte Leseverstehensaufgabe fuer einen B1/B2-Lerner.\n"
        "Bewerte nur diese Aufgabe, nicht die allgemeine Unterhaltung.\n"
        "Gib eine Note von 1 bis 5. 1 ist am besten, 5 am schlechtesten.\n"
        "question_results muss fuer jede Frage genau einen Eintrag enthalten.\n"
        "Bewerte Antworten inhaltlich grosszuegig, wenn sie den Text korrekt wiedergeben.\n"
        "Wenn eine Antwort ungenau oder falsch ist, gib in correction eine gute Musterantwort an.\n"
        "chat_explanation ist eine kurze Lehrerklaerung fuer das Chatfenster.\n"
        f"Aufgabe JSON: {json.dumps(task.to_dict(), ensure_ascii=True)}\n"
        f"Antworten JSON: {json.dumps([answer.to_dict() for answer in answers], ensure_ascii=True)}"
    )


def build_writing_validation_prompt(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
) -> str:
    """Build the structured prompt for guided-writing validation."""
    return (
        "Du validierst eine freie Schreibaufgabe fuer einen B1/B2-Lerner.\n"
        "Bewerte nur diese Aufgabe, nicht die allgemeine Unterhaltung.\n"
        "Gib eine Note von 1 bis 5. 1 ist am besten, 5 am schlechtesten.\n"
        "writing_result muss genau einen Eintrag fuer den eingereichten Text enthalten.\n"
        "corrected_answer soll eine natuerlich korrigierte Version des gesamten Textes sein.\n"
        "explanation soll die wichtigsten Verbesserungen kurz erklaeren.\n"
        "chat_explanation ist eine kurze Lehrerklaerung fuer das Chatfenster.\n"
        f"Aufgabe JSON: {json.dumps(task.to_dict(), ensure_ascii=True)}\n"
        f"Antworten JSON: {json.dumps([answer.to_dict() for answer in answers], ensure_ascii=True)}"
    )


def parse_response_payload(response) -> dict:
    """Extract a JSON mapping from a structured Gemini response."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    return json.loads(response.text or "{}")


def hydrate_task_payload(
    payload: dict,
    request_text: str,
    task_family: str,
) -> WorkbookTask:
    """Build a workbook task from the structured model payload."""
    if task_family == WorkbookTaskFamily.GUIDED_WRITING.value:
        return hydrate_writing_task_payload(payload, request_text)
    if task_family == WorkbookTaskFamily.READING_COMPREHENSION.value:
        return hydrate_reading_task_payload(payload, request_text)
    return hydrate_fill_in_blank_task_payload(payload, request_text)


def hydrate_fill_in_blank_task_payload(payload: dict, request_text: str) -> WorkbookTask:
    """Build one fill-in-the-blank task from the model payload."""
    task_id = str(uuid4())
    assistance_mode = resolve_assistance_mode(payload)
    paragraphs = tuple(hydrate_paragraphs(payload))
    blanks = tuple(hydrate_blanks(payload, assistance_mode))
    word_bank = tuple(hydrate_word_bank(payload, blanks, assistance_mode))
    return WorkbookTask(
        task_id=task_id,
        family=WorkbookTaskFamily.FILL_IN_BLANK,
        title=str(payload.get("title") or "Neue Aufgabe"),
        instructions=str(payload.get("instructions") or "Fuelle die Luecken aus."),
        prompt_seed=request_text.strip(),
        assistance_mode=assistance_mode,
        word_bank=word_bank,
        paragraphs=paragraphs,
        blanks=blanks,
    )


def hydrate_reading_task_payload(payload: dict, request_text: str) -> WorkbookTask:
    """Build one reading-comprehension task from the model payload."""
    return WorkbookTask(
        task_id=str(uuid4()),
        family=WorkbookTaskFamily.READING_COMPREHENSION,
        title=str(payload.get("title") or "Leseverstehen"),
        instructions=str(payload.get("instructions") or "Lies den Text und beantworte die Fragen."),
        prompt_seed=request_text.strip(),
        assistance_mode=WorkbookAssistanceMode.OPEN,
        word_bank=(),
        paragraphs=(),
        blanks=(),
        reading_text=str(payload.get("reading_text") or ""),
        questions=tuple(hydrate_questions(payload)),
    )


def hydrate_writing_task_payload(payload: dict, request_text: str) -> WorkbookTask:
    """Build one guided-writing task from the model payload."""
    return WorkbookTask(
        task_id=str(uuid4()),
        family=WorkbookTaskFamily.GUIDED_WRITING,
        title=str(payload.get("title") or "Schreibaufgabe"),
        instructions=str(payload.get("instructions") or "Schreibe einen zusammenhaengenden Text."),
        prompt_seed=request_text.strip(),
        assistance_mode=WorkbookAssistanceMode.OPEN,
        word_bank=(),
        paragraphs=(),
        blanks=(),
        writing_prompt=str(payload.get("writing_prompt") or ""),
        guidance_points=tuple(str(item) for item in (payload.get("guidance_points") or [])),
    )


def resolve_assistance_mode(payload: dict) -> WorkbookAssistanceMode:
    """Normalize the generated assistance mode."""
    raw_value = payload.get("assistance_mode") or WorkbookAssistanceMode.OPEN.value
    return WorkbookAssistanceMode(raw_value)


def hydrate_paragraphs(payload: dict) -> list[WorkbookParagraph]:
    """Build paragraph objects from the model payload."""
    paragraphs = payload.get("paragraphs") or []
    return [WorkbookParagraph.from_dict(item) for item in paragraphs]


def hydrate_blanks(
    payload: dict,
    assistance_mode: WorkbookAssistanceMode,
) -> list[WorkbookBlank]:
    """Build blank definitions from the model payload."""
    blanks = payload.get("blanks") or []
    return [build_blank(item, assistance_mode) for item in blanks]


def build_blank(
    payload: dict,
    assistance_mode: WorkbookAssistanceMode,
) -> WorkbookBlank:
    """Build one blank definition with normalized assistance data."""
    bracket_hint = str(payload.get("bracket_hint") or "")
    if assistance_mode != WorkbookAssistanceMode.BRACKET_HINT:
        bracket_hint = ""
    return WorkbookBlank(
        blank_id=str(payload.get("blank_id") or ""),
        prompt=str(payload.get("prompt") or ""),
        correct_answer=str(payload.get("correct_answer") or ""),
        accepted_answers=tuple(str(item) for item in (payload.get("accepted_answers") or [])),
        bracket_hint=bracket_hint,
    )


def hydrate_questions(payload: dict) -> list[WorkbookQuestion]:
    """Build reading-question definitions from the model payload."""
    questions = payload.get("questions") or []
    return [WorkbookQuestion.from_dict(item) for item in questions]


def hydrate_word_bank(
    payload: dict,
    blanks: tuple[WorkbookBlank, ...] | list[WorkbookBlank],
    assistance_mode: WorkbookAssistanceMode,
) -> list[str]:
    """Normalize the word bank based on the selected assistance mode."""
    if assistance_mode != WorkbookAssistanceMode.WORD_BANK:
        return []
    word_bank = [str(item) for item in (payload.get("word_bank") or []) if str(item).strip()]
    return word_bank or [blank.correct_answer for blank in blanks]


def parse_answers_payload(payload: list[dict]) -> tuple[WorkbookAnswer, ...]:
    """Parse serialized answers into workbook answer objects."""
    return tuple(WorkbookAnswer.from_dict(item) for item in payload)


def hydrate_validation_payload(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> WorkbookValidationResult:
    """Build a validation result with fallback correction data."""
    if task.family == WorkbookTaskFamily.GUIDED_WRITING:
        return hydrate_writing_validation_payload(task, answers, payload)
    if task.family == WorkbookTaskFamily.READING_COMPREHENSION:
        return hydrate_reading_validation_payload(task, answers, payload)
    return hydrate_fill_in_blank_validation_payload(task, answers, payload)


def hydrate_fill_in_blank_validation_payload(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> WorkbookValidationResult:
    """Build one fill-in-the-blank validation result."""
    blank_results = tuple(build_blank_results(task, answers, payload))
    return WorkbookValidationResult(
        task_id=task.task_id,
        grade=max(1, min(5, int(payload.get("grade") or 5))),
        summary=str(payload.get("summary") or ""),
        chat_explanation=str(payload.get("chat_explanation") or ""),
        blank_results=blank_results,
    )


def hydrate_reading_validation_payload(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> WorkbookValidationResult:
    """Build one reading-comprehension validation result."""
    return WorkbookValidationResult(
        task_id=task.task_id,
        grade=max(1, min(5, int(payload.get("grade") or 5))),
        summary=str(payload.get("summary") or ""),
        chat_explanation=str(payload.get("chat_explanation") or ""),
        blank_results=(),
        question_results=tuple(build_question_results(task, answers, payload)),
    )


def hydrate_writing_validation_payload(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> WorkbookValidationResult:
    """Build one guided-writing validation result."""
    return WorkbookValidationResult(
        task_id=task.task_id,
        grade=max(1, min(5, int(payload.get("grade") or 5))),
        summary=str(payload.get("summary") or ""),
        chat_explanation=str(payload.get("chat_explanation") or ""),
        blank_results=(),
        writing_result=build_writing_result(task, answers, payload),
    )


def build_blank_results(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> list[WorkbookBlankResult]:
    """Build one validation record for each submitted blank."""
    by_id = answer_map(answers)
    raw_results = payload.get("blank_results") or []
    result_map = {str(item.get("blank_id") or ""): item for item in raw_results}
    return [build_blank_result(task, blank.blank_id, by_id, result_map) for blank in task.blanks]


def answer_map(answers: tuple[WorkbookAnswer, ...]) -> dict[str, WorkbookAnswer]:
    """Index answers by their target identifier."""
    return {answer.blank_id: answer for answer in answers}


def build_question_results(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> list[WorkbookQuestionResult]:
    """Build one validation record for each reading question."""
    by_id = answer_map(answers)
    raw_results = payload.get("question_results") or []
    result_map = {str(item.get("question_id") or ""): item for item in raw_results}
    return [
        build_question_result(task, question.question_id, by_id, result_map)
        for question in task.questions
    ]


def build_writing_result(
    task: WorkbookTask,
    answers: tuple[WorkbookAnswer, ...],
    payload: dict,
) -> WorkbookWritingResult:
    """Build the validation record for one guided-writing response."""
    answer_id = task.answer_ids()[0]
    answer = answer_map(answers).get(answer_id)
    raw_result = payload.get("writing_result") or {}
    return WorkbookWritingResult(
        answer_id=str(raw_result.get("answer_id") or answer_id),
        submitted_answer=read_submitted_answer(answer, raw_result),
        corrected_answer=str(raw_result.get("corrected_answer") or ""),
        explanation=str(raw_result.get("explanation") or ""),
    )


def build_question_result(
    task: WorkbookTask,
    question_id: str,
    answers: dict[str, WorkbookAnswer],
    result_map: dict[str, dict],
) -> WorkbookQuestionResult:
    """Build one reading-question result with sensible fallbacks."""
    question = task.question_for(question_id)
    answer = answers.get(question_id)
    payload = result_map.get(question_id) or {}
    correction = str(payload.get("correction") or question.correct_answer if question else "")
    return WorkbookQuestionResult(
        question_id=question_id,
        submitted_answer=read_submitted_answer(answer, payload),
        is_correct=bool(payload.get("is_correct")),
        correction=correction,
        explanation=str(payload.get("explanation") or ""),
    )


def build_blank_result(
    task: WorkbookTask,
    blank_id: str,
    answers: dict[str, WorkbookAnswer],
    result_map: dict[str, dict],
) -> WorkbookBlankResult:
    """Build one blank result with sensible fallbacks."""
    blank = task.blank_for(blank_id)
    answer = answers.get(blank_id)
    payload = result_map.get(blank_id) or {}
    correction = str(payload.get("correction") or blank.correct_answer if blank else "")
    return WorkbookBlankResult(
        blank_id=blank_id,
        submitted_answer=read_submitted_answer(answer, payload),
        is_correct=bool(payload.get("is_correct")),
        correction=correction,
        explanation=str(payload.get("explanation") or ""),
    )


def read_submitted_answer(answer: WorkbookAnswer | None, payload: dict) -> str:
    """Read the submitted answer from either local or model state."""
    if answer is not None:
        return answer.answer_text
    return str(payload.get("submitted_answer") or "")


def build_creation_status(task: WorkbookTask) -> dict[str, object]:
    """Return the compact tool response for task generation."""
    return {
        "status": "ok",
        "task_id": task.task_id,
        "family": task.family.value,
        "title": task.title,
        "assistance_mode": task.assistance_mode.value,
        "blank_count": len(task.blanks),
        "question_count": len(task.questions),
        "vocabulary_count": len(task.vocabulary_items),
    }


def build_validation_status(result: WorkbookValidationResult) -> dict[str, object]:
    """Return the compact tool response for task validation."""
    return {
        "status": "ok",
        "task_id": result.task_id,
        "grade": result.grade,
        "summary": result.summary,
        "chat_explanation": result.chat_explanation,
    }