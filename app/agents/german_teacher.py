import asyncio
import json
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.core.workbook import (
    WorkbookAnswer,
    WorkbookAssistanceMode,
    WorkbookEvent,
    WorkbookTask,
)

from .workbook_tools import WorkbookToolService


SYSTEM_PROMPT = """You are a German language teacher for an intermediate (B1/B2) level student.

Rules for accepting student input:
- The student must write in German.
- Exceptions where non-German input is allowed:
  1. The student asks for a translation of a word or phrase (e.g. "How do you say 'apple' in German?").
  2. The student asks to explain a German grammar rule or language concept (e.g. "Can you explain the Akkusativ?").
  3. The student describes a task or exercise they would like to do (e.g. "Give me a fill-in-the-blank exercise").
  4. The student is responding to a task that requires translating FROM German into another language.
- If the student writes in a non-German language and none of the exceptions above apply, \
politely remind them in German that they should write in German.

Rules for your responses:
- Always respond in German.
- Exception: when the task explicitly requires producing text in another language \
(e.g. translating a German sentence into English), provide that part in the target language.
- Keep your language at an intermediate level — avoid overly simple sentences but also avoid \
highly advanced vocabulary without explanation.

Grading:
- After the student completes a task or exercise, give a grade from 1 (best) to 5 (worst).
- Briefly explain in German what was good and what can be improved.
- Do NOT grade casual conversation — only grade exercises and tasks.

Behaviour:
- Be encouraging and supportive.
- Correct mistakes inline and explain the correction briefly in German.
- Proactively suggest exercises or conversation topics if the student seems unsure what to do.

Workbook tools:
- If the student asks for a workbook exercise or task, use the workbook creation tool.
- Do not print the full workbook exercise into chat when the tool is available.
- After creating a workbook task, tell the student briefly that the task is ready in the Arbeitsbuch.
- If a workbook submission needs grading, use the workbook validation tool.
- After validation, keep the chat explanation concise because the Arbeitsbuch shows the detailed corrections.
"""

MODEL = "gemini-3-flash-preview"
LOGGER = logging.getLogger("projekt_grimm.agent")


@dataclass(frozen=True)
class TeacherResponse:
    """One teacher response plus any workbook UI events."""

    text: str
    workbook_events: tuple[WorkbookEvent, ...] = ()


class GermanTeacherAgent:
    """Wraps a google-genai chat session acting as a German teacher."""

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)
        self._workbook_tools = WorkbookToolService(api_key)
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[
                self._workbook_tools.create_workbook_task,
                self._workbook_tools.validate_workbook_task,
            ],
        )
        self._chat = self._create_chat()
        LOGGER.info("German teacher agent initialized")

    async def send_message(self, user_input: str) -> TeacherResponse:
        """Send a message and return the agent's response text."""
        LOGGER.info("Student message submitted")
        response = await self._chat.send_message(user_input)
        LOGGER.info("Agent response received")
        return TeacherResponse(
            text=response.text or "",
            workbook_events=self._consume_workbook_events(),
        )

    async def generate_workbook_task(
        self,
        request_text: str,
        assistance_mode: str = WorkbookAssistanceMode.AUTO.value,
    ) -> TeacherResponse:
        """Create a workbook task from the dedicated Arbeitsbuch controls."""
        await asyncio.to_thread(
            self._workbook_tools.create_workbook_task,
            request_text,
            assistance_mode,
        )
        return TeacherResponse(text="", workbook_events=self._consume_workbook_events())

    async def validate_workbook_task(
        self,
        task: WorkbookTask,
        answers: list[WorkbookAnswer],
    ) -> TeacherResponse:
        """Validate a workbook submission from the Arbeitsbuch controls."""
        status = await asyncio.to_thread(
            self._workbook_tools.validate_workbook_task,
            json.dumps(task.to_dict(), ensure_ascii=True),
            json.dumps([answer.to_dict() for answer in answers], ensure_ascii=True),
        )
        return TeacherResponse(
            text=str(status.get("chat_explanation") or ""),
            workbook_events=self._consume_workbook_events(),
        )

    def reset_lesson(self):
        """Start a fresh chat session without prior lesson context."""
        self._chat = self._create_chat()
        self._workbook_tools.consume_events()
        LOGGER.info("Lesson reset")

    def _create_chat(self):
        """Create a new Gemini chat session with the teacher prompt."""
        return self._client.aio.chats.create(
            model=MODEL,
            config=self._config,
        )

    def _consume_workbook_events(self) -> tuple[WorkbookEvent, ...]:
        """Return and clear workbook events emitted during the last turn."""
        return self._workbook_tools.consume_events()