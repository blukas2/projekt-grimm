from nicegui import ui

from app.agents import GermanTeacherAgent


class ChatUI:
    """NiceGUI chat interface wired to a GermanTeacherAgent."""

    def __init__(self, agent: GermanTeacherAgent):
        self._agent = agent

    def build(self):
        """Construct the chat page layout."""
        ui.colors(primary="#1a1a2e")

        with ui.column().classes("w-full max-w-2xl mx-auto h-screen p-4"):
            self._build_header()
            self._build_message_area()
            self._build_input_area()

    def _build_header(self):
        """Create the title row and new lesson action."""
        with ui.row().classes("w-full items-center justify-between mb-2"):
            ui.label("Deutsch Lehrer").classes("text-2xl font-bold")
            ui.button("Neue Lektion", on_click=self._start_new_lesson).props(
                "outline"
            )

    def _build_message_area(self):
        """Create the scrollable message container."""
        self._scroll = ui.scroll_area().classes("flex-grow w-full")
        with self._scroll:
            self._messages = ui.column().classes("w-full p-2 gap-2")

    def _build_input_area(self):
        """Create the text input and send button row."""
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            self._input = ui.input(
                placeholder="Nachricht schreiben..."
            ).classes("flex-grow").on("keydown.enter", self._handle_send)

            ui.button(icon="send", on_click=self._handle_send).props(
                "round dense"
            )

    def _start_new_lesson(self):
        """Reset the UI and agent state for a new lesson."""
        self._agent.reset_lesson()
        self._input.value = ""
        self._messages.clear()

    async def _handle_send(self):
        """Process a user message and display the agent's response."""
        text = self._input.value.strip()
        if not text:
            return

        self._input.value = ""
        self._append_message(text, is_user=True)

        thinking = self._append_thinking()
        response = await self._agent.send_message(text)
        self._messages.remove(thinking)

        self._append_message(response, is_user=False)
        self._scroll.scroll_to(percent=100)

    def _append_message(self, text: str, *, is_user: bool):
        """Add a chat bubble to the message container."""
        with self._messages:
            ui.chat_message(
                text=text,
                name="Du" if is_user else "Lehrer",
                sent=is_user,
            )

    def _append_thinking(self):
        """Add a temporary 'thinking' indicator and return it."""
        with self._messages:
            return ui.chat_message(
                text="...",
                name="Lehrer",
                sent=False,
            )