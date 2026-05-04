import asyncio
import logging
from pathlib import Path

from nicegui import ui

from app.agents import GemmaTranslator, GermanTeacherAgent, TranslationError, TranslationResult
from app.core import VocabularyRepository, VocabularyRow


LOGGER = logging.getLogger("projekt_grimm.ui")
LIVE_ASSET_DIRECTORY = Path(__file__).with_name("assets")
LIVE_AUDIO_SCRIPT_FILE = LIVE_ASSET_DIRECTORY / "live_audio.js"


class ChatUI:
    """NiceGUI chat interface wired to a GermanTeacherAgent."""

    def __init__(
        self,
        agent: GermanTeacherAgent,
        translator: GemmaTranslator,
        vocabulary_repository: VocabularyRepository,
    ):
        self._agent = agent
        self._translator = translator
        self._vocabulary_repository = vocabulary_repository
        self._translator_busy = False
        self._active_tab = "chat"

    def build(self):
        """Construct the chat page layout."""
        ui.colors(primary="#1a1a2e", secondary="#0f766e")
        with ui.element("div").classes("w-full min-h-screen bg-slate-100"):
            with ui.column().classes("w-full min-h-screen gap-4 p-4"):
                self._build_header()
                self._build_main_content()
        LOGGER.info("Chat UI built")

    def _build_main_content(self):
        """Render the active lesson content below the shared header."""
        with ui.tab_panels(self._tabs, value="chat").classes("w-full flex-grow"):
            with ui.tab_panel("chat").classes("w-full p-0"):
                with ui.row().classes("w-full flex-grow items-stretch gap-4 flex-col lg:flex-row"):
                    self._build_chat_panel()
                    self._build_translator_panel()
            with ui.tab_panel("live").classes("w-full p-0"):
                self._build_live_panel()
            with ui.tab_panel("vocabulary").classes("w-full p-0"):
                self._build_vocabulary_panel()

    def _build_chat_panel(self):
        """Create the lesson chat card."""
        with ui.card().classes("w-full lg:flex-[2] shadow-sm"):
            with ui.column().classes("w-full h-full p-4 gap-0"):
                self._build_message_area()
                self._build_input_area()

    def _build_translator_panel(self):
        """Create the independent translator card."""
        with ui.card().classes("w-full lg:flex-1 shadow-sm"):
            with ui.column().classes("w-full h-full p-4 gap-4"):
                self._build_translator_header()
                self._build_translator_controls()
                self._build_translator_output()

    def _build_header(self):
        """Create the title row and new lesson action."""
        with ui.card().classes("w-full shadow-sm"):
            with ui.row().classes("w-full items-center justify-between gap-3 p-4 pb-2"):
                ui.label("Deutsch Lehrer").classes("text-2xl font-bold")
                ui.button("Neue Lektion", on_click=self._start_new_lesson).props(
                    "outline"
                )
            with ui.tabs(value="chat", on_change=self._handle_tab_change).classes(
                "w-full px-4 pb-4"
            ) as self._tabs:
                ui.tab("chat", label="Chat Erlebnis")
                ui.tab("live", label="Live Erlebnis")
                ui.tab("vocabulary", label="Wortschatz")

    def _build_live_panel(self):
        """Create the live conversation card with session controls."""
        with ui.card().classes("w-full shadow-sm"):
            with ui.column().classes("w-full p-6 gap-4"):
                ui.label("Live Erlebnis").classes("text-2xl font-bold")
                ui.label(
                    "Sprich direkt mit dem Lehrer. Das Live-Erlebnis zeigt klar an, wenn die Unterhaltung aktiv ist."
                ).classes("text-sm text-slate-600")
                ui.html(self._live_status_markup()).classes("w-full")
                with ui.row().classes("w-full items-center gap-3"):
                    ui.button(
                        "Gespraech starten",
                        on_click=self._handle_start_live,
                    ).props("id=start-live-button color=secondary")
                    ui.button(
                        "Gespraech beenden",
                        on_click=self._handle_end_live,
                    ).props("id=stop-live-button outline color=negative").classes(
                        "opacity-50 pointer-events-none"
                    )
                ui.html(self._live_transcript_markup()).classes("w-full")

    def _build_vocabulary_panel(self):
        """Create the stored vocabulary view."""
        with ui.card().classes("w-full shadow-sm"):
            with ui.column().classes("w-full p-6 gap-4"):
                self._build_vocabulary_header()
                self._build_vocabulary_table()

    def _build_vocabulary_header(self):
        """Create the vocabulary title row and refresh action."""
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-1"):
                ui.label("Wortschatz").classes("text-2xl font-bold")
                ui.label(
                    "Automatisch gespeicherte Woerter und Begriffe aus dem Uebersetzer."
                ).classes("text-sm text-slate-600")
            ui.button("Aktualisieren", on_click=self._refresh_vocabulary_table).props(
                "outline"
            )

    def _build_vocabulary_table(self):
        """Create the refreshable vocabulary table container."""
        self._vocabulary_table = ui.column().classes("w-full gap-0 rounded-lg border border-slate-200 bg-white")
        self._refresh_vocabulary_table()

    def _build_message_area(self):
        """Create the scrollable message container."""
        self._scroll = ui.scroll_area().classes("flex-grow w-full min-h-0")
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

    def _build_translator_header(self):
        """Create the translator title and helper copy."""
        ui.label("Gemma Uebersetzer").classes("text-2xl font-bold")
        ui.label(
            "Einzelne Woerter liefern Grammatikdaten, Saetze bleiben bei einer reinen Uebersetzung."
        ).classes("text-sm text-slate-600")

    def _build_translator_controls(self):
        """Create the translator direction, input, and action controls."""
        self._translator_direction = ui.toggle(
            {"en_to_de": "English -> Deutsch", "de_to_en": "Deutsch -> English"},
            value="en_to_de",
        ).props("unelevated toggle-color=secondary")
        self._translator_input = ui.textarea(
            placeholder="Wort, Phrase oder Satz eingeben..."
        ).props("outlined autogrow").classes("w-full")
        ui.button("Uebersetzen", on_click=self._handle_translate).props("color=secondary")

    def _build_translator_output(self):
        """Create the translator result area."""
        with ui.column().classes("w-full gap-2 rounded-lg bg-slate-50 p-3 min-h-48"):
            self._translator_output = ui.column().classes("w-full gap-2")
        self._show_translation_placeholder()

    def _start_new_lesson(self):
        """Reset the UI and agent state for a new lesson."""
        self._agent.reset_lesson()
        self._input.value = ""
        self._messages.clear()
        LOGGER.info("New lesson started from UI")

    def _handle_tab_change(self, event):
        """Keep track of the active Deutsch Lehrer tab."""
        self._active_tab = event.value

    async def _handle_start_live(self):
        """Load the browser controller and begin the live conversation."""
        await ui.run_javascript(self._live_controller_call("startConversation"), timeout=15.0)

    async def _handle_end_live(self):
        """Stop the current live conversation in the browser."""
        await ui.run_javascript(self._live_controller_call("stopConversation"), timeout=5.0)

    def _live_controller_call(self, action: str) -> str:
        """Return JavaScript that ensures the live controller exists and calls it."""
        script = LIVE_AUDIO_SCRIPT_FILE.read_text(encoding="utf-8")
        return f"{script}\nwindow.projektGrimmLive.{action}();"

    def _live_status_markup(self) -> str:
        """Return the browser-controlled live status markup."""
        return """
        <div class="flex items-center gap-3 rounded-lg bg-slate-50 p-4">
            <span id="live-status-badge" class="rounded-full bg-slate-500 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white">
                Live aus
            </span>
            <p id="live-status-text" class="text-sm font-medium text-slate-700">
                Noch keine Live-Unterhaltung aktiv.
            </p>
        </div>
        """

    def _live_transcript_markup(self) -> str:
        """Return the browser-controlled transcript markup."""
        return """
        <div class="rounded-lg bg-white p-4 shadow-sm ring-1 ring-slate-200">
            <div class="mb-2 flex items-center justify-between gap-3">
                <p class="text-sm font-semibold text-slate-700">Live-Protokoll</p>
                <p class="text-xs text-slate-500">Mikrofonfreigabe im Browser erforderlich</p>
            </div>
            <div id="live-transcript" class="flex min-h-32 flex-col gap-2 text-sm text-slate-700">
                <p class="text-slate-500">Starte ein Live-Gespraech, um Transkripte und Statusmeldungen zu sehen.</p>
            </div>
        </div>
        """

    async def _handle_send(self):
        """Process a user message and display the agent's response."""
        text = self._input.value.strip()
        if not text:
            return

        self._input.value = ""
        self._append_message(text, is_user=True)
        LOGGER.info("User message rendered")

        thinking = self._append_thinking()
        response = await self._agent.send_message(text)
        self._messages.remove(thinking)

        self._append_message(response, is_user=False)
        self._scroll.scroll_to(percent=100)
        LOGGER.info("Agent message rendered")

    async def _handle_translate(self):
        """Run a translation request without affecting lesson state."""
        if self._translator_busy:
            return
        source_text = self._translator_input.value.strip()
        if not source_text:
            self._show_translation_error("Bitte zuerst Text fuer die Uebersetzung eingeben.")
            return
        self._translator_busy = True
        self._show_translation_loading()
        await self._run_translation(source_text)
        self._translator_busy = False

    async def _run_translation(self, source_text: str):
        """Execute the translation request and update the result panel."""
        try:
            result = await asyncio.to_thread(
                self._translator.translate,
                source_text,
                self._translator_direction.value,
            )
        except TranslationError as exc:
            LOGGER.exception("Translator request failed in UI")
            self._show_translation_error(str(exc))
            return
        await asyncio.to_thread(
            self._vocabulary_repository.add_missing_entries,
            result.vocabulary_entries,
        )
        self._refresh_vocabulary_table()
        self._show_translation_result(result)
        LOGGER.info("Translator result rendered")

    def _refresh_vocabulary_table(self):
        """Reload the stored vocabulary rows into the table only."""
        rows = self._vocabulary_repository.load_rows()
        self._vocabulary_table.clear()
        with self._vocabulary_table:
            self._render_vocabulary_table_header()
            if rows:
                self._render_vocabulary_rows(rows)
            else:
                self._render_empty_vocabulary_row()

    def _render_vocabulary_table_header(self):
        """Render the German column headers for the vocabulary table."""
        with ui.grid().classes(
            "w-full grid-cols-1 gap-3 border-b border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-700 md:grid-cols-4"
        ):
            ui.label("Deutsches Wort")
            ui.label("Englisch")
            ui.label("Weitere Formen")
            ui.label("Staerke")

    def _render_vocabulary_rows(self, rows: list[VocabularyRow]):
        """Render all stored vocabulary rows."""
        for row in rows:
            self._render_vocabulary_row(row)

    def _render_vocabulary_row(self, row: VocabularyRow):
        """Render one vocabulary row."""
        with ui.grid().classes(
            "w-full grid-cols-1 gap-3 border-b border-slate-100 px-4 py-3 text-sm text-slate-700 last:border-b-0 md:grid-cols-4"
        ):
            ui.label(row.german_root).classes("whitespace-pre-wrap font-medium")
            ui.label(row.english_translation).classes("whitespace-pre-wrap")
            ui.label(row.other_forms or "-").classes("whitespace-pre-wrap text-slate-600")
            ui.label(str(row.strength)).classes("whitespace-pre-wrap")

    def _render_empty_vocabulary_row(self):
        """Render the empty state inside the vocabulary table."""
        with ui.row().classes("w-full px-4 py-6"):
            ui.label("Noch keine Eintraege im Wortschatz.").classes("text-sm text-slate-500")

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

    def _show_translation_placeholder(self):
        """Render the initial empty translator state."""
        self._translator_output.clear()
        with self._translator_output:
            ui.label("Noch keine Uebersetzung.").classes("text-base text-slate-500")

    def _show_translation_loading(self):
        """Render the loading state for the translator."""
        self._translator_output.clear()
        with self._translator_output:
            ui.label("Gemma uebersetzt...").classes("text-base text-slate-600")

    def _show_translation_error(self, message: str):
        """Render a translator error without touching the lesson chat."""
        self._translator_output.clear()
        with self._translator_output:
            ui.label("Uebersetzung fehlgeschlagen").classes("text-base font-medium text-red-700")
            ui.label(message).classes("whitespace-pre-wrap text-sm text-red-600")

    def _show_translation_result(self, result: TranslationResult):
        """Render a structured translator result."""
        self._translator_output.clear()
        with self._translator_output:
            ui.label(self._direction_label(result)).classes("text-xs uppercase tracking-wide text-slate-500")
            ui.label(f"Quelle: {result.source_text}").classes("text-sm text-slate-600")
            self._render_result_body(result)
            if result.result_type == "lexical":
                ui.label(self._lexical_label(result)).classes("text-xs font-medium text-secondary")

    def _render_result_body(self, result: TranslationResult):
        """Render either a plain translation or multiple lexical entries."""
        if result.result_type == "lexical":
            self._render_lexical_translations(result)
            return
        self._render_translation_text(result.display_text)

    def _render_translation_text(self, display_text: str):
        """Render the main translation in bold and details below it."""
        first_line, remaining_text = self._split_translation_text(display_text)
        ui.label(first_line).classes("whitespace-pre-wrap text-base font-bold")
        if remaining_text:
            ui.label(remaining_text).classes("whitespace-pre-wrap text-base")

    def _render_lexical_translations(self, result: TranslationResult):
        """Render each relevant lexical translation as its own block."""
        for index, lexical_translation in enumerate(result.lexical_translations):
            self._render_translation_text(lexical_translation.display_text)
            if index < len(result.lexical_translations) - 1:
                ui.separator().classes("w-full my-1")

    def _split_translation_text(self, display_text: str) -> tuple[str, str]:
        """Split translation output into its headline and detail lines."""
        lines = display_text.splitlines()
        if not lines:
            return "", ""
        return lines[0], "\n".join(lines[1:])

    def _direction_label(self, result: TranslationResult) -> str:
        """Return a short direction label for the result panel."""
        if result.direction == "en_to_de":
            return "English to Deutsch"
        return "Deutsch to English"

    def _lexical_label(self, result: TranslationResult) -> str:
        """Return a short lexical badge label."""
        translation_count = len(result.lexical_translations)
        if translation_count == 1:
            return f"Lexical lookup: {result.lexical_type}"
        return f"Lexical lookup: {translation_count} relevant translations"