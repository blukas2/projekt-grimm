import asyncio
import logging
from pathlib import Path

from nicegui import ui

from app.agents import GemmaTranslator, GermanTeacherAgent, TranslationError, TranslationResult
from app.core import (
    VocabularyRepository,
    VocabularyRow,
    WorkbookAnswer,
    WorkbookAssistanceMode,
    WorkbookEventKind,
    WorkbookSegmentKind,
    WorkbookTask,
    WorkbookValidationResult,
)


LOGGER = logging.getLogger("projekt_grimm.ui")
LIVE_ASSET_DIRECTORY = Path(__file__).with_name("assets")
LIVE_AUDIO_SCRIPT_FILE = LIVE_ASSET_DIRECTORY / "live_audio.js"
VOCABULARY_SORT_COLUMNS = (
    ("german_root", "Deutsches Wort"),
    ("english_translation", "Englisch"),
    ("other_forms", "Weitere Formen"),
    ("strength", "Staerke"),
)
GERMAN_SORT_PREFIXES = {"der", "die", "das"}


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
        self._active_tab = "arbeitsbuch"
        self._chat_history: list[tuple[str, bool]] = []
        self._chat_draft = ""
        self._assistant_is_thinking = False
        self._workbook_busy = False
        self._workbook_task: WorkbookTask | None = None
        self._workbook_validation: WorkbookValidationResult | None = None
        self._workbook_error_message = ""
        self._workbook_request_text = ""
        self._workbook_assistance_mode_value = WorkbookAssistanceMode.AUTO.value
        self._workbook_answers: dict[str, str] = {}
        self._chat_message_columns = []
        self._chat_scroll_areas = []
        self._chat_inputs = []
        self._workbook_request_inputs = []
        self._workbook_assistance_selects = []
        self._workbook_output_columns = []
        self._translator_direction_value = "en_to_de"
        self._translator_source_text = ""
        self._translator_output_state = "placeholder"
        self._translator_error_message = ""
        self._translator_result: TranslationResult | None = None
        self._translator_direction_toggles = []
        self._translator_input_fields = []
        self._translator_output_columns = []
        self._vocabulary_sort_column = "german_root"
        self._vocabulary_sort_descending = False

    def build(self):
        """Construct the chat page layout."""
        ui.colors(primary="#1a1a2e", secondary="#0f766e")
        with ui.element("div").classes("w-full h-screen bg-slate-100 overflow-hidden"):
            with ui.column().classes("w-full h-full gap-4 p-4 min-h-0"):
                self._build_header()
                self._build_main_content()
        LOGGER.info("Chat UI built")

    def _build_main_content(self):
        """Render the active lesson content below the shared header."""
        with ui.tab_panels(self._tabs, value="arbeitsbuch").classes("w-full flex-grow min-h-0"):
            with ui.tab_panel("arbeitsbuch").classes("w-full h-full p-0"):
                self._build_arbeitsbuch_panel()
            with ui.tab_panel("chat").classes("w-full h-full p-0"):
                with ui.row().classes("w-full h-full min-h-0 items-stretch gap-4 flex-col lg:flex-row"):
                    self._build_chat_panel()
                    self._build_translator_panel()
            with ui.tab_panel("live").classes("w-full p-0"):
                self._build_live_panel()
            with ui.tab_panel("vocabulary").classes("w-full p-0"):
                self._build_vocabulary_panel()

    def _build_arbeitsbuch_panel(self):
        """Create the workbook layout with a large task area and right sidebar."""
        with ui.row().classes("w-full h-full min-h-0 items-stretch gap-4 flex-col lg:flex-row"):
            self._build_workbook_panel()
            with ui.column().classes("w-full h-full min-h-0 gap-4 lg:flex-1"):
                self._build_chat_panel(card_classes="w-full min-h-0 flex-[3] shadow-sm")
                self._build_translator_panel(
                    card_classes="w-full min-h-0 flex-[2] shadow-sm"
                )

    def _build_workbook_panel(self):
        """Create the workbook area with generation controls and task state."""
        with ui.card().classes("w-full h-full min-h-0 lg:flex-[2] shadow-sm"):
            with ui.column().classes("w-full h-full min-h-0 p-6 gap-4 overflow-auto"):
                self._build_workbook_header()
                self._build_workbook_controls()
                workbook_output = ui.column().classes("w-full gap-4")
        self._workbook_output_columns.append(workbook_output)
        self._render_workbook_output(workbook_output)

    def _build_workbook_header(self):
        """Create the workbook title and helper copy."""
        ui.label("Arbeitsbuch").classes("text-3xl font-bold")
        ui.label(
            "Hier erscheinen Aufgaben, die der Lehrer fuer dich erstellt oder bewertet."
        ).classes("text-base text-slate-700")

    def _build_workbook_controls(self):
        """Create the dedicated workbook generation controls."""
        with ui.card().classes("w-full bg-slate-50 shadow-none ring-1 ring-slate-200"):
            with ui.column().classes("w-full gap-3 p-4"):
                ui.label("Neue Aufgabe").classes("text-lg font-semibold text-slate-800")
                request_input = ui.input(
                    placeholder="z. B. Artikel, Praeteritum oder trennbare Verben",
                    value=self._workbook_request_text,
                ).classes("w-full")
                request_input.on(
                    "update:model-value",
                    lambda _: self._sync_workbook_request_text(request_input.value or ""),
                )
                self._workbook_request_inputs.append(request_input)
                assistance_select = ui.select(
                    {
                        WorkbookAssistanceMode.AUTO.value: "Automatisch",
                        WorkbookAssistanceMode.BRACKET_HINT.value: "Form in Klammern",
                        WorkbookAssistanceMode.WORD_BANK.value: "Wortliste",
                        WorkbookAssistanceMode.OPEN.value: "Ohne Hilfe",
                    },
                    value=self._workbook_assistance_mode_value,
                    label="Hilfsmodus",
                ).classes("w-full")
                assistance_select.on(
                    "update:model-value",
                    lambda _: self._sync_workbook_assistance_mode(assistance_select.value),
                )
                self._workbook_assistance_selects.append(assistance_select)
                ui.button(
                    "Aufgabe erstellen",
                    on_click=self._generate_workbook_from_controls,
                ).props("color=secondary")

    def _build_chat_panel(self, card_classes: str = "w-full h-full min-h-0 lg:flex-[2] shadow-sm"):
        """Create the lesson chat card."""
        with ui.card().classes(card_classes):
            with ui.column().classes("w-full h-full p-4 gap-0"):
                self._build_message_area()
                self._build_input_area()

    def _build_translator_panel(self, card_classes: str = "w-full h-full min-h-0 lg:flex-1 shadow-sm"):
        """Create the independent translator card."""
        with ui.card().classes(card_classes):
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
            with ui.tabs(value="arbeitsbuch", on_change=self._handle_tab_change).classes(
                "w-full px-4 pb-4"
            ) as self._tabs:
                ui.tab("arbeitsbuch", label="Arbeitsbuch")
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
        scroll_area = ui.scroll_area().classes("flex-grow w-full min-h-0")
        self._chat_scroll_areas.append(scroll_area)
        with scroll_area:
            messages = ui.column().classes("w-full p-2 gap-2")
        self._chat_message_columns.append(messages)
        self._render_chat_messages(messages)

    def _build_input_area(self):
        """Create the text input and send button row."""
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            chat_input = ui.input(
                placeholder="Nachricht schreiben...",
                value=self._chat_draft,
            ).classes("flex-grow")
            chat_input.on(
                "keydown.enter",
                lambda _: self._send_from_chat_input(chat_input),
            )
            chat_input.on(
                "update:model-value",
                lambda _: self._sync_chat_draft(chat_input.value or ""),
            )
            self._chat_inputs.append(chat_input)

            ui.button(
                icon="send",
                on_click=lambda: self._send_from_chat_input(chat_input),
            ).props(
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
        direction_toggle = ui.toggle(
            {"en_to_de": "English -> Deutsch", "de_to_en": "Deutsch -> English"},
            value=self._translator_direction_value,
        ).props("unelevated toggle-color=secondary")
        direction_toggle.on(
            "update:model-value",
            lambda _: self._sync_translator_direction(direction_toggle.value),
        )
        self._translator_direction_toggles.append(direction_toggle)
        translator_input = ui.textarea(
            placeholder="Wort, Phrase oder Satz eingeben...",
            value=self._translator_source_text,
        ).props("outlined autogrow").classes("w-full")
        translator_input.on(
            "update:model-value",
            lambda _: self._sync_translator_source_text(translator_input.value or ""),
        )
        self._translator_input_fields.append(translator_input)
        ui.button(
            "Uebersetzen",
            on_click=lambda: self._translate_from_controls(
                translator_input,
                direction_toggle,
            ),
        ).props("color=secondary")

    def _build_translator_output(self):
        """Create the translator result area."""
        with ui.column().classes("w-full gap-2 rounded-lg bg-slate-50 p-3 min-h-48"):
            translator_output = ui.column().classes("w-full gap-2")
        self._translator_output_columns.append(translator_output)
        self._render_translator_output(translator_output)

    def _start_new_lesson(self):
        """Reset the UI and agent state for a new lesson."""
        self._agent.reset_lesson()
        self._chat_history.clear()
        self._assistant_is_thinking = False
        self._show_workbook_placeholder()
        self._sync_chat_draft("")
        self._refresh_chat_views()
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
        text = self._chat_draft.strip()
        if not text:
            return

        self._sync_chat_draft("")
        self._append_message(text, is_user=True)
        LOGGER.info("User message rendered")

        self._assistant_is_thinking = True
        self._refresh_chat_views()
        response = await self._agent.send_message(text)
        self._assistant_is_thinking = False
        self._apply_workbook_events(response.workbook_events)
        if response.text.strip():
            self._append_message(response.text, is_user=False)
            LOGGER.info("Agent message rendered")

    async def _handle_send_from_input(self, chat_input):
        """Submit a message using the currently active chat input."""
        self._sync_chat_draft(chat_input.value or "")
        await self._handle_send()

    def _send_from_chat_input(self, chat_input):
        """Schedule a send request from one chat input field."""
        asyncio.create_task(self._handle_send_from_input(chat_input))

    async def _handle_translate(self):
        """Run a translation request without affecting lesson state."""
        if self._translator_busy:
            return
        source_text = self._translator_source_text.strip()
        if not source_text:
            self._show_translation_error("Bitte zuerst Text fuer die Uebersetzung eingeben.")
            return
        self._translator_busy = True
        self._show_translation_loading()
        await self._run_translation(source_text)
        self._translator_busy = False

    async def _handle_translate_from_controls(self, translator_input, direction_toggle):
        """Run translation using the active workbook or chat translator controls."""
        self._sync_translator_source_text(translator_input.value or "")
        self._sync_translator_direction(direction_toggle.value)
        await self._handle_translate()

    def _translate_from_controls(self, translator_input, direction_toggle):
        """Schedule translation from one translator control set."""
        asyncio.create_task(
            self._handle_translate_from_controls(translator_input, direction_toggle)
        )

    async def _handle_generate_workbook_task(self):
        """Create one workbook task from the dedicated controls."""
        if self._workbook_busy:
            return
        request_text = self._workbook_request_text.strip()
        if not request_text:
            self._show_workbook_error("Bitte zuerst ein Thema oder eine Uebungsidee eingeben.")
            return
        self._show_workbook_loading()
        try:
            response = await self._agent.generate_workbook_task(
                request_text,
                self._workbook_assistance_mode_value,
            )
        except Exception:
            LOGGER.exception("Workbook generation failed in UI")
            self._show_workbook_error("Die Aufgabe konnte gerade nicht erstellt werden.")
            return
        self._workbook_busy = False
        self._apply_workbook_events(response.workbook_events)
        if response.text.strip():
            self._append_message(response.text, is_user=False)

    def _generate_workbook_from_controls(self):
        """Schedule workbook generation from the dedicated controls."""
        asyncio.create_task(self._handle_generate_workbook_task())

    async def _handle_submit_workbook_task(self):
        """Validate the active workbook task and mirror the explanation to chat."""
        if self._workbook_busy or self._workbook_task is None:
            return
        answers = self._build_workbook_answers()
        if self._has_empty_workbook_answers(answers):
            self._show_workbook_error("Bitte alle Luecken ausfuellen, bevor du abgibst.")
            return
        self._show_workbook_loading()
        try:
            response = await self._agent.validate_workbook_task(self._workbook_task, answers)
        except Exception:
            LOGGER.exception("Workbook validation failed in UI")
            self._show_workbook_error("Die Aufgabe konnte gerade nicht bewertet werden.")
            return
        self._workbook_busy = False
        self._apply_workbook_events(response.workbook_events)
        if response.text.strip():
            self._append_message(response.text, is_user=False)

    def _submit_workbook_task(self):
        """Schedule workbook validation from the Arbeitsbuch controls."""
        asyncio.create_task(self._handle_submit_workbook_task())

    async def _run_translation(self, source_text: str):
        """Execute the translation request and update the result panel."""
        try:
            result = await asyncio.to_thread(
                self._translator.translate,
                source_text,
                self._translator_direction_value,
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
        rows = self._sorted_vocabulary_rows(self._vocabulary_repository.load_rows())
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
            for column_name, label in VOCABULARY_SORT_COLUMNS:
                self._render_vocabulary_sort_button(column_name, label)

    def _render_vocabulary_sort_button(self, column_name: str, label: str):
        """Render one sortable vocabulary header control."""
        ui.button(
            self._vocabulary_header_label(column_name, label),
            on_click=lambda sort_column=column_name: self._handle_vocabulary_sort(sort_column),
        ).props("flat no-caps dense align=left").classes(
            self._vocabulary_header_classes(column_name)
        )

    def _vocabulary_header_label(self, column_name: str, label: str) -> str:
        """Return the current header label including sort direction."""
        if column_name != self._vocabulary_sort_column:
            return label
        direction = "v" if self._vocabulary_sort_descending else "^"
        return f"{label} {direction}"

    def _vocabulary_header_classes(self, column_name: str) -> str:
        """Return the classes for one vocabulary header button."""
        base_classes = "w-full justify-start rounded-md px-2 py-1 text-left text-sm font-semibold"
        if column_name == self._vocabulary_sort_column:
            return f"{base_classes} bg-white text-secondary shadow-sm"
        return f"{base_classes} text-slate-700"

    def _handle_vocabulary_sort(self, column_name: str):
        """Toggle the current table sort and refresh the display."""
        if column_name == self._vocabulary_sort_column:
            self._vocabulary_sort_descending = not self._vocabulary_sort_descending
        else:
            self._vocabulary_sort_column = column_name
            self._vocabulary_sort_descending = False
        self._refresh_vocabulary_table()

    def _sorted_vocabulary_rows(self, rows: list[VocabularyRow]) -> list[VocabularyRow]:
        """Return vocabulary rows sorted for the current UI state."""
        return sorted(
            rows,
            key=self._vocabulary_sort_key,
            reverse=self._vocabulary_sort_descending,
        )

    def _vocabulary_sort_key(self, row: VocabularyRow):
        """Return the active sort key for one vocabulary row."""
        if self._vocabulary_sort_column == "strength":
            return row.strength
        value = getattr(row, self._vocabulary_sort_column)
        if self._vocabulary_sort_column == "german_root":
            return self._normalize_german_sort_value(value)
        return self._normalize_text_sort_value(value)

    def _normalize_text_sort_value(self, value: str) -> str:
        """Normalize plain text values for case-insensitive sorting."""
        return " ".join(value.split()).casefold()

    def _normalize_german_sort_value(self, value: str) -> str:
        """Normalize German nouns while ignoring a leading article."""
        normalized = " ".join(value.split())
        parts = normalized.split(" ", 1)
        if len(parts) == 2 and parts[0].casefold() in GERMAN_SORT_PREFIXES:
            return parts[1].casefold()
        return normalized.casefold()

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
        self._chat_history.append((text, is_user))
        self._refresh_chat_views()

    def _sync_chat_draft(self, value: str):
        """Keep all chat inputs aligned to the same draft text."""
        self._chat_draft = value
        for chat_input in self._chat_inputs:
            if chat_input.value != value:
                chat_input.value = value

    def _refresh_chat_views(self):
        """Redraw all chat message columns and keep them scrolled to the bottom."""
        for messages in self._chat_message_columns:
            self._render_chat_messages(messages)
        for scroll_area in self._chat_scroll_areas:
            scroll_area.scroll_to(percent=100)

    def _sync_workbook_request_text(self, value: str):
        """Keep all workbook request inputs aligned."""
        self._workbook_request_text = value
        for request_input in self._workbook_request_inputs:
            if request_input.value != value:
                request_input.value = value

    def _sync_workbook_assistance_mode(self, value: str):
        """Keep all workbook assistance selectors aligned."""
        self._workbook_assistance_mode_value = value
        for assistance_select in self._workbook_assistance_selects:
            if assistance_select.value != value:
                assistance_select.value = value

    def _build_workbook_answers(self) -> list[WorkbookAnswer]:
        """Return the current answers in task order."""
        if self._workbook_task is None:
            return []
        return [
            WorkbookAnswer(blank.blank_id, self._workbook_answers.get(blank.blank_id, ""))
            for blank in self._workbook_task.blanks
        ]

    def _has_empty_workbook_answers(self, answers: list[WorkbookAnswer]) -> bool:
        """Return whether any workbook blank is still empty."""
        return any(not answer.answer_text.strip() for answer in answers)

    def _apply_workbook_events(self, events):
        """Apply workbook events emitted by the teacher agent."""
        for event in events:
            if event.kind == WorkbookEventKind.TASK_GENERATED and event.task is not None:
                self._show_workbook_task(event.task)
            if event.kind == WorkbookEventKind.TASK_VALIDATED and event.validation is not None:
                self._show_workbook_validation(event.validation)

    def _show_workbook_placeholder(self):
        """Reset the workbook area to its initial empty state."""
        self._workbook_busy = False
        self._workbook_task = None
        self._workbook_validation = None
        self._workbook_error_message = ""
        self._workbook_answers = {}
        self._sync_workbook_request_text("")
        self._refresh_workbook_views()

    def _show_workbook_loading(self):
        """Render the workbook loading state without clearing the active task."""
        self._workbook_busy = True
        self._workbook_error_message = ""
        self._refresh_workbook_views()

    def _show_workbook_error(self, message: str):
        """Render a workbook-local error state."""
        self._workbook_busy = False
        self._workbook_error_message = message
        self._refresh_workbook_views()

    def _show_workbook_task(self, task: WorkbookTask):
        """Load one newly generated workbook task."""
        self._workbook_busy = False
        self._workbook_task = task
        self._workbook_validation = None
        self._workbook_error_message = ""
        self._workbook_answers = {blank.blank_id: "" for blank in task.blanks}
        self._switch_to_workbook_tab()
        self._refresh_workbook_views()

    def _show_workbook_validation(self, validation: WorkbookValidationResult):
        """Render the validation state for the current workbook task."""
        self._workbook_busy = False
        self._workbook_validation = validation
        self._workbook_error_message = ""
        self._switch_to_workbook_tab()
        self._refresh_workbook_views()

    def _switch_to_workbook_tab(self):
        """Focus the Arbeitsbuch tab after a workbook event."""
        self._active_tab = "arbeitsbuch"
        self._tabs.value = "arbeitsbuch"

    def _refresh_workbook_views(self):
        """Redraw all workbook output panes from shared state."""
        for workbook_output in self._workbook_output_columns:
            self._render_workbook_output(workbook_output)

    def _render_workbook_output(self, workbook_output):
        """Render the workbook state inside one output pane."""
        workbook_output.clear()
        with workbook_output:
            if self._workbook_error_message:
                self._render_workbook_error()
            if self._workbook_busy:
                ui.label("Arbeitsbuch wird aktualisiert...").classes("text-sm text-slate-500")
            if self._workbook_task is None:
                self._render_workbook_placeholder()
                return
            self._render_workbook_task(self._workbook_task)
            if self._workbook_validation is not None:
                self._render_workbook_results(self._workbook_task, self._workbook_validation)

    def _render_workbook_placeholder(self):
        """Render the empty workbook state."""
        with ui.card().classes("w-full bg-white shadow-none ring-1 ring-slate-200"):
            with ui.column().classes("w-full gap-2 p-5"):
                ui.label("Noch keine Aufgabe vorhanden").classes(
                    "text-lg font-semibold text-slate-800"
                )
                ui.label(
                    "Nutze den Chat oder die Schaltflaeche oben, um eine Aufgabe zu erstellen."
                ).classes("text-sm text-slate-600")

    def _render_workbook_error(self):
        """Render the workbook error banner."""
        with ui.card().classes("w-full bg-red-50 shadow-none ring-1 ring-red-200"):
            with ui.column().classes("w-full gap-1 p-4"):
                ui.label("Arbeitsbuch-Fehler").classes("text-base font-semibold text-red-700")
                ui.label(self._workbook_error_message).classes("text-sm text-red-600")

    def _render_workbook_task(self, task: WorkbookTask):
        """Render the active workbook task and its inline inputs."""
        with ui.card().classes("w-full bg-white shadow-none ring-1 ring-slate-200"):
            with ui.column().classes("w-full gap-4 p-5"):
                ui.label(task.title).classes("text-2xl font-bold text-slate-900")
                ui.label(task.instructions).classes("whitespace-pre-wrap text-sm text-slate-700")
                self._render_workbook_word_bank(task)
                self._render_workbook_paragraphs(task)
                ui.button("Loesung abgeben", on_click=self._submit_workbook_task).props(
                    "color=secondary"
                )

    def _render_workbook_word_bank(self, task: WorkbookTask):
        """Render the optional workbook word bank."""
        if task.assistance_mode != WorkbookAssistanceMode.WORD_BANK or not task.word_bank:
            return
        with ui.row().classes("w-full flex-wrap gap-2 rounded-lg bg-slate-50 p-3"):
            ui.label("Wortliste:").classes("text-sm font-medium text-slate-700")
            for word in task.word_bank:
                ui.badge(word).props("outline color=secondary")

    def _render_workbook_paragraphs(self, task: WorkbookTask):
        """Render all workbook paragraphs with inline answer fields."""
        for paragraph in task.paragraphs:
            self._render_workbook_paragraph(task, paragraph)

    def _render_workbook_paragraph(self, task: WorkbookTask, paragraph):
        """Render one paragraph of the active workbook task."""
        with ui.row().classes(
            "w-full flex-wrap items-end gap-2 rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200"
        ):
            for segment in paragraph.segments:
                if segment.kind == WorkbookSegmentKind.TEXT:
                    ui.label(segment.text).classes("whitespace-pre-wrap text-base text-slate-800")
                    continue
                self._render_workbook_input_segment(task, segment.blank_id)

    def _render_workbook_input_segment(self, task: WorkbookTask, blank_id: str):
        """Render one editable blank inside the workbook text."""
        blank = task.blank_for(blank_id)
        answer_value = self._workbook_answers.get(blank_id, "")
        input_field = ui.input(value=answer_value, placeholder="...").classes("w-28")
        input_field.props("outlined dense")
        input_field.on(
            "update:model-value",
            lambda event, current_blank=blank_id: self._sync_workbook_answer(
                current_blank,
                event.args or "",
            ),
        )
        if blank is not None and blank.bracket_hint:
            ui.label(f"({blank.bracket_hint})").classes("text-sm text-slate-500")

    def _sync_workbook_answer(self, blank_id: str, value: str):
        """Store the current answer for one workbook blank."""
        self._workbook_answers[blank_id] = value

    def _render_workbook_results(
        self,
        task: WorkbookTask,
        validation: WorkbookValidationResult,
    ):
        """Render the validated workbook result below the active task."""
        with ui.card().classes("w-full bg-emerald-50 shadow-none ring-1 ring-emerald-200"):
            with ui.column().classes("w-full gap-2 p-5"):
                ui.label(f"Bewertung: Note {validation.grade}").classes(
                    "text-lg font-semibold text-emerald-800"
                )
                ui.label(validation.summary).classes("whitespace-pre-wrap text-sm text-emerald-700")
        self._render_workbook_user_solution(task, validation)
        self._render_workbook_corrected_solution(task, validation)

    def _render_workbook_user_solution(
        self,
        task: WorkbookTask,
        validation: WorkbookValidationResult,
    ):
        """Render the submitted workbook solution with inline marking."""
        with ui.card().classes("w-full bg-white shadow-none ring-1 ring-slate-200"):
            with ui.column().classes("w-full gap-3 p-5"):
                ui.label("Deine Loesung").classes("text-lg font-semibold text-slate-900")
                self._render_workbook_solution_text(task, validation, corrected=False)

    def _render_workbook_corrected_solution(
        self,
        task: WorkbookTask,
        validation: WorkbookValidationResult,
    ):
        """Render the corrected workbook solution."""
        with ui.card().classes("w-full bg-white shadow-none ring-1 ring-slate-200"):
            with ui.column().classes("w-full gap-3 p-5"):
                ui.label("Korrigierte Version").classes("text-lg font-semibold text-slate-900")
                self._render_workbook_solution_text(task, validation, corrected=True)

    def _render_workbook_solution_text(
        self,
        task: WorkbookTask,
        validation: WorkbookValidationResult,
        *,
        corrected: bool,
    ):
        """Render one workbook solution view using either answers or corrections."""
        for paragraph in task.paragraphs:
            with ui.row().classes(
                "w-full flex-wrap items-end gap-2 rounded-lg bg-slate-50 p-4 ring-1 ring-slate-200"
            ):
                for segment in paragraph.segments:
                    if segment.kind == WorkbookSegmentKind.TEXT:
                        ui.label(segment.text).classes("whitespace-pre-wrap text-base text-slate-800")
                        continue
                    self._render_workbook_solution_segment(validation, segment.blank_id, corrected)

    def _render_workbook_solution_segment(
        self,
        validation: WorkbookValidationResult,
        blank_id: str,
        corrected: bool,
    ):
        """Render one workbook blank inside a validated solution view."""
        result = validation.result_for(blank_id)
        if result is None:
            with ui.element("span").classes(
                "inline-flex items-center rounded-md bg-slate-400 px-3 py-1 text-sm font-semibold text-white"
            ):
                ui.label("-").classes("text-inherit")
            return
        label = result.correction if corrected else (result.submitted_answer or "-")
        with ui.element("span").classes(self._result_badge_classes(result, corrected)):
            ui.label(label).classes("text-inherit")
        if corrected or not result.explanation:
            return
        ui.label(result.explanation).classes("text-xs text-slate-500")

    def _result_badge_classes(self, result, corrected: bool) -> str:
        """Return the visual style for one validated blank."""
        base_classes = "inline-flex items-center rounded-md px-3 py-1 text-sm font-semibold text-white"
        if corrected:
            return f"{base_classes} bg-slate-600"
        if result.is_correct:
            return f"{base_classes} bg-emerald-600"
        return f"{base_classes} bg-red-600"

    def _render_chat_messages(self, messages):
        """Render the full conversation history inside one chat column."""
        messages.clear()
        with messages:
            for text, is_user in self._chat_history:
                ui.chat_message(
                    text=text,
                    name="Du" if is_user else "Lehrer",
                    sent=is_user,
                )
            if self._assistant_is_thinking:
                ui.chat_message(text="...", name="Lehrer", sent=False)

    def _sync_translator_direction(self, value: str):
        """Keep all translator direction toggles aligned."""
        self._translator_direction_value = value
        for direction_toggle in self._translator_direction_toggles:
            if direction_toggle.value != value:
                direction_toggle.value = value

    def _sync_translator_source_text(self, value: str):
        """Keep all translator textareas aligned."""
        self._translator_source_text = value
        for translator_input in self._translator_input_fields:
            if translator_input.value != value:
                translator_input.value = value

    def _show_translation_placeholder(self):
        """Render the initial empty translator state."""
        self._translator_output_state = "placeholder"
        self._translator_error_message = ""
        self._translator_result = None
        self._refresh_translator_outputs()

    def _show_translation_loading(self):
        """Render the loading state for the translator."""
        self._translator_output_state = "loading"
        self._translator_error_message = ""
        self._translator_result = None
        self._refresh_translator_outputs()

    def _show_translation_error(self, message: str):
        """Render a translator error without touching the lesson chat."""
        self._translator_output_state = "error"
        self._translator_error_message = message
        self._translator_result = None
        self._refresh_translator_outputs()

    def _show_translation_result(self, result: TranslationResult):
        """Render a structured translator result."""
        self._translator_output_state = "result"
        self._translator_error_message = ""
        self._translator_result = result
        self._refresh_translator_outputs()

    def _refresh_translator_outputs(self):
        """Redraw all translator output panes from the shared state."""
        for translator_output in self._translator_output_columns:
            self._render_translator_output(translator_output)

    def _render_translator_output(self, translator_output):
        """Render the translator output state inside one output pane."""
        translator_output.clear()
        with translator_output:
            if self._translator_output_state == "loading":
                ui.label("Gemma uebersetzt...").classes("text-base text-slate-600")
                return
            if self._translator_output_state == "error":
                ui.label("Uebersetzung fehlgeschlagen").classes(
                    "text-base font-medium text-red-700"
                )
                ui.label(self._translator_error_message).classes(
                    "whitespace-pre-wrap text-sm text-red-600"
                )
                return
            if self._translator_output_state != "result" or self._translator_result is None:
                ui.label("Noch keine Uebersetzung.").classes("text-base text-slate-500")
                return
            ui.label(self._direction_label(self._translator_result)).classes(
                "text-xs uppercase tracking-wide text-slate-500"
            )
            ui.label(f"Quelle: {self._translator_result.source_text}").classes(
                "text-sm text-slate-600"
            )
            self._render_result_body(self._translator_result)
            if self._translator_result.result_type == "lexical":
                ui.label(self._lexical_label(self._translator_result)).classes(
                    "text-xs font-medium text-secondary"
                )

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