import os

from dotenv import load_dotenv
from nicegui import ui

from app.agents import GemmaTranslator, GermanTeacherAgent, LiveGermanTeacherAgent
from app.core import configure_logging
from app.ui import ChatUI

load_dotenv()


def main():
    """Configure dependencies and launch the NiceGUI application."""
    logger = configure_logging()
    api_key = read_api_key(logger)
    chat = build_chat_ui(api_key)
    chat.build()
    logger.info("Application started")
    ui.run(title="Deutsch Lehrer")


def read_api_key(logger):
    """Load the shared Google API key from the process environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    logger.error("Application startup failed: missing API key")
    raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before starting the app.")


def build_chat_ui(api_key: str) -> ChatUI:
    """Create the teacher chat UI and the independent translator panel."""
    teacher_agent = GermanTeacherAgent(api_key)
    live_teacher = LiveGermanTeacherAgent(api_key)
    live_teacher.register_routes()
    translator = GemmaTranslator(api_key)
    return ChatUI(teacher_agent, translator)


if __name__ in {"__main__", "__mp_main__"}:
    main()
