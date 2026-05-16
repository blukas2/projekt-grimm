import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv
from nicegui import ui

from app.agents import GemmaTranslator, GermanTeacherAgent, LiveGermanTeacherAgent
from app.core import VocabularyRepository, configure_logging
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
    vocabulary_repository = VocabularyRepository(read_user_data_root_folder())
    teacher_agent = GermanTeacherAgent(api_key, vocabulary_repository)
    live_teacher = LiveGermanTeacherAgent(api_key)
    live_teacher.register_routes()
    translator = GemmaTranslator(api_key)
    return ChatUI(teacher_agent, translator, vocabulary_repository)


def read_user_data_root_folder() -> Path:
    """Load the configured user-data root from app/.locations.py."""
    locations_module = load_locations_module()
    root_folder = getattr(locations_module, "USER_DATA_ROOT_FOLDER", "")
    if isinstance(root_folder, str) and root_folder.strip():
        return Path(root_folder)
    raise SystemExit("USER_DATA_ROOT_FOLDER is not configured in app/.locations.py.")


def load_locations_module():
    """Import the hidden locations module by file path."""
    locations_path = Path(__file__).resolve().parent / "app" / ".locations.py"
    spec = importlib.util.spec_from_file_location("projekt_grimm_locations", locations_path)
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load app/.locations.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ in {"__main__", "__mp_main__"}:
    main()
