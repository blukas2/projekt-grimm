import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path


LOG_DIRECTORY = Path(".logs")
LOG_FILE = LOG_DIRECTORY / "application.log"
RETENTION_WINDOW = timedelta(minutes=5)


class RecentEntriesFileHandler(logging.Handler):
    """Writes log entries to one file and prunes expired records."""

    def __init__(self, file_path: Path):
        super().__init__()
        self._file_path = file_path
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.touch(exist_ok=True)

    def emit(self, record: logging.LogRecord):
        entry = self._build_entry(record)
        self.acquire()
        try:
            self._write_entries(entry)
        finally:
            self.release()

    def _build_entry(self, record: logging.LogRecord) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).isoformat()
        message = self.format(record).replace("\n", "\\n")
        return {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

    def _write_entries(self, entry: dict[str, str]):
        entries = self._read_retained_entries()
        entries.append(entry)
        content = self._serialize_entries(entries)
        self._file_path.write_text(content, encoding="utf-8")

    def _read_retained_entries(self) -> list[dict[str, str]]:
        cutoff = datetime.now(timezone.utc) - RETENTION_WINDOW
        entries: list[dict[str, str]] = []
        for line in self._file_path.read_text(encoding="utf-8").splitlines():
            entry = self._parse_entry(line)
            if entry and self._is_recent(entry, cutoff):
                entries.append(entry)
        return entries

    def _parse_entry(self, line: str) -> dict[str, str] | None:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _is_recent(self, entry: dict[str, str], cutoff: datetime) -> bool:
        timestamp = entry.get("timestamp")
        if not timestamp:
            return False
        try:
            return datetime.fromisoformat(timestamp) >= cutoff
        except ValueError:
            return False

    def _serialize_entries(self, entries: list[dict[str, str]]) -> str:
        lines = [json.dumps(entry, ensure_ascii=True) for entry in entries]
        return "\n".join(lines) + ("\n" if lines else "")


def configure_logging() -> logging.Logger:
    """Configure the application logger for the current process."""
    logger = logging.getLogger("projekt_grimm")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RecentEntriesFileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger