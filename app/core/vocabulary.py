import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.agents.translator import VocabularyEntry


VOCABULARY_DIRECTORY_NAME = "vocabulary"
VOCABULARY_FILE_NAME = "vocabulary.csv"
CSV_FIELD_NAMES = (
    "german_root",
    "english_translation",
    "other_forms",
    "strenght",
)


@dataclass(frozen=True)
class VocabularyRow:
    """One stored vocabulary row loaded from the CSV file."""

    german_root: str
    english_translation: str
    other_forms: str
    strength: int

    def to_table_row(self) -> dict[str, str | int]:
        """Return one UI-ready row with stable keys."""
        return {
            "german_root": self.german_root,
            "english_translation": self.english_translation,
            "other_forms": self.other_forms,
            "strength": self.strength,
        }


class VocabularyRepository:
    """Persist vocabulary rows in one deduplicated CSV file."""

    def __init__(self, user_data_root: Path):
        self._directory = user_data_root / VOCABULARY_DIRECTORY_NAME
        self._file_path = self._directory / VOCABULARY_FILE_NAME
        self._ensure_file_exists()

    def load_rows(self) -> list[VocabularyRow]:
        """Read all stored vocabulary rows in file order."""
        with self._file_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [self._build_row(row) for row in reader if row]

    def add_missing_entries(self, entries: Sequence[VocabularyEntry]) -> int:
        """Append only vocabulary rows not already present in storage."""
        existing_keys = self._load_existing_keys()
        new_rows = self._build_missing_rows(entries, existing_keys)
        if new_rows:
            self._append_rows(new_rows)
        return len(new_rows)

    def replace_rows(self, rows: Sequence[VocabularyRow]):
        """Rewrite the vocabulary file with the provided rows."""
        with self._file_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELD_NAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(self._serialize_row(row))

    def _ensure_file_exists(self):
        self._directory.mkdir(parents=True, exist_ok=True)
        if self._file_path.exists() and self._file_path.stat().st_size > 0:
            return
        with self._file_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELD_NAMES)
            writer.writeheader()

    def _load_existing_keys(self) -> set[tuple[str, str]]:
        rows = self.load_rows()
        return {self._build_key(row.german_root, row.english_translation) for row in rows}

    def _build_missing_rows(
        self,
        entries: Sequence[VocabularyEntry],
        existing_keys: set[tuple[str, str]],
    ) -> list[VocabularyRow]:
        new_rows: list[VocabularyRow] = []
        for entry in entries:
            key = self._build_key(entry.german_root, entry.english_translation)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_rows.append(self._build_new_row(entry))
        return new_rows

    def _build_new_row(self, entry: VocabularyEntry) -> VocabularyRow:
        return VocabularyRow(
            german_root=entry.german_root.strip(),
            english_translation=entry.english_translation.strip(),
            other_forms=entry.other_forms.strip(),
            strength=1,
        )

    def _append_rows(self, rows: Sequence[VocabularyRow]):
        with self._file_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELD_NAMES)
            for row in rows:
                writer.writerow(self._serialize_row(row))

    def _build_row(self, payload: dict[str, str]) -> VocabularyRow:
        return VocabularyRow(
            german_root=(payload.get("german_root") or "").strip(),
            english_translation=(payload.get("english_translation") or "").strip(),
            other_forms=(payload.get("other_forms") or "").strip(),
            strength=self._parse_strength(payload.get("strenght") or "1"),
        )

    def _serialize_row(self, row: VocabularyRow) -> dict[str, str | int]:
        return {
            "german_root": row.german_root,
            "english_translation": row.english_translation,
            "other_forms": row.other_forms,
            "strenght": row.strength,
        }

    def _parse_strength(self, value: str) -> int:
        try:
            strength = int(value)
        except ValueError:
            return 1
        return min(5, max(1, strength))

    def _build_key(self, german_root: str, english_translation: str) -> tuple[str, str]:
        return self._normalize_key(german_root), self._normalize_key(english_translation)

    def _normalize_key(self, value: str) -> str:
        return " ".join(value.split()).casefold()