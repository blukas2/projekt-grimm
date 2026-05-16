import unittest

from app.core.workbook import (
    WorkbookAssistanceMode,
    WorkbookPracticeType,
    WorkbookTask,
    WorkbookTaskFamily,
    WorkbookValidationResult,
    WorkbookVocabularyItem,
    WorkbookVocabularyResult,
)


class WorkbookVocabularyPracticeSerializationTest(unittest.TestCase):
    def test_task_round_trip_preserves_vocabulary_items(self):
        task = WorkbookTask(
            task_id="task-1",
            family=WorkbookTaskFamily.VOCABULARY_PRACTICE,
            title="Wortschatztraining",
            instructions="Beantworte alle Fragen.",
            prompt_seed="vocabulary practice",
            assistance_mode=WorkbookAssistanceMode.OPEN,
            word_bank=(),
            paragraphs=(),
            blanks=(),
            vocabulary_items=(
                WorkbookVocabularyItem(
                    item_id="item-1",
                    practice_type=WorkbookPracticeType.GERMAN_TO_ENGLISH,
                    prompt="Was bedeutet 'gehen'?",
                    correct_answer="to go",
                    accepted_answers=("to go", "go"),
                    choice_options=(),
                    german_root="gehen",
                    english_translation="go",
                    other_forms="3rd person singular: geht",
                    strength_before=3,
                    strength_cap=4,
                ),
            ),
        )

        restored = WorkbookTask.from_dict(task.to_dict())

        self.assertEqual(restored.family, WorkbookTaskFamily.VOCABULARY_PRACTICE)
        self.assertEqual(restored.answer_ids(), ("item-1",))
        self.assertEqual(restored.vocabulary_items[0].practice_type, WorkbookPracticeType.GERMAN_TO_ENGLISH)

    def test_validation_round_trip_preserves_strengths(self):
        result = WorkbookValidationResult(
            task_id="task-1",
            grade=2,
            summary="Fast alles richtig.",
            chat_explanation="Kurze Rueckmeldung",
            blank_results=(),
            vocabulary_results=(
                WorkbookVocabularyResult(
                    item_id="item-1",
                    submitted_answer="go",
                    is_correct=True,
                    correction="go",
                    explanation="Richtig.",
                    strength_before=3,
                    strength_after=4,
                ),
            ),
        )

        restored = WorkbookValidationResult.from_dict(result.to_dict())

        self.assertEqual(restored.vocabulary_results[0].strength_before, 3)
        self.assertEqual(restored.vocabulary_results[0].strength_after, 4)


if __name__ == "__main__":
    unittest.main()