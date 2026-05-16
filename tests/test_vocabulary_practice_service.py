import unittest
from random import Random

from app.core.vocabulary import VocabularyRow
from app.core.vocabulary_practice import (
    VocabularyPracticeService,
    parse_practice_word,
    practice_types_for_word,
)
from app.core.workbook import (
    WorkbookAnswer,
    WorkbookAssistanceMode,
    WorkbookPracticeType,
    WorkbookTask,
    WorkbookTaskFamily,
    WorkbookVocabularyItem,
)


class FakeVocabularyRepository:
    def __init__(self, rows):
        self._rows = list(rows)
        self.updated_strengths = {}

    def load_rows(self):
        return list(self._rows)

    def update_strengths(self, strengths_by_key):
        self.updated_strengths = dict(strengths_by_key)


class VocabularyPracticeServiceTest(unittest.TestCase):
    def test_practice_types_follow_word_eligibility(self):
        noun = parse_practice_word(
            VocabularyRow("der Apfel", "apple", "Plural: die Aepfel", 1)
        )
        verb = parse_practice_word(
            VocabularyRow(
                "gehen",
                "go",
                "3rd person singular: geht; Simple past: ging; Past participle: ist gegangen",
                1,
            )
        )
        other = parse_practice_word(VocabularyRow("schnell", "fast", "", 1))

        noun_types = set(practice_types_for_word(noun))
        verb_types = set(practice_types_for_word(verb))
        other_types = set(practice_types_for_word(other))

        self.assertIn(WorkbookPracticeType.PAIR_ARTICLE, noun_types)
        self.assertIn(WorkbookPracticeType.ALTERNATIVE_FORMS, noun_types)
        self.assertIn(WorkbookPracticeType.ALTERNATIVE_FORMS, verb_types)
        self.assertNotIn(WorkbookPracticeType.PAIR_ARTICLE, verb_types)
        self.assertNotIn(WorkbookPracticeType.PAIR_ARTICLE, other_types)
        self.assertNotIn(WorkbookPracticeType.ALTERNATIVE_FORMS, other_types)

    def test_selection_prefers_weaker_words(self):
        weak_rows = [
            VocabularyRow(f"wort-{index}", f"word-{index}", "", 1)
            for index in range(8)
        ]
        strong_rows = [
            VocabularyRow(f"stark-{index}", f"strong-{index}", "", 5)
            for index in range(4)
        ]
        repository = FakeVocabularyRepository(weak_rows + strong_rows)
        service = VocabularyPracticeService(repository, Random(7), strong_word_probability=0.0)

        weak_count = 0
        strong_count = 0
        for _ in range(40):
            task = service.create_task("wortschatz")
            for item in task.vocabulary_items:
                if item.strength_before == 1:
                    weak_count += 1
                if item.strength_before == 5:
                    strong_count += 1

        self.assertGreater(weak_count, strong_count)

    def test_selection_can_include_mastered_words(self):
        rows = [
            VocabularyRow(f"wort-{index}", f"word-{index}", "", 1)
            for index in range(10)
        ]
        rows.append(VocabularyRow("der Hund", "dog", "Plural: die Hunde", 5))
        repository = FakeVocabularyRepository(rows)
        service = VocabularyPracticeService(repository, Random(3), strong_word_probability=1.0)

        task = service.create_task("wortschatz")

        strengths = {item.strength_before for item in task.vocabulary_items}
        self.assertIn(5, strengths)

    def test_validation_updates_strengths_with_caps_and_penalties(self):
        repository = FakeVocabularyRepository([])
        service = VocabularyPracticeService(repository, Random(1))
        task = WorkbookTask(
            task_id="task-1",
            family=WorkbookTaskFamily.VOCABULARY_PRACTICE,
            title="Wortschatztraining",
            instructions="Beantworte alle Fragen.",
            prompt_seed="manual",
            assistance_mode=WorkbookAssistanceMode.OPEN,
            word_bank=(),
            paragraphs=(),
            blanks=(),
            vocabulary_items=(
                WorkbookVocabularyItem(
                    item_id="item-1",
                    practice_type=WorkbookPracticeType.PAIR_TRANSLATION,
                    prompt="Frage 1",
                    correct_answer="apple",
                    accepted_answers=("apple",),
                    choice_options=("apple", "pear"),
                    german_root="der Apfel",
                    english_translation="apple",
                    other_forms="Plural: die Aepfel",
                    strength_before=3,
                    strength_cap=3,
                ),
                WorkbookVocabularyItem(
                    item_id="item-2",
                    practice_type=WorkbookPracticeType.ENGLISH_TO_GERMAN,
                    prompt="Frage 2",
                    correct_answer="gehen",
                    accepted_answers=("gehen",),
                    choice_options=(),
                    german_root="gehen",
                    english_translation="go",
                    other_forms="3rd person singular: geht; Simple past: ging; Past participle: ist gegangen",
                    strength_before=4,
                    strength_cap=5,
                ),
            ),
        )
        answers = (
            WorkbookAnswer("item-1", "apple"),
            WorkbookAnswer("item-2", "wrong"),
        )

        result = service.validate_task(task, answers)

        first_result = result.vocabulary_result_for("item-1")
        second_result = result.vocabulary_result_for("item-2")
        self.assertEqual(first_result.strength_after, 3)
        self.assertEqual(second_result.strength_after, 3)
        self.assertEqual(len(repository.updated_strengths), 2)


if __name__ == "__main__":
    unittest.main()