from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuizType(str, Enum):
    ARITHMETIC = "arithmetic"
    EQUATION = "equation"


AnswerType = Literal[int, float]


@dataclass
class QuizItem:
    """
    Represents a single math quiz.

    - `prompt`: the question to show to the user (e.g. "3 + 5 = ?")
    - `answer`: the numeric ground-truth answer
    - `difficulty`: difficulty level
    - `quiz_type`: arithmetic vs equation type
    - `meta`: optional metadata (e.g. operator, operands) for logging/analytics
    """

    prompt: str
    answer: AnswerType
    difficulty: Difficulty
    quiz_type: QuizType
    meta: dict | None = None



