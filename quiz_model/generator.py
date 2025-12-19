from __future__ import annotations

from typing import Iterable, List

from .arithmetic_generator import ArithmeticGenerator
from .equation_generator import EquationGenerator
from .learned_model import (
    LearnedArithmeticGenerator,
    LearnedEquationGenerator,
    has_learned_equation_weights,
    has_learned_weights,
)
from .models import Difficulty, QuizItem, QuizType


class MathQuizGenerator:
    """
    High-level API to generate math quizzes for a web app.

    By default, it will:
    - use the *learned* arithmetic model if trained weights are available
      at `quiz_model/quiznet.pt`
    - otherwise fall back to the rule-based arithmetic generator.

    Usage:

    ```python
    gen = MathQuizGenerator(seed=42)
    item = gen.generate_one(quiz_type=QuizType.ARITHMETIC, difficulty=Difficulty.EASY)
    # item.prompt -> string to show in UI
    # item.answer -> numeric solution
    ```
    """

    def __init__(
        self,
        seed: int | None = None,
        use_learned_arithmetic: bool = False,
        use_learned_equation: bool = False,
    ) -> None:
        # use different seeds derived from base seed so results are reproducible
        arithmetic_seed = None if seed is None else seed + 1
        equation_seed = None if seed is None else seed + 2

        # Equation can be learned or rule-based.
        self._equation_rule = EquationGenerator(seed=equation_seed)
        self._equation_learned = (
            LearnedEquationGenerator.from_weights()
            if use_learned_equation and has_learned_equation_weights()
            else None
        )

        # Arithmetic can be learned or rule-based.
        self._arithmetic_rule = ArithmeticGenerator(seed=arithmetic_seed)
        self._arithmetic_learned = (
            LearnedArithmeticGenerator.from_weights()
            if use_learned_arithmetic and has_learned_weights()
            else None
        )

    def _arithmetic_backend(self):
        if self._arithmetic_learned is not None:
            return self._arithmetic_learned
        return self._arithmetic_rule

    def _equation_backend(self):
        if self._equation_learned is not None:
            return self._equation_learned
        return self._equation_rule

    def generate_one(self, quiz_type: QuizType, difficulty: Difficulty) -> QuizItem:
        if quiz_type == QuizType.ARITHMETIC:
            backend = self._arithmetic_backend()
            return backend.generate(difficulty)
        if quiz_type == QuizType.EQUATION:
            backend = self._equation_backend()
            return backend.generate(difficulty)
        raise ValueError(f"Unsupported quiz type: {quiz_type}")

    def generate_batch(
        self,
        quiz_type: QuizType,
        difficulty: Difficulty,
        n: int,
    ) -> List[QuizItem]:
        if n <= 0:
            return []
        return [self.generate_one(quiz_type=quiz_type, difficulty=difficulty) for _ in range(n)]

    def generate_mixed(
        self,
        spec: Iterable[tuple[QuizType, Difficulty, int]],
    ) -> List[QuizItem]:
        """
        Generate a mixed list of quizzes based on a specification.

        Example:
            spec = [
                (QuizType.ARITHMETIC, Difficulty.EASY, 5),
                (QuizType.EQUATION, Difficulty.MEDIUM, 3),
            ]
        """
        quizzes: List[QuizItem] = []
        for quiz_type, difficulty, n in spec:
            quizzes.extend(self.generate_batch(quiz_type, difficulty, n))
        return quizzes



