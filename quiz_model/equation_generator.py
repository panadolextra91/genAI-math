from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Difficulty, QuizItem, QuizType


@dataclass
class _LinearCoeffs:
    a: int
    b: int
    x: int


class EquationGenerator:
    """Generate simple algebraic equation quizzes."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def _linear_coeffs(self, difficulty: Difficulty) -> _LinearCoeffs:
        if difficulty == Difficulty.EASY:
            a = self._rng.randint(1, 10)
            x = self._rng.randint(0, 10)
            b = self._rng.randint(0, 10)
        elif difficulty == Difficulty.MEDIUM:
            a = self._rng.randint(1, 15)
            x = self._rng.randint(-10, 10)
            b = self._rng.randint(-10, 10)
        else:
            a = self._rng.randint(2, 20)
            x = self._rng.randint(-20, 20)
            b = self._rng.randint(-20, 20)
        return _LinearCoeffs(a=a, b=b, x=x)

    def _format_linear(self, coeffs: _LinearCoeffs) -> tuple[str, int]:
        # a * x + b = c  → solve for x
        a, b, x = coeffs.a, coeffs.b, coeffs.x
        c = a * x + b
        # prompt like "Solve for x: 3x + 4 = 19"
        # handle cases for signs
        if b >= 0:
            left = f"{a}x + {b}"
        else:
            left = f"{a}x - {abs(b)}"
        prompt = f"Solve for x: {left} = {c}"
        return prompt, x

    def _two_step_linear(self, difficulty: Difficulty) -> tuple[str, int, dict]:
        coeffs = self._linear_coeffs(difficulty)
        prompt, x = self._format_linear(coeffs)
        meta = {
            "type": "linear",
            "a": coeffs.a,
            "b": coeffs.b,
            "x": x,
        }
        return prompt, x, meta

    def generate(self, difficulty: Difficulty) -> QuizItem:
        # For now we generate linear equations only, scaling ranges with difficulty.
        prompt, solution, meta = self._two_step_linear(difficulty)
        return QuizItem(
            prompt=prompt,
            answer=solution,
            difficulty=difficulty,
            quiz_type=QuizType.EQUATION,
            meta=meta,
        )



