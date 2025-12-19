from __future__ import annotations

import operator
import random
from dataclasses import dataclass
from typing import Callable

from .models import Difficulty, QuizItem, QuizType


@dataclass
class _OpConfig:
    symbol: str
    func: Callable[[int, int], int]


_OPS = {
    "+": _OpConfig("+", operator.add),
    "-": _OpConfig("-", operator.sub),
    "*": _OpConfig("*", operator.mul),
    "/": _OpConfig("/", operator.truediv),
}


class ArithmeticGenerator:
    """Generate arithmetic quizzes for different difficulty levels."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def _rand_range(self, difficulty: Difficulty) -> tuple[int, int]:
        if difficulty == Difficulty.EASY:
            return 0, 10
        if difficulty == Difficulty.MEDIUM:
            return 0, 50
        # HARD (wider range, includes negatives)
        return -100, 100

    def _all_ops(self) -> list[str]:
        """All difficulty levels can use all 4 operators."""
        return ["+", "-", "*", "/"]

    def _generate_division(
        self, difficulty: Difficulty
    ) -> tuple[int, int, int | float]:
        """
        Generate a / b where b != 0 and the result is "nice".

        For easy/medium we keep integer results; hard can have larger magnitudes.
        """
        if difficulty == Difficulty.EASY:
            divisor = self._rng.randint(1, 10)
            quotient = self._rng.randint(0, 10)
        elif difficulty == Difficulty.MEDIUM:
            divisor = self._rng.randint(1, 12)
            quotient = self._rng.randint(-20, 20)
        else:  # HARD
            divisor = self._rng.randint(1, 20)
            quotient = self._rng.randint(-50, 50)

        dividend = divisor * quotient
        return dividend, divisor, quotient

    def _generate_single_step(self, difficulty: Difficulty) -> QuizItem:
        """Single-operation expression, using all 4 operators."""
        op_symbol = self._rng.choice(self._all_ops())
        cfg = _OPS[op_symbol]

        if op_symbol == "/":
            a, b, result = self._generate_division(difficulty)
        else:
            lo, hi = self._rand_range(difficulty)
            a = self._rng.randint(lo, hi)
            b = self._rng.randint(lo, hi)
            result = cfg.func(a, b)

        prompt = f"{a} {op_symbol} {b} = ?"
        meta = {
            "operands": [a, b],
            "operator": op_symbol,
            "steps": 1,
        }
        return QuizItem(
            prompt=prompt,
            answer=result,
            difficulty=difficulty,
            quiz_type=QuizType.ARITHMETIC,
            meta=meta,
        )

    def _generate_multi_step_hard(self) -> QuizItem:
        """
        Hard-level multi-step expression.

        We keep division out of the multi-step chain to avoid ugly fractions,
        but hard questions can still use / via the single-step path.
        """
        lo, hi = self._rand_range(Difficulty.HARD)
        # Only +, -, * inside the parentheses expression
        op_symbol = self._rng.choice(["+", "-", "*"])
        cfg = _OPS[op_symbol]

        a = self._rng.randint(lo, hi)
        b = self._rng.randint(lo, hi)
        c = self._rng.randint(lo, hi)

        # random choice of pattern: (a op b) op c or a op (b op c)
        if self._rng.random() < 0.5:
            tmp = cfg.func(a, b)
            result = cfg.func(tmp, c)
            prompt = f"({a} {op_symbol} {b}) {op_symbol} {c} = ?"
        else:
            tmp = cfg.func(b, c)
            result = cfg.func(a, tmp)
            prompt = f"{a} {op_symbol} ({b} {op_symbol} {c}) = ?"

        meta = {
            "operands": [a, b, c],
            "operator": op_symbol,
            "steps": 2,
        }
        return QuizItem(
            prompt=prompt,
            answer=result,
            difficulty=Difficulty.HARD,
            quiz_type=QuizType.ARITHMETIC,
            meta=meta,
        )

    def generate(self, difficulty: Difficulty) -> QuizItem:
        if difficulty == Difficulty.HARD and self._rng.random() < 0.7:
            # Mostly multi-step for hard, with +, -, *.
            return self._generate_multi_step_hard()

        # For all difficulties (including some hard questions), use single-step
        # expressions that can include all 4 operators.
        return self._generate_single_step(difficulty)


