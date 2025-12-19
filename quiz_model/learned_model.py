from __future__ import annotations

"""
Simple learned generators for arithmetic and equation quizzes.

Arithmetic:
- operator in {+, -, *, /}
- operands a, b within a fixed integer range

Equations:
- coefficients for linear equation a*x + b = c with integer x

Both are trained on synthetic data produced by the existing rule-based generators.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from .models import Difficulty, QuizItem, QuizType


DEVICE = torch.device("cpu")
WEIGHTS_PATH = Path(__file__).with_name("quiznet.pt")
WEIGHTS_PATH_EQ = Path(__file__).with_name("quiznet_equation.pt")


DIFF_TO_IDX: Dict[Difficulty, int] = {
    Difficulty.EASY: 0,
    Difficulty.MEDIUM: 1,
    Difficulty.HARD: 2,
}

IDX_TO_OP = {0: "+", 1: "-", 2: "*", 3: "/"}
OP_TO_IDX = {v: k for k, v in IDX_TO_OP.items()}

MAX_ABS_VAL = 100  # operands a, b are mapped into [-100, 100]
VAL_RANGE = 2 * MAX_ABS_VAL + 1  # 201 bins


class QuizNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = 3 + 4  # difficulty one-hot (3) + noise (4)
        hidden = 128
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_op = nn.Linear(hidden, 4)  # + - * /
        self.head_a = nn.Linear(hidden, VAL_RANGE)
        self.head_b = nn.Linear(hidden, VAL_RANGE)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits_op = self.head_op(h)
        logits_a = self.head_a(h)
        logits_b = self.head_b(h)
        return logits_op, logits_a, logits_b


def has_learned_weights() -> bool:
    return WEIGHTS_PATH.is_file()


def has_learned_equation_weights() -> bool:
    return WEIGHTS_PATH_EQ.is_file()


@dataclass
class LearnedArithmeticGenerator:
    """
    Wrapper around QuizNet to generate arithmetic QuizItem objects.
    """

    model: QuizNet

    @classmethod
    def from_weights(cls, path: Path | None = None) -> "LearnedArithmeticGenerator":
        weights = path or WEIGHTS_PATH
        model = QuizNet().to(DEVICE)
        state = torch.load(weights, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model)

    def _sample_raw(self, difficulty: Difficulty) -> Tuple[str, int, int]:
        diff_idx = DIFF_TO_IDX[difficulty]
        # difficulty one-hot
        diff_one_hot = torch.zeros(3, device=DEVICE)
        diff_one_hot[diff_idx] = 1.0
        # random noise
        noise = torch.randn(4, device=DEVICE)
        x = torch.cat([diff_one_hot, noise], dim=0).unsqueeze(0)  # (1, 7)

        with torch.no_grad():
            logits_op, logits_a, logits_b = self.model(x)
            op_idx = torch.distributions.Categorical(logits=logits_op).sample()[0].item()
            a_idx = torch.distributions.Categorical(logits=logits_a).sample()[0].item()
            b_idx = torch.distributions.Categorical(logits=logits_b).sample()[0].item()

        op_symbol = IDX_TO_OP[op_idx]
        a = a_idx - MAX_ABS_VAL
        b = b_idx - MAX_ABS_VAL
        return op_symbol, a, b

    def generate(self, difficulty: Difficulty) -> QuizItem:
        op_symbol, a, b = self._sample_raw(difficulty)

        # Compute exact numeric answer.
        if op_symbol == "+":
            result = a + b
        elif op_symbol == "-":
            result = a - b
        elif op_symbol == "*":
            result = a * b
        else:  # "/"
            # avoid division by zero; if model outputs b == 0, force 1
            if b == 0:
                b = 1
            result = a / b

        prompt = f"{a} {op_symbol} {b} = ?"
        meta = {
            "operands": [a, b],
            "operator": op_symbol,
            "steps": 1,
            "backend": "learned",
        }
        return QuizItem(
            prompt=prompt,
            answer=result,
            difficulty=difficulty,
            quiz_type=QuizType.ARITHMETIC,
            meta=meta,
        )


class EquationNet(nn.Module):
    """
    Simple model to generate (a, b, x) for linear equations a*x + b = c.

    We model:
    - a in [1, 20]
    - x in [-20, 20]
    - b in [-20, 20]
    """

    def __init__(self) -> None:
        super().__init__()
        input_dim = 3 + 4  # difficulty one-hot (3) + noise (4)
        hidden = 128
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # heads: a_idx in [0, 19] -> a = idx + 1
        # x_idx, b_idx in [0, 40] -> value = idx - 20
        self.head_a = nn.Linear(hidden, 20)
        self.head_x = nn.Linear(hidden, 41)
        self.head_b = nn.Linear(hidden, 41)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits_a = self.head_a(h)
        logits_x = self.head_x(h)
        logits_b = self.head_b(h)
        return logits_a, logits_x, logits_b


@dataclass
class LearnedEquationGenerator:
    """
    Wrapper around EquationNet to generate equation QuizItem objects.
    """

    model: EquationNet

    @classmethod
    def from_weights(cls, path: Path | None = None) -> "LearnedEquationGenerator":
        weights = path or WEIGHTS_PATH_EQ
        model = EquationNet().to(DEVICE)
        state = torch.load(weights, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model)

    def _sample_raw(self, difficulty: Difficulty) -> Tuple[int, int, int]:
        diff_idx = DIFF_TO_IDX[difficulty]
        diff_one_hot = torch.zeros(3, device=DEVICE)
        diff_one_hot[diff_idx] = 1.0
        noise = torch.randn(4, device=DEVICE)
        x_in = torch.cat([diff_one_hot, noise], dim=0).unsqueeze(0)

        with torch.no_grad():
            logits_a, logits_x, logits_b = self.model(x_in)
            a_idx = torch.distributions.Categorical(logits=logits_a).sample()[0].item()
            x_idx = torch.distributions.Categorical(logits=logits_x).sample()[0].item()
            b_idx = torch.distributions.Categorical(logits=logits_b).sample()[0].item()

        a = a_idx + 1  # [1, 20]
        x = x_idx - 20  # [-20, 20]
        b = b_idx - 20  # [-20, 20]
        return a, b, x

    def generate(self, difficulty: Difficulty) -> QuizItem:
        a, b, x = self._sample_raw(difficulty)
        c = a * x + b

        # Format like: Solve for x: 3x + 4 = 19
        if b >= 0:
            left = f"{a}x + {b}"
        else:
            left = f"{a}x - {abs(b)}"
        prompt = f"Solve for x: {left} = {c}"
        meta = {
            "type": "linear",
            "a": a,
            "b": b,
            "x": x,
            "backend": "learned",
        }
        return QuizItem(
            prompt=prompt,
            answer=x,
            difficulty=difficulty,
            quiz_type=QuizType.EQUATION,
            meta=meta,
        )



