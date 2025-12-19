from __future__ import annotations

"""
Evaluate the learned arithmetic and equation models.

This script:
- regenerates synthetic eval datasets using the same logic as training
- runs the learned models on them
- reports per-head accuracies and a confusion matrix for operators.

Usage:
    python evaluate_models.py
"""

from typing import Tuple

import torch

from quiz_model.learned_model import (
    IDX_TO_OP,
    EquationNet,
    QuizNet,
    WEIGHTS_PATH,
    WEIGHTS_PATH_EQ,
)
from train_equation_model import synthesize_equation_dataset
from train_quiz_model import synthesize_dataset


def _confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t, p] += 1
    return cm


def evaluate_arithmetic(
    n_samples: int = 2000,
) -> None:
    if not WEIGHTS_PATH.is_file():
        print("[eval_arithmetic] no weights found at", WEIGHTS_PATH)
        return

    print(f"[eval_arithmetic] generating {n_samples} eval samples...")
    X, y_op, y_a, y_b = synthesize_dataset(n_samples=n_samples, seed=999)

    model = QuizNet()
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits_op, logits_a, logits_b = model(X)
        pred_op = logits_op.argmax(dim=1)
        pred_a = logits_a.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)

    acc_op = (pred_op == y_op).float().mean().item()
    acc_a = (pred_a == y_a).float().mean().item()
    acc_b = (pred_b == y_b).float().mean().item()

    print(f"[eval_arithmetic] operator accuracy: {acc_op:.3f}")
    print(f"[eval_arithmetic] operand a accuracy: {acc_a:.3f}")
    print(f"[eval_arithmetic] operand b accuracy: {acc_b:.3f}")

    cm = _confusion_matrix(y_op, pred_op, num_classes=4)
    print("[eval_arithmetic] operator confusion matrix (rows=true, cols=pred):")
    # pretty-print with operator labels
    labels = [IDX_TO_OP[i] for i in range(4)]
    header = "      " + " ".join(f"{l:>5}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{int(v):5d}" for v in row)
        print(f" {labels[i]:>3}   {row_str}")


def evaluate_equation(
    n_samples: int = 2000,
) -> None:
    if not WEIGHTS_PATH_EQ.is_file():
        print("[eval_equation] no weights found at", WEIGHTS_PATH_EQ)
        return

    print(f"[eval_equation] generating {n_samples} eval samples...")
    X, y_a, y_x, y_b = synthesize_equation_dataset(n_samples=n_samples, seed=999)

    model = EquationNet()
    state = torch.load(WEIGHTS_PATH_EQ, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits_a, logits_x, logits_b = model(X)
        pred_a = logits_a.argmax(dim=1)
        pred_x = logits_x.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)

    acc_a = (pred_a == y_a).float().mean().item()
    acc_x = (pred_x == y_x).float().mean().item()
    acc_b = (pred_b == y_b).float().mean().item()

    # joint accuracy (all three correct simultaneously)
    acc_joint = ((pred_a == y_a) & (pred_x == y_x) & (pred_b == y_b)).float().mean().item()

    print(f"[eval_equation] a accuracy:      {acc_a:.3f}")
    print(f"[eval_equation] x accuracy:      {acc_x:.3f}")
    print(f"[eval_equation] b accuracy:      {acc_b:.3f}")
    print(f"[eval_equation] joint accuracy:  {acc_joint:.3f}")


if __name__ == "__main__":
    evaluate_arithmetic()
    print()
    evaluate_equation()


