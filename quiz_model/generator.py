from __future__ import annotations

import hashlib
from collections import Counter
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


class QualityController:
    """
    Quality control mechanism to prevent mode collapse and ensure diversity.
    
    Tracks:
    - Generated questions (deduplication)
    - Operator distribution (for arithmetic)
    - Problem type distribution
    - Difficulty distribution
    """

    def __init__(self, max_retries: int = 50) -> None:
        self.max_retries = max_retries
        self._generated_hashes: set[str] = set()
        self._operator_counts: Counter[str] = Counter()
        self._type_counts: Counter[QuizType] = Counter()
        self._difficulty_counts: Counter[Difficulty] = Counter()
        self._recent_items: List[QuizItem] = []  # For mode collapse detection
        self._max_recent = 100  # Keep last 100 items for similarity check

    def _hash_item(self, item: QuizItem) -> str:
        """Generate a hash for deduplication based on prompt."""
        return hashlib.md5(item.prompt.encode()).hexdigest()

    def _is_duplicate(self, item: QuizItem) -> bool:
        """Check if item is an exact duplicate."""
        item_hash = self._hash_item(item)
        return item_hash in self._generated_hashes

    def _is_too_similar(self, item: QuizItem) -> bool:
        """
        Check if item is too similar to recently generated items (mode collapse detection).
        
        For arithmetic: check if same operator and similar operands
        For equations: check if same coefficient pattern
        """
        if len(self._recent_items) < 5:
            return False

        # Check last 10 items for similarity
        recent = self._recent_items[-10:]
        
        if item.quiz_type == QuizType.ARITHMETIC:
            # Check if same operator and very similar operands
            item_op = item.meta.get("operator") if item.meta else None
            item_operands = tuple(sorted(item.meta.get("operands", []))) if item.meta else None
            
            for recent_item in recent:
                if recent_item.quiz_type != QuizType.ARITHMETIC:
                    continue
                recent_op = recent_item.meta.get("operator") if recent_item.meta else None
                recent_operands = tuple(sorted(recent_item.meta.get("operands", []))) if recent_item.meta else None
                
                # Same operator and operands within 2 units
                if item_op == recent_op and item_operands and recent_operands:
                    if len(item_operands) == len(recent_operands):
                        if all(abs(a - b) <= 2 for a, b in zip(item_operands, recent_operands)):
                            return True
        
        elif item.quiz_type == QuizType.EQUATION:
            # Check if same coefficient pattern
            item_a = item.meta.get("a") if item.meta else None
            item_b = item.meta.get("b") if item.meta else None
            
            for recent_item in recent:
                if recent_item.quiz_type != QuizType.EQUATION:
                    continue
                recent_a = recent_item.meta.get("a") if recent_item.meta else None
                recent_b = recent_item.meta.get("b") if recent_item.meta else None
                
                # Same coefficients
                if item_a == recent_a and item_b == recent_b:
                    return True

        return False

    def _get_operator_distribution(self, target_n: int) -> dict[str, float]:
        """Get current operator distribution for arithmetic problems."""
        total = sum(self._operator_counts.values())
        if total == 0:
            return {"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25}
        
        dist = {op: count / total for op, count in self._operator_counts.items()}
        # Ensure all operators are represented
        for op in ["+", "-", "*", "/"]:
            if op not in dist:
                dist[op] = 0.0
        
        return dist

    def _should_prefer_operator(self, target_n: int) -> str | None:
        """
        Suggest an operator to balance distribution.
        Returns operator that is underrepresented, or None if balanced.
        """
        if target_n < 4:
            return None  # Too small to balance
        
        dist = self._get_operator_distribution(target_n)
        target_ratio = 1.0 / 4.0  # 25% each
        
        # Find most underrepresented operator
        min_ratio = min(dist.values())
        if min_ratio < target_ratio * 0.7:  # If any operator is < 17.5%
            for op in ["+", "-", "*", "/"]:
                if dist.get(op, 0) == min_ratio:
                    return op
        
        return None

    def accept_item(self, item: QuizItem, enforce_diversity: bool = True) -> bool:
        """
        Check if item should be accepted based on quality criteria.
        
        Returns True if item is acceptable, False if it should be rejected.
        """
        # Check for exact duplicates
        if self._is_duplicate(item):
            return False

        # Check for mode collapse (too similar to recent items)
        if enforce_diversity and self._is_too_similar(item):
            return False

        return True

    def register_item(self, item: QuizItem) -> None:
        """Register an accepted item in the tracking system."""
        item_hash = self._hash_item(item)
        self._generated_hashes.add(item_hash)
        
        # Track distributions
        self._type_counts[item.quiz_type] += 1
        self._difficulty_counts[item.difficulty] += 1
        
        # Track operators for arithmetic
        if item.quiz_type == QuizType.ARITHMETIC and item.meta:
            operator = item.meta.get("operator")
            if operator:
                self._operator_counts[operator] += 1
        
        # Track recent items for similarity detection
        self._recent_items.append(item)
        if len(self._recent_items) > self._max_recent:
            self._recent_items.pop(0)

    def reset(self) -> None:
        """Reset all tracking (useful for testing or new sessions)."""
        self._generated_hashes.clear()
        self._operator_counts.clear()
        self._type_counts.clear()
        self._difficulty_counts.clear()
        self._recent_items.clear()

    def get_stats(self) -> dict:
        """Get current quality control statistics."""
        return {
            "total_generated": len(self._generated_hashes),
            "operator_distribution": dict(self._operator_counts),
            "type_distribution": {k.value: v for k, v in self._type_counts.items()},
            "difficulty_distribution": {k.value: v for k, v in self._difficulty_counts.items()},
        }


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
        enable_quality_control: bool = True,
        quality_max_retries: int = 50,
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

        # Quality control system
        self._quality_control = QualityController(max_retries=quality_max_retries) if enable_quality_control else None

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
        enforce_diversity: bool = True,
    ) -> List[QuizItem]:
        """
        Generate a batch of quizzes with quality control.
        
        Args:
            quiz_type: Type of quiz to generate
            difficulty: Difficulty level
            n: Number of quizzes to generate
            enforce_diversity: Whether to enforce diversity checks (deduplication, mode collapse prevention)
        
        Returns:
            List of unique, diverse QuizItem objects
        """
        if n <= 0:
            return []
        
        # If quality control is disabled, use simple generation
        if self._quality_control is None:
            return [self.generate_one(quiz_type=quiz_type, difficulty=difficulty) for _ in range(n)]
        
        items: List[QuizItem] = []
        retry_count = 0
        max_total_attempts = n * self._quality_control.max_retries
        
        # For arithmetic, try to balance operator distribution
        prefer_operator = None
        if quiz_type == QuizType.ARITHMETIC and enforce_diversity:
            prefer_operator = self._quality_control._should_prefer_operator(n)
        
        attempts = 0
        while len(items) < n and attempts < max_total_attempts:
            attempts += 1
            
            # Generate candidate item
            item = self.generate_one(quiz_type=quiz_type, difficulty=difficulty)
            
            # For arithmetic with distribution control, prefer underrepresented operators
            if prefer_operator and item.quiz_type == QuizType.ARITHMETIC:
                if item.meta and item.meta.get("operator") != prefer_operator:
                    # If we're trying to balance and this isn't the preferred operator,
                    # still check it, but we might retry if we have many items already
                    if len(items) < n * 0.8:  # Allow some flexibility in first 80%
                        retry_count += 1
                        if retry_count < 10:  # Give it a few tries
                            continue
            
            # Quality check
            if self._quality_control.accept_item(item, enforce_diversity=enforce_diversity):
                self._quality_control.register_item(item)
                items.append(item)
                retry_count = 0  # Reset retry count on success
            else:
                retry_count += 1
                if retry_count >= self._quality_control.max_retries:
                    # If we've retried too many times, relax diversity checks slightly
                    # This prevents infinite loops while still maintaining quality
                    if enforce_diversity:
                        # Try one more time with relaxed checks
                        if self._quality_control.accept_item(item, enforce_diversity=False):
                            self._quality_control.register_item(item)
                            items.append(item)
                    retry_count = 0
        
        # If we couldn't generate enough items, return what we have
        # (this can happen if the problem space is too constrained)
        return items

    def generate_mixed(
        self,
        spec: Iterable[tuple[QuizType, Difficulty, int]],
        enforce_diversity: bool = True,
    ) -> List[QuizItem]:
        """
        Generate a mixed list of quizzes based on a specification with quality control.

        Example:
            spec = [
                (QuizType.ARITHMETIC, Difficulty.EASY, 5),
                (QuizType.EQUATION, Difficulty.MEDIUM, 3),
            ]
        
        Args:
            spec: Iterable of (quiz_type, difficulty, count) tuples
            enforce_diversity: Whether to enforce diversity checks across the entire batch
        
        Returns:
            List of unique, diverse QuizItem objects covering the specification
        """
        quizzes: List[QuizItem] = []
        for quiz_type, difficulty, n in spec:
            batch = self.generate_batch(quiz_type, difficulty, n, enforce_diversity=enforce_diversity)
            quizzes.extend(batch)
        return quizzes

    def reset_quality_control(self) -> None:
        """Reset the quality control tracking (useful for new sessions)."""
        if self._quality_control:
            self._quality_control.reset()

    def get_quality_stats(self) -> dict | None:
        """Get quality control statistics."""
        if self._quality_control:
            return self._quality_control.get_stats()
        return None



