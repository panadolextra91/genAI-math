from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from quiz_model import Difficulty, QuizType, MathQuizGenerator


app = FastAPI(title="Math Quiz Generator Service")
generator = MathQuizGenerator()


class GenerateRequest(BaseModel):
    quiz_type: QuizType
    difficulty: Difficulty
    n: int = 10


class QuizItemResponse(BaseModel):
    prompt: str
    answer: float
    difficulty: str
    quizType: str


@app.post("/generate", response_model=List[QuizItemResponse])
def generate_quizzes(body: GenerateRequest) -> List[QuizItemResponse]:
    items = generator.generate_batch(
        quiz_type=body.quiz_type,
        difficulty=body.difficulty,
        n=body.n,
    )
    return [
        QuizItemResponse(
            prompt=i.prompt,
            answer=float(i.answer),
            difficulty=i.difficulty.value,
            quizType=i.quiz_type.value,
        )
        for i in items
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}



