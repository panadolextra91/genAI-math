## Math Quiz Generator (Python)

This package provides a small, deterministic "genAI-style" math quiz generator
for use inside a web application. It supports:

- **Two quiz types**: `arithmetic` and `equation`
- **Three difficulty levels**: `easy`, `medium`, `hard`

### Package layout

- `quiz_model/`
  - `models.py`: `Difficulty`, `QuizType`, `QuizItem` dataclass
  - `arithmetic_generator.py`: rule-based arithmetic question generator
  - `equation_generator.py`: rule-based equation (linear) generator
  - `learned_model.py`: neural network models (experimental, not used by default)
  - `generator.py`: `MathQuizGenerator` high-level API
- `service.py`: FastAPI HTTP service that exposes the generator to other backends
- `train_quiz_model.py`: training script for arithmetic learned model (experimental)
- `train_equation_model.py`: training script for equation learned model (experimental)
- `evaluate_models.py`: evaluation script to analyze learned model performance
- `requirements.txt`: Python dependencies for the service

### Core Python usage (direct import)

```python
from quiz_model import Difficulty, QuizType, MathQuizGenerator

gen = MathQuizGenerator(seed=42)  # seed for reproducible quizzes (optional)

# Single quiz item
item = gen.generate_one(quiz_type=QuizType.ARITHMETIC, difficulty=Difficulty.EASY)
print(item.prompt)  # e.g. "3 + 7 = ?"
print(item.answer)  # e.g. 10

# Batch of items for a quiz page
items = gen.generate_batch(quiz_type=QuizType.EQUATION, difficulty=Difficulty.MEDIUM, n=5)
```

### Rule-based vs Learned Models

This package includes **both rule-based and learned (neural network) generators**:

- **Rule-based generators** (`ArithmeticGenerator`, `EquationGenerator`): Deterministic, fully controllable logic that directly implements difficulty ranges, operators, and question structures.
- **Learned models** (`LearnedArithmeticGenerator`, `LearnedEquationGenerator`): Small neural networks trained to imitate the rule-based generators.

**We use rule-based models by default** because:

1. **Guaranteed correctness**: Rule-based generators compute answers exactly from operators/operands, ensuring all questions are mathematically valid.
2. **Explicit difficulty control**: You directly control number ranges, operators, and multi-step complexity per difficulty level.
3. **No training overhead**: No need to train, evaluate, or maintain model weights.
4. **Better for this use case**: Learned models struggle to accurately predict specific question parameters given only difficulty level, as the rule-based generator's outputs are inherently stochastic. Evaluation showed learned models perform only slightly better than random guessing.

The learned models are available for experimentation (see below), but for production quiz generation, the rule-based approach is recommended.

### Experimental: Learned Models (Optional)

The package includes training scripts and neural network implementations for learned models, but these are **not recommended for production use** due to poor accuracy when predicting question parameters from difficulty alone.

If you want to experiment with them:

```bash
# Install PyTorch (only needed for learned models)
pip install -r requirements.txt  # includes torch

# Train arithmetic model
python train_quiz_model.py

# Train equation model
python train_equation_model.py

# Evaluate model performance
python evaluate_models.py
```

To use learned models, you must explicitly enable them:

```python
gen = MathQuizGenerator(
    use_learned_arithmetic=True,  # requires quiznet.pt
    use_learned_equation=True,    # requires quiznet_equation.pt
)
```

By default, `use_learned_arithmetic=False` and `use_learned_equation=False`, so the rule-based generators are used.

### Python HTTP service (`service.py`)

The service exposes the generator via REST so a Node/Express backend can call it.

- `POST /generate`
  - **Request JSON**:

    ```json
    {
      "quiz_type": "arithmetic",  // or "equation"
      "difficulty": "easy",       // "easy" | "medium" | "hard"
      "n": 10
    }
    ```

  - **Response JSON**: array of objects

    ```json
    [
      {
        "prompt": "3 + 7 = ?",
        "answer": 10.0,
        "difficulty": "easy",
        "quizType": "arithmetic"
      }
    ]
    ```

- `GET /health` → `{ "status": "ok" }`

#### Run the Python service

From the project root:

```bash
# Install dependencies (torch is only needed if experimenting with learned models)
pip install -r requirements.txt
uvicorn service:app --host 0.0.0.0 --port 8000
```

**Note**: The `torch` dependency in `requirements.txt` is only needed if you want to experiment with learned models. For rule-based generation (the default), you can remove `torch` from `requirements.txt` if you want a lighter installation.

Test quickly:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"quiz_type":"arithmetic","difficulty":"easy","n":5}'
```

### Node + Express backend integration

Your Node backend calls the Python service and exposes a REST API to the client.

#### Express route

```js
// quizRoutes.js
const express = require('express');
const axios = require('axios');

const router = express.Router();

// Adjust if Python service runs elsewhere
const PYTHON_SERVICE_URL = 'http://localhost:8000/generate';

router.get('/quizzes', async (req, res) => {
  try {
    const {
      difficulty = 'easy',       // "easy" | "medium" | "hard" from UI
      quizType = 'arithmetic',   // "arithmetic" | "equation" from UI
      n = '10',
    } = req.query;

    const payload = {
      quiz_type: quizType,       // must match Python field name
      difficulty,
      n: Number(n),
    };

    const { data } = await axios.post(PYTHON_SERVICE_URL, payload);

    // Optionally hide answers from the client
    const hideAnswers = true;
    const result = hideAnswers
      ? data.map(({ answer, ...rest }) => rest)
      : data;

    res.json(result);
  } catch (err) {
    console.error('Error calling Python service:', err.message);
    res.status(502).json({ error: 'Failed to generate quizzes' });
  }
});

module.exports = router;
```

In your main Express app:

```js
// server.js
const express = require('express');
const quizRoutes = require('./quizRoutes');

const app = express();
app.use(express.json());

// Expose GET /api/quizzes
app.use('/api', quizRoutes);

app.listen(3000, () => {
  console.log('Node server running on http://localhost:3000');
});
```

The client can now call:

```text
GET http://localhost:3000/api/quizzes?difficulty=medium&quizType=equation&n=5
```

### Optional: Direct FastAPI usage (without Node)

Example with **FastAPI** (works similarly with Flask/Django):

```python
from fastapi import FastAPI, Query
from quiz_model import Difficulty, QuizType, MathQuizGenerator

app = FastAPI()
generator = MathQuizGenerator()


@app.get("/api/quizzes")
def get_quizzes(
    quiz_type: QuizType = Query(QuizType.ARITHMETIC),
    difficulty: Difficulty = Query(Difficulty.EASY),
    n: int = Query(10, ge=1, le=100),
):
    items = generator.generate_batch(quiz_type=quiz_type, difficulty=difficulty, n=n)
    # Convert dataclasses to plain dicts for JSON
    return [
        {
            "prompt": i.prompt,
            "answer": i.answer,
            "difficulty": i.difficulty.value,
            "quizType": i.quiz_type.value,
        }
        for i in items
    ]
```

You can hide the `answer` field from the initial response if you only want it on
answer-checking endpoints.


