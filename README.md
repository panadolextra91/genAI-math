## Math Quiz Generator (Python)

This package provides a small, deterministic "genAI-style" math quiz generator
for use inside a web application. It supports:

- **Two quiz types**: `arithmetic` and `equation`
- **Three difficulty levels**: `easy`, `medium`, `hard`

### Package layout

- `quiz_model/`
  - `models.py`: `Difficulty`, `QuizType`, `QuizItem` dataclass
  - `arithmetic_generator.py`: arithmetic question generator
  - `equation_generator.py`: simple equation (linear) generator
  - `generator.py`: `MathQuizGenerator` high-level API
- `service.py`: FastAPI HTTP service that exposes the generator to other backends
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

By default, `MathQuizGenerator` will:

- use the **learned arithmetic model** if trained weights exist at `quiz_model/quiznet.pt`
- otherwise fall back to the **rule-based arithmetic generator**.

Equations are currently always rule-based.

### Training your own arithmetic model (from scratch)

You can train the small neural network yourself, using synthetic data from the
rule-based generator:

```bash
pip install -r requirements.txt  # includes torch
python train_quiz_model.py
```

This will:

- generate arithmetic questions using the existing logic
- train `quiz_model.learned_model.QuizNet`
- save weights to `quiz_model/quiznet.pt`

After that, the Python service (and thus your Node backend) will automatically
use the **learned model** for arithmetic questions.

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
pip install -r requirements.txt
uvicorn service:app --host 0.0.0.0 --port 8000
```

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


