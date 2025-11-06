# Decision Tree Generator - Pydantic AI Showcase

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Pydantic AI](https://img.shields.io/badge/Pydantic%20AI-latest-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This project is a **comprehensive demonstration** of [Pydantic AI's](https://ai.pydantic.dev) capabilities, showcasing how to build production-ready AI applications with type safety, data validation, and seamless FastAPI integration.

### What This Project Demonstrates

✅ **Type-Safe AI Agents** - Complex decision tree generation with guaranteed structure  
✅ **Data Validation** - Nested Pydantic models with runtime validation  
✅ **FastAPI Integration** - Full-stack application with AI-powered endpoints  
✅ **Authentication** - JWT-based auth with automatic token management  
✅ **Interactive UI** - Browser-based decision tree walkthrough  
✅ **Database Integration** - SQLModel with vector embeddings for similarity search  

---

## 🚀 What is Pydantic AI?

**Pydantic AI** is a production-ready Python framework for building AI-powered applications with **guaranteed structured outputs**. Unlike traditional LLM integrations where responses are unpredictable strings, Pydantic AI ensures every AI response matches your defined schema with full type safety and validation.

### Why This Matters

```python
# ❌ Traditional approach - unpredictable
response = openai.chat.completions.create(...)
data = json.loads(response.choices[0].message.content)
# What if the format is wrong? What if fields are missing?

# ✅ Pydantic AI - guaranteed structure
result = agent.run_sync("Generate decision tree")
tree: DecisionFlowChart = result.output  # Fully validated!
```

---

## 🏗️ Architecture

This application demonstrates a complete AI-powered decision-making system:

```
┌─────────────────┐
│  Web Interface  │  ← Interactive decision tree walkthrough
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI API   │  ← RESTful endpoints with Pydantic validation
└────────┬────────┘
         │
┌────────▼────────┐
│  Pydantic AI    │  ← Type-safe agent with complex output schema
│     Agent       │
└────────┬────────┘
         │
┌────────▼────────┐
│   OpenAI GPT    │  ← LLM generates structured decision trees
└─────────────────┘
```

---

## 🎨 Key Features Showcased

### 1. Complex Nested Data Models

The project uses sophisticated recursive Pydantic models to represent decision trees:

```python
class FlowChartNode(BaseModel):
    question: str
    if_yes: "FlowChartNode | Determination"  # Recursive structure
    if_no: "FlowChartNode | Determination"

class Determination(BaseModel):
    result: Literal["affirmative", "negative", "requires_deeper_consideration"]
    reasoning: str
    consideration_topics: list[ConsiderationTopic] = []
```

**Why This Is Powerful:**
- 🎯 AI generates arbitrarily deep decision trees
- ✅ Every node is validated at runtime
- 🔒 Type safety throughout the entire structure
- 📝 Automatic OpenAPI documentation

### 2. Pydantic AI Agent Implementation

```python
from pydantic_ai import Agent

decision_making_agent = Agent(
    'openai:gpt-4o',
    result_type=DecisionFlowChart,
    system_prompt='''You are an expert decision-making assistant...'''
)

# Generate validated decision trees
result = await decision_making_agent.run(
    f"Please generate a flowchart for: {prompt}"
)
tree: DecisionFlowChart = result.output  # Guaranteed valid!
```

**What This Demonstrates:**
- Type-safe agent configuration
- Structured output validation
- System prompt engineering
- Async/await support

### 3. FastAPI Integration

```python
@router.post("/", response_model=DecisionFullRead)
async def create_new_decision(
    db: DatabaseConnection, 
    user_id: UserId, 
    prompt: str
) -> DecisionFullRead:
    # AI generates the decision tree
    res = await decision_making_agent.run(
        f"Please generate a flowchart for: {prompt}"
    )
    
    # Validated output is database-ready
    decision = res.output.to_sql(prompt, user_id)
    db.add(decision)
    db.commit()
    
    return decision.to_model()
```

**Integration Benefits:**
- Same Pydantic models for API, database, and AI
- Automatic request/response validation
- Type-safe dependency injection
- Auto-generated OpenAPI docs

### 4. Authentication & User Management

- JWT-based authentication
- User-specific decision history
- Token stored in browser localStorage
- Automatic re-authentication on expiry

### 5. Vector Embeddings & Search

```python
class Decision(DecisionRequest, table=True):
    id: uuid.UUID
    embedding: list[float] = Field(sa_column=Column(Vector(1536)))
    flow_chart: str = Field(sa_type=TEXT)
```

**Demonstrates:**
- PostgreSQL pgvector integration
- Semantic search capabilities
- Efficient similarity matching

---

## 📁 Project Structure

```
decide_api/
├── api/
│   ├── agents/
│   │   └── decision_making_agent.py    # Pydantic AI agent
│   ├── routes/
│   │   ├── auth.py                      # JWT authentication
│   │   └── decisions.py                 # Decision CRUD endpoints
│   ├── models/
│   │   └── decision.py                  # Pydantic/SQLModel schemas
│   ├── auth/
│   │   ├── jwt_utils.py                 # Token management
│   │   └── dependencies.py              # FastAPI dependencies
│   └── templates/
│       └── index.html                   # Interactive UI
├── main.py                              # Application entry point
├── pyproject.toml                       # Dependencies
└── README.md                            # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd decide_api

# Install dependencies with Poetry
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Environment Variables

```bash
OPENAI_API_KEY=your-api-key-here
DATABASE_URL=sqlite:///./test_db.sqlite3  # or PostgreSQL for vector search
SECRET_KEY=your-secret-key-for-jwt
```

---

## 🚦 Running the Application

### Start the API Server

```bash
# Development mode with auto-reload
poetry run uvicorn api:app --reload

# Or with Python directly
python -m uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### Access the Interactive UI

Open your browser and navigate to:
- **Application:** `http://localhost:8000/` (if serving templates)
- **API Docs:** `http://localhost:8000/docs`
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`

---

## 🎯 API Endpoints

### Authentication

```bash
POST /auth/login
```
Generates a JWT token with a unique user ID.

**Response:**
```json
{
  "access_token": "eyJ0eXAi...",
  "token_type": "bearer",
  "user_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### Decision Management

```bash
GET /decision/
```
List all decisions for the authenticated user.

```bash
POST /decision/?prompt=Should+I+buy+a+car
```
Create a new decision tree using AI.

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "prompt": "Should I buy a car?",
  "flow_chart": {
    "question": "Do you have the financial means?",
    "if_yes": {
      "question": "Do you need it for daily commuting?",
      "if_yes": { "result": "affirmative", "name": "Buy the car", ... },
      "if_no": { ... }
    },
    "if_no": { ... }
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

```bash
GET /decision/{id}
```
Get a specific decision by ID.

---

## 💡 Usage Examples

### Python Client

```python
import httpx

# Login
response = httpx.post("http://localhost:8000/auth/login")
token = response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# Create a decision
response = httpx.post(
    "http://localhost:8000/decision/",
    params={"prompt": "Should I switch careers?"},
    headers=headers
)
decision = response.json()

# Navigate the decision tree
def walk_tree(node):
    if "question" in node:
        print(f"Q: {node['question']}")
        answer = input("Yes/No? ").lower()
        next_node = node['if_yes'] if answer == 'yes' else node['if_no']
        walk_tree(next_node)
    else:
        print(f"Decision: {node['result']}")
        print(f"Reasoning: {node['reasoning']}")

walk_tree(decision["flow_chart"])
```

### Command Line

```bash
# Run a decision from the command line
poetry run python main.py --decision "Should I invest in stocks?"
```

---

## 🎓 Learning Resources

This project demonstrates concepts covered in:

- **[Pydantic AI Documentation](https://ai.pydantic.dev)** - Official docs and guides
- **[FastAPI Documentation](https://fastapi.tiangolo.com)** - API framework
- **[Pydantic Documentation](https://docs.pydantic.dev)** - Data validation

### Presentation

Check out `pydantic-ai-presentation.md` for a comprehensive MARP presentation covering:
- Pydantic AI fundamentals
- Data validation patterns
- FastAPI integration
- Production best practices

---

## 🔍 Code Highlights

### Agent Definition

See `api/agents/decision_making_agent.py` for the complete Pydantic AI agent implementation with:
- Complex system prompting
- Structured output schemas
- Retry logic
- Error handling

### Data Models

See `api/models/decision.py` for:
- Recursive Pydantic models
- Discriminated unions
- Database integration
- Vector embedding support

### API Routes

See `api/routes/decisions.py` for:
- FastAPI endpoint implementation
- Dependency injection
- Authentication
- Type-safe responses

---

## 🧪 Testing

```bash
# Run tests
poetry run pytest

# With coverage
poetry run pytest --cov=api
```

The project includes examples of testing Pydantic AI agents with mocked models:

```python
from pydantic_ai.models.test import TestModel

test_model = TestModel()
agent = Agent(test_model, result_type=DecisionFlowChart)

# Inject predetermined responses
test_model.add_response(mock_decision_tree)

# Test without API calls
result = agent.run_sync("Test prompt")
assert isinstance(result.output, DecisionFlowChart)
```

---

## 🌟 Why This Project Matters

### For Developers

- **Learn by Example** - See Pydantic AI in a real application
- **Copy-Paste Ready** - Production-ready patterns you can use
- **Best Practices** - Type safety, validation, error handling
- **Full Stack** - Frontend, backend, database, AI all integrated

### For Teams

- **Maintainable** - Type safety catches errors early
- **Scalable** - Clean architecture, dependency injection
- **Testable** - Mock models for reliable testing
- **Documented** - Auto-generated API docs

### For Production

- **Reliable** - Validated AI outputs every time
- **Secure** - JWT authentication, input validation
- **Observable** - Structured logging, error tracking
- **Performant** - Async operations, efficient validation

---

## 🤝 Contributing

This is a demonstration project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - feel free to use this code in your own projects!

---

## 🙏 Acknowledgments

- **[Pydantic](https://github.com/pydantic/pydantic)** - The foundation for data validation
- **[Pydantic AI](https://github.com/pydantic/pydantic-ai)** - Type-safe AI framework
- **[FastAPI](https://github.com/tiangolo/fastapi)** - Modern web framework

---

## 📞 Contact & Support

For questions about:
- **Pydantic AI:** See [official documentation](https://ai.pydantic.dev)
- **This Project:** Open an issue on GitHub

---

## 🎉 Next Steps

1. **Explore the code** - Start with `api/agents/decision_making_agent.py`
2. **Run the application** - Try creating your own decision trees
3. **Modify the prompts** - Experiment with different system prompts
4. **Add features** - Extend with new endpoints or AI capabilities
5. **Share your learnings** - Build your own Pydantic AI projects!

