---
marp: true
theme: default
class: invert

paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #eee;

  }
  h1 {
    color: #e94560;
    font-size: 2.5em;
  }
  h2 {
    color: #0f3460;
    background: linear-gradient(90deg, #e94560, #0f3460);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  code {
    background-color: #16213e;
    color: #e94560;
  }
  pre {
    background-color: #16213e;
    border-radius: 8px;
  }
  strong {
    color: #e94560;
  }
  a {
    color: #00d4ff;
  }
---

# **Pydantic AI**
## The Power of Type-Safe AI Development

### Combining Runtime Validation with AI Agents

---

## **What is Pydantic AI?**

🚀 **Production-ready Python framework** for building AI-powered applications

Built on top of **Pydantic v2** - the most widely used data validation library in Python

### Key Features:
- ✅ Type-safe AI agent development
- ✅ Structured output validation
- ✅ Seamless FastAPI integration
- ✅ Multi-model support (OpenAI, Anthropic, Gemini, etc.)
- ✅ Streaming & function calling

---

## **Why Pydantic AI?**

### **The Problem:**
```python
# Traditional AI responses are unpredictable
response = openai.chat.completions.create(...)
data = json.loads(response.choices[0].message.content)
# ❌ What if the format is wrong?
# ❌ What if fields are missing?
# ❌ What if types don't match?
```

---

## **Why Pydantic AI?**

### **The Pydantic AI Solution:**
```python
# Guaranteed structured, validated output
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

result = agent.run_sync("Create a user profile")
user = result.output  # ✅ Fully validated User instance
```

---

## **Core Concept: Type-Safe Agents**

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: int
    
agent = Agent(
    'openai:gpt-4o',
    result_type=WeatherResponse,
    system_prompt='You are a weather assistant'
)

result = agent.run_sync('What is the weather in NYC?')
weather: WeatherResponse = result.output

# ✅ Type checking at development time
# ✅ Runtime validation guaranteed
# ✅ IDE autocomplete support
print(f"It's {weather.temperature}°F and {weather.condition}")
```

---

## **Data Validation: Pydantic's Superpower**

### **Automatic Type Coercion & Validation**

```python
from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime

class UserProfile(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    email: EmailStr
    age: int = Field(ge=0, le=150)
    created_at: datetime
    tags: list[str] = []
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v
```
---
### **Automatic Type Coercion & Validation**


```python
# Pydantic AI ensures LLM output matches this schema
agent = Agent('openai:gpt-4o', result_type=UserProfile)
result = agent.run_sync('Create a user profile for John Doe')
# ✅ All validation rules enforced!
```

---

## **Complex Nested Structures**

```python
from pydantic import BaseModel
from typing import Literal

class FlowChartNode(BaseModel):
    question: str
    if_yes: "FlowChartNode | Determination"
    if_no: "FlowChartNode | Determination"

class Determination(BaseModel):
    result: Literal["affirmative", "negative", "requires_consideration"]
    reasoning: str
    name: str | None = None
    
class DecisionTree(BaseModel):
    flow_chart: FlowChartNode

# AI generates complex, validated decision trees
decision_agent = Agent('openai:gpt-4o', result_type=DecisionTree)
result = decision_agent.run_sync('Should I buy a new car?')
tree: DecisionTree = result.output  # ✅ Fully structured & validated
```

---

## **FastAPI Integration: Perfect Match**

### **Pydantic Powers FastAPI**

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from pydantic_ai import Agent
from agents import decision_agent, DecisionRequest, DecisionResponse

app = FastAPI()


@app.post("/decision", response_model=DecisionResponse)
async def make_decision(request: DecisionRequest):
    result = await decision_agent.run(request.prompt)
    return result.output  # ✅ Automatic validation & serialization
```

---

## **FastAPI + Pydantic AI Benefits**

### **🔥 Type Safety End-to-End**

```python
# Request validation (Pydantic)
# ↓
# AI processing (Pydantic AI)
# ↓
# Response validation (Pydantic)
# ↓
# OpenAPI documentation (automatic)
```

### **Single Source of Truth**
- Same models for API, database, and AI validation
- Changes propagate automatically
- IDE support throughout the stack
- Auto-generated API documentation

---

## **Advanced Features: Function Calling**

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o', deps_type=str)

@agent.tool
async def get_weather(ctx: RunContext[str], city: str) -> dict:
    """Get current weather for a city."""
    # ctx.deps contains dependency injection data
    api_key = ctx.deps
    # Call weather API
    return {"temp": 72, "condition": "sunny"}

@agent.tool
async def send_email(ctx: RunContext[str], to: str, subject: str) -> str:
    """Send an email notification."""
    # AI can compose and send emails
    return f"Email sent to {to}"

# Agent intelligently decides which tools to use
result = await agent.run(
    "What's the weather in SF and email me the forecast?",
    deps="your-api-key"
)
```

---

## **Dependency Injection**

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDependencies:
    db_connection: DatabaseConnection
    user_id: str
    api_keys: dict[str, str]

agent = Agent('openai:gpt-4o', deps_type=AppDependencies)

@agent.tool
async def save_to_db(ctx: RunContext[AppDependencies], data: dict):
    # Access injected dependencies
    db = ctx.deps.db_connection
    user_id = ctx.deps.user_id
    await db.insert(user_id, data)
    return "Saved"

# Run with dependencies
result = await agent.run(
    "Save this user data...",
    deps=AppDependencies(db=db, user_id="123", api_keys=keys)
)
```

---

## **Streaming Responses**

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async with agent.run_stream('Write a story about AI') as result:
    async for message in result.stream_text():
        print(message, end='', flush=True)
    
    # Final validated result still available
    final_output = await result.get_data()

# Great for:
# - Real-time chat interfaces
# - Progress indicators
# - Long-form content generation
```

---

## **Multi-Model Support**

```python
from pydantic_ai import Agent

# OpenAI
agent_gpt = Agent('openai:gpt-4o')

# Anthropic Claude
agent_claude = Agent('anthropic:claude-3-5-sonnet-20241022')

# Google Gemini
agent_gemini = Agent('gemini:gemini-1.5-pro')

# Ollama (local)
agent_local = Agent('ollama:llama3.1')

# Groq (fast inference)
agent_groq = Agent('groq:llama-3.1-70b-versatile')

# Same interface, different models
# Easy to switch or A/B test
```

---

## **Error Handling & Retries**

```python
from pydantic_ai import Agent, ModelRetry

agent = Agent('openai:gpt-4o', retries=3)

@agent.tool
async def validate_data(ctx: RunContext, data: dict) -> str:
    if not is_valid(data):
        # Signal retry with corrective feedback
        raise ModelRetry('Data validation failed: missing email field')
    return "Valid"

# Agent automatically retries with the error message
# LLM learns from mistakes and tries again
result = await agent.run('Process this user data...')

# Built-in retry logic with exponential backoff
# Graceful degradation on persistent failures
```

---

## **Real-World Example: Decision Tree Generator**

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ConsiderationTopic(BaseModel):
    name: str
    description: str

class Determination(BaseModel):
    result: Literal["affirmative", "negative", "requires_consideration"]
    reasoning: str
    consideration_topics: list[ConsiderationTopic] = []

class FlowChartNode(BaseModel):
    question: str
    if_yes: "FlowChartNode | Determination"
    if_no: "FlowChartNode | Determination"

class DecisionFlowChart(BaseModel):
    flow_chart: FlowChartNode

decision_agent = Agent(
    'openai:gpt-4o',
    result_type=DecisionFlowChart,
    system_prompt='Generate decision trees with yes/no questions'
)
```

---

## **Real-World Example: Usage**

```python
# Generate a complex decision tree
result = await decision_agent.run(
    "Should I invest in cryptocurrency?"
)

tree: DecisionFlowChart = result.output

# Navigate the validated tree structure
def walk_tree(node: FlowChartNode | Determination):
    if isinstance(node, FlowChartNode):
        print(f"Q: {node.question}")
        user_answer = input("Yes/No? ").lower()
        next_node = node.if_yes if user_answer == 'yes' else node.if_no
        walk_tree(next_node)
    else:
        print(f"Decision: {node.result}")
        print(f"Reasoning: {node.reasoning}")

walk_tree(tree.flow_chart)
```

---

## **Testing Made Easy**

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Mock model for testing
test_model = TestModel()

agent = Agent(test_model, result_type=WeatherResponse)

# Inject predetermined responses
test_model.add_response(
    WeatherResponse(temperature=72.5, condition="sunny", humidity=45)
)

# Test without calling real APIs
result = agent.run_sync("What's the weather?")
assert result.output.temperature == 72.5
assert result.output.condition == "sunny"

# ✅ Fast, deterministic tests
# ✅ No API costs
# ✅ Full type safety maintained
```

---

## **Performance & Efficiency**

### **Pydantic v2 Benefits:**
- **🚀 5-50x faster** than Pydantic v1
- **Rust-powered** core validation
- **Lazy validation** options
- **Minimal overhead** for type checking

### **Pydantic AI Advantages:**
- **Efficient retries** with context preservation
- **Streaming** for real-time responses
- **Caching** support for repeated queries
- **Batch processing** capabilities

---

## **Best Practices**

### **1. Design Clear Schemas**
```python
class Product(BaseModel):
    """Product information with validation."""
    name: str = Field(description="Product name")
    price: float = Field(gt=0, description="Price in USD")
    tags: list[str] = Field(default=[], description="Product tags")
```

### **2. Use System Prompts Effectively**
```python
agent = Agent(
    'openai:gpt-4o',
    result_type=Product,
    system_prompt='''You are a product data extractor.
    Always include accurate pricing and relevant tags.
    Follow the schema strictly.'''
)
```

---

## **Best Practices (continued)**

### **3. Leverage Validators**
```python
class Analysis(BaseModel):
    sentiment: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    
    @validator('sentiment')
    def round_sentiment(cls, v):
        return round(v, 2)
```

### **4. Handle Errors Gracefully**
```python
try:
    result = await agent.run(user_input, retries=3)
except ValidationError as e:
    # Log validation failures
    logger.error(f"AI output validation failed: {e}")
except Exception as e:
    # Handle other errors
    logger.error(f"Agent execution failed: {e}")
```

---

## **Migration from Other Frameworks**

### **From LangChain:**
```python
# LangChain
from langchain.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=User)
# Complex setup, manual parsing

# Pydantic AI
agent = Agent('openai:gpt-4o', result_type=User)
# Simple, automatic validation
```

### **From Raw OpenAI:**
```python
# Raw OpenAI
response = openai.chat.completions.create(...)
data = json.loads(response.choices[0].message.content)
user = User(**data)  # Manual validation

# Pydantic AI
result = agent.run_sync(prompt)
user = result.output  # Automatic validation
```

---

## **Use Cases**

### **💼 Enterprise Applications**
- Data extraction from documents
- Customer support automation
- Report generation
- Content moderation

### **🔬 Research & Analysis**
- Sentiment analysis with structured output
- Literature review summarization
- Data classification
- Trend analysis

---

## **Use Cases - Continued**

### **🛠️ Developer Tools**
- Code generation with validation
- API response parsing
- Test data generation
- Documentation automation

---

## **Real Success Story: Decision API**

```python
# FastAPI endpoint powered by Pydantic AI
@router.post("/")
async def create_new_decision(
    db: DatabaseConnection, 
    user_id: UserId, 
    prompt: str
) -> DecisionFullRead:
    # Agent generates complex decision tree
    res = await decision_making_agent.run(
        f"Please generate a flowchart for: {prompt}"
    )
    
    # Output is validated and database-ready
    decision = res.output.to_sql(prompt, user_id)
    db.add(decision)
    db.commit()
    
    return decision.to_model()  # Type-safe response
```

### **Benefits Achieved:**
✅ Zero invalid AI responses | ✅ Full type safety | ✅ Clean architecture

---

## **Community & Ecosystem**

### **Growing Fast:**
- 🌟 Active GitHub repository
- 📚 Comprehensive documentation
- 💬 Helpful community
- 🔄 Regular updates

### **Integrations:**
- FastAPI (native support)
- SQLModel (database models)
- Logfire (observability)
- Major LLM providers

### **Resources:**
- 📖 [Documentation](https://ai.pydantic.dev)
- 💻 [GitHub](https://github.com/pydantic/pydantic-ai)
- 🎓 [Examples](https://ai.pydantic.dev/examples/)

---

## **Getting Started**

### **Installation:**
```bash
pip install pydantic-ai

# With specific model support
pip install 'pydantic-ai[openai]'
pip install 'pydantic-ai[anthropic]'
pip install 'pydantic-ai[gemini]'
```

### **First Agent:**
```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.data)
```

### **With Validation:**
```python
from pydantic import BaseModel

class Capital(BaseModel):
    city: str
    country: str
    population: int

agent = Agent('openai:gpt-4o', result_type=Capital)
result = agent.run_sync('What is the capital of France?')
print(f"{result.output.city}: {result.output.population:,} people")
```

---

## **Key Takeaways**

### **🎯 Why Choose Pydantic AI?**

1. **Type Safety** - Catch errors before production
2. **Data Validation** - Guaranteed correct structure
3. **FastAPI Integration** - Seamless full-stack development
4. **Developer Experience** - IDE support, clear errors
5. **Production Ready** - Built on battle-tested Pydantic
6. **Flexibility** - Multi-model, streaming, function calling

### **Perfect For:**
✅ Production applications | ✅ Team collaboration | ✅ Complex data structures

---

## **The Future is Type-Safe**

### **Pydantic AI represents the evolution of AI development:**

❌ **Old Way:**
- Unpredictable outputs
- Manual validation
- Runtime errors
- Poor developer experience

---


✅ **Pydantic AI Way:**
- Guaranteed structure
- Automatic validation
- Type safety
- Excellent DX

---

# **Thank You!**

## **Start Building Type-Safe AI Applications Today**

### **Resources:**
- 📖 **Documentation:** https://ai.pydantic.dev
- 💻 **GitHub:** https://github.com/pydantic/pydantic-ai
- 🐦 **Twitter:** @pydantic
- 💬 **Discord:** Join the community

### **Questions?**

---

## **Bonus: Advanced Patterns**

### **Agent Composition:**
```python
# Combine multiple agents
summarizer = Agent('openai:gpt-4o', result_type=Summary)
translator = Agent('openai:gpt-4o', result_type=Translation)

# Pipeline processing
summary = await summarizer.run(long_text)
translation = await translator.run(
    f"Translate to Spanish: {summary.output.text}"
)
```

---

### **Conditional Logic:**
```python
@agent.tool
async def get_data(ctx: RunContext, query: str) -> dict:
    if "weather" in query:
        return await weather_api.get(query)
    elif "news" in query:
        return await news_api.get(query)
    return {"error": "Unknown query type"}
```

---

## **Bonus: Monitoring & Observability**

### **Built-in Logfire Integration:**
```python
import logfire
from pydantic_ai import Agent

logfire.configure()

agent = Agent('openai:gpt-4o')

with logfire.span('ai-request'):
    result = await agent.run(prompt)
    logfire.info('request-complete', 
                 tokens=result.usage().total_tokens)

# Automatic tracking of:
# - Token usage
# - Latency
# - Retries
# - Validation errors
```

---

## **Bonus: Cost Optimization**

```python
from pydantic_ai import Agent

# Use cheaper models for simple tasks
classifier = Agent('openai:gpt-4o-mini', result_type=Category)

# Use powerful models only when needed
analyzer = Agent('openai:gpt-4o', result_type=DetailedAnalysis)

# Implement caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_classification(text: str) -> Category:
    result = classifier.run_sync(text)
    return result.output

# Monitor costs with usage tracking
result = await agent.run(prompt)
print(f"Tokens used: {result.usage().total_tokens}")
```
