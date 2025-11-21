---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #FFFFFF;
    color: #333333;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    padding: 60px 80px;
  }
  h1 {
    color: #FF8C42 !important;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 0.3em;
    letter-spacing: -0.02em;
  }
  h1 strong {
    color: #FF8C42 !important;
  }
  h2 {
    color: #E70865 !important;
    font-size: 1.8em;
    font-weight: 600;
    margin-top: 0.5em;
    margin-bottom: 0.8em;
  }
  h2 strong {
    color: #FF8C42 !important;
  }
  h3 {
    color: #333333;
    font-size: 1.3em;
    font-weight: 400;
    margin-top: 0.3em;
    line-height: 1.5;
  }
  code {
    background-color: #F5F7F9;
    color: #E70865;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 0.9em;
  }
  pre {
    background-color: #F5F7F9;
    border-radius: 8px;
    border-left: 4px solid #FF8C42;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  pre code {
    background-color: transparent;
    color: #333333;
    padding: 0;
    font-size: 0.85em;
  }
  strong {
    color: #FF8C42;
    font-weight: 600;
  }
  a {
    color: #E70865;
    text-decoration: none;
  }
  a:hover {
    text-decoration: underline;
  }
  ul, ol {
    margin: 15px 0;
  }
  li {
    margin: 10px 0;
    line-height: 1.6;
  }
  footer {
    color: #999999;
    font-size: 0.8em;
  }
  /* Accent elements */
  section::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 8px;
    background: linear-gradient(90deg, #FF8C42 0%, #E70865 100%);
  }
---

# **Pydantic AI**
## Reliable Structured Data from LLMs

### A Lightning Talk

---

## **The Problem**

```python
# Traditional approach: unpredictable outputs
response = openai.chat.completions.create(...)
data = json.loads(response.choices[0].message.content)

# What could go wrong?
# ❌ Invalid JSON format
# ❌ Missing required fields
# ❌ Wrong data types
# ❌ No validation
```

---

## **The Pydantic AI Solution**

**Guaranteed structured, validated output every time**

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent('openai:gpt-4o', output_type=CityLocation)
result = agent.run_sync('Where were the 2012 Olympics?')
print(result.output)
#> city='London' country='United Kingdom'
```

✅ Type-safe | ✅ Validated | ✅ Predictable

---

## **How It Works**

### **1. Define Your Schema**
Use Pydantic models to specify exactly what you want

### **2. Create an Agent**
Pass your schema as the `output_type`

### **3. Run & Get Validated Data**
The agent ensures the LLM returns data matching your schema

---

## **Automatic Validation**

```python
from pydantic import BaseModel, Field, EmailStr

class UserProfile(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: int = Field(ge=0, le=150)
    email: EmailStr

agent = Agent('openai:gpt-4o', output_type=UserProfile)
result = agent.run_sync('Create profile for John, age 25')

# Pydantic automatically validates:
# ✅ Username length (3-20 chars)
# ✅ Age range (0-150)
# ✅ Email format
```

---

## **Complex Nested Structures**

```python
class FlowChartNode(BaseModel):
    question: str
    if_yes: "FlowChartNode | Determination"
    if_no: "FlowChartNode | Determination"

class Determination(BaseModel):
    result: Literal["affirmative", "negative", "requires_consideration"]
    reasoning: str

class DecisionTree(BaseModel):
    flow_chart: FlowChartNode

agent = Agent('openai:gpt-4o', output_type=DecisionTree)
result = agent.run_sync('Should I invest in stocks?')
# Returns fully validated decision tree
```

---

## **Multiple Output Types**

```python
class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str

agent = Agent(
    'openai:gpt-4o',
    output_type=[Box, str],  # Union type!
    system_prompt="Extract dimensions or ask for clarification"
)

result = agent.run_sync('The box is 10x20x30')
#> "Please provide the units (e.g., cm, in, m)"

result = agent.run_sync('The box is 10x20x30 cm')
#> Box(width=10, height=20, depth=30, units='cm')
```

---

## **Real Example: Customer Support**

```python
class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice to customer')
    block_card: bool = Field(description='Whether to block card')
    risk: int = Field(description='Risk level', ge=0, le=10)

support_agent = Agent(
    'openai:gpt-4o',
    output_type=SupportOutput,
    system_prompt='You are a bank support agent'
)

result = support_agent.run_sync('I just lost my card!')
#> SupportOutput(
#>   support_advice="We're blocking your card now",
#>   block_card=True,
#>   risk=8
#> )
```

---

## **Real Example: Data Extraction**

```python
from datetime import date
from typing_extensions import TypedDict

class UserProfile(TypedDict):
    name: str
    dob: date
    bio: str

agent = Agent('openai:gpt-4o', output_type=UserProfile)

text = 'My name is Ben, born January 28th 1990, I like dogs'
result = agent.run_sync(text)

print(result.output)
#> {'name': 'Ben',
#>  'dob': datetime.date(1990, 1, 28),
#>  'bio': 'I like dogs'}
```

---

## **Why This Matters**

### **🎯 Reliability**
No more parsing errors or malformed data

### **🛡️ Type Safety**
Catch errors at development time, not production

### **⚡ Productivity**
Focus on business logic, not data validation

### **📊 Consistency**
Same structure every time, guaranteed

---

## **Getting Started**

```bash
pip install pydantic-ai
```

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class MyData(BaseModel):
    field1: str
    field2: int

agent = Agent('openai:gpt-4o', output_type=MyData)
result = agent.run_sync('Your prompt here')
print(result.output)  # Validated MyData instance
```

**That's it!**

---

# **Thank You!**

## **Key Takeaway:**
Pydantic AI makes LLM outputs **reliable, predictable, and type-safe**

### **Resources:**
- 📖 **Docs:** https://ai.pydantic.dev
- 💻 **GitHub:** github.com/pydantic/pydantic-ai
- 🐦 **Twitter:** @pydantic

### **Questions?**
