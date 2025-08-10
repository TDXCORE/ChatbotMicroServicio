name: "WhatsApp Chatbot PRP Template - Context-Rich with LangChain Patterns"
description: |

## Purpose
Template optimizado para implementar sistemas de chatbot WhatsApp con detección de intenciones usando LangChain/LangGraph y OpenAI. Incluye contexto específico para WhatsApp Web.js integration, session management, y patterns de conversational AI.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, y WhatsApp-specific gotchas
2. **LangChain Patterns**: Use proper agent, chain, y memory management patterns
3. **WhatsApp Constraints**: Account for session persistence, rate limiting, y connection stability
4. **Validation Loops**: Provide executable tests para WhatsApp, OpenAI, y Redis integration
5. **Production Ready**: Include deployment, monitoring, y error recovery considerations
6. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
[What needs to be built - specific para chatbot functionality, intent classification, y WhatsApp integration]

## Why
- [Business value de automated customer routing y 24/7 availability]
- [Integration benefits con existing customer service workflow]
- [Cost reduction y efficiency gains específicos]

## What
[User-visible behavior in WhatsApp y technical requirements]

### Success Criteria
- [ ] WhatsApp connection stable con session persistence
- [ ] Intent classification accuracy 85%+
- [ ] Response time < 3 seconds
- [ ] Handles 100+ concurrent conversations
- [ ] Redis session management working
- [ ] OpenAI API integration con rate limiting
- [ ] Render deployment successful
- [ ] Test coverage > 80%

## All Needed Context

### Documentation & References (MUST READ for WhatsApp chatbot)
```yaml
# CRITICAL - LangChain/LangGraph Patterns
- url: https://python.langchain.com/docs/get_started/introduction
  why: Core concepts for agent architecture
  critical: Agent composition patterns y tool integration

- url: https://langchain-ai.github.io/langgraph/tutorials/introduction/
  why: State management y conversation workflows
  critical: ConversationBufferMemory y context compression

- url: https://python.langchain.com/docs/modules/agents/
  why: Agent patterns for intent classification y routing
  critical: Tool calling y structured output patterns

# CRITICAL - WhatsApp Web.js Integration  
- url: https://github.com/pedroslopez/whatsapp-web.js
  why: Core library para WhatsApp integration
  critical: Session persistence y authentication flows

- url: https://docs.wwebjs.dev/
  why: API reference completa
  critical: Message handling y client events

- url: https://wwebjs.dev/guide/authentication.html
  why: Session management (CRÍTICO para production)
  critical: QR authentication y session storage

# CRITICAL - OpenAI Integration
- url: https://platform.openai.com/docs/guides/function-calling
  why: Structured output para intent classification
  critical: Function schema definition para Pydantic models

- url: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
  why: Implementation patterns for structured classification
  critical: Error handling y rate limit management

# PRODUCTION - Render Deployment
- url: https://docs.render.com/deploy-fastapi
  why: FastAPI deployment patterns
  critical: Environment configuration y health checks

- file: examples/langchain_agents/intent_classifier_agent.py
  why: Pattern para intent classification con LangChain
  critical: Structured output y confidence scoring

- file: examples/whatsapp_integration/whatsapp_wrapper.py
  why: Asyncio wrapper pattern para Node.js subprocess
  critical: Session persistence y reconnection logic
```

### Current Codebase Structure
```bash
whatsapp-chatbot/
├── src/
│   ├── agents/                 # LangChain agents
│   ├── chains/                 # LangChain chains  
│   ├── services/               # Integration services
│   ├── models/                 # Pydantic models
│   └── utils/                  # Utilities
├── tests/                      # Test suite
├── examples/                   # Implementation patterns
├── main.py                     # FastAPI app
└── requirements.txt            # Dependencies
```

### Desired Codebase Structure (add files needed)
```bash
whatsapp-chatbot/
├── src/
│   ├── agents/
│   │   ├── intent_classifier.py      # NEW: Intent classification agent
│   │   ├── conversation_agent.py     # NEW: Conversation management
│   │   └── routing_agent.py          # NEW: Routing decisions
│   ├── chains/
│   │   ├── intent_chain.py           # NEW: Intent detection pipeline
│   │   ├── response_chain.py         # NEW: Response generation
│   │   └── context_chain.py          # NEW: Context management
│   ├── services/
│   │   ├── whatsapp_service.py       # NEW: WhatsApp Web.js wrapper
│   │   ├── llm_service.py            # NEW: OpenAI service
│   │   └── context_service.py        # NEW: Redis session management
├── requirements.txt                   # UPDATE: Add langchain, openai, redis
├── .env.example                       # NEW: Environment variables
└── render.yaml                        # NEW: Render deployment config
```

### Known Gotchas & WhatsApp/LangChain Quirks
```python
# CRÍTICO: WhatsApp Web.js requires persistent session storage
# Without this, bot disconnects every restart y requires QR re-auth
WHATSAPP_SESSION_PATH = "./session"  # MUST be persistent filesystem

# CRÍTICO: WhatsApp rate limiting es muy agresivo
# Sending messages faster than 1 per second puede result in ban
await asyncio.sleep(1.5)  # MINIMUM delay between messages

# CRÍTICO: LangChain memory management
# ConversationBufferMemory can cause memory leaks en long conversations
# MUST implement conversation compression después de 20+ messages

# CRÍTICO: OpenAI rate limits
# Function calling has separate rate limits (500 requests/day en free tier)
# MUST implement exponential backoff y circuit breaker

# CRÍTICO: Render timeout constraints  
# HTTP requests timeout after 30 seconds
# Background tasks required para longer processing

# CRÍTICO: Context window limitations
# GPT-4 has 8k token limit para function calling
# MUST summarize conversation history when approaching limit
```

## Implementation Blueprint

### Data Models and Structure (Pydantic models FIRST)

```python
# src/models/intents.py - Core intent classification models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class IntentType(str, Enum):
    SALES = "ventas"
    SUPPORT = "soporte" 
    BILLING = "facturacion"
    GENERAL = "general"
    COMPLAINT = "reclamo"
    INFORMATION = "informacion"
    UNKNOWN = "desconocido"

class MessageIntent(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)
    routing_department: str
    priority: int = Field(ge=1, le=5)

class ConversationContext(BaseModel):
    user_id: str
    conversation_id: str  
    message_history: List[str] = Field(max_items=50)
    current_intent: Optional[MessageIntent] = None
    session_start: datetime = Field(default_factory=datetime.now)
    is_active: bool = True

class WhatsAppMessage(BaseModel):
    id: str
    from_number: str
    body: str
    timestamp: datetime
    message_type: str = "text"
```

### Task List (ORDEN ESPECÍFICO para chatbot development)

```yaml
Task 1: Setup Core Dependencies y Environment
CREATE requirements.txt:
  - langchain>=0.1.0
  - langgraph>=0.0.40  
  - openai>=1.0.0
  - fastapi>=0.100.0
  - pydantic>=2.0.0
  - redis>=4.5.0
  - python-dotenv>=1.0.0
  - pytest>=7.0.0

CREATE .env.example:
  - PATTERN: All critical env vars documented
  - Include OpenAI, WhatsApp, Redis configuration
  - Production y development configurations

Task 2: Implement WhatsApp Service Integration
CREATE src/services/whatsapp_service.py:
  - PATTERN: Asyncio wrapper para whatsapp-web.js subprocess
  - CRITICAL: Session persistence con filesystem storage
  - Queue management para incoming/outgoing messages
  - Reconnection logic con exponential backoff
  - Rate limiting enforcement (1.5s delay minimum)

Task 3: Create LLM Service con OpenAI Integration  
CREATE src/services/llm_service.py:
  - PATTERN: OpenAI client con structured output
  - Function calling setup para intent classification
  - Rate limiting con exponential backoff
  - Context window management y conversation compression
  - Error handling para API failures

Task 4: Build Intent Classification Agent
CREATE src/agents/intent_classifier.py:
  - PATTERN: LangChain Agent con OpenAI function calling
  - Few-shot prompting examples para cada intent type
  - Confidence scoring y fallback to rule-based classification
  - Entity extraction usando structured output
  - Integration con conversation context

Task 5: Implement Context Management Service
CREATE src/services/context_service.py:
  - PATTERN: Redis-based session storage
  - Conversation history management con TTL
  - Context compression para long conversations
  - Session cleanup y garbage collection
  - Backup to persistent storage

Task 6: Create Routing Agent con LangGraph
CREATE src/agents/routing_agent.py:
  - PATTERN: LangGraph workflow para routing decisions
  - State machine para conversation flows
  - Department routing rules based on intent
  - Priority handling para urgent cases
  - Handoff protocols to human agents

Task 7: Build Response Generation Chain
CREATE src/chains/response_chain.py:
  - PATTERN: LangChain chain para contextual responses
  - Template-based responses por intent type
  - Personalization based on conversation history
  - Multi-language support if needed
  - Integration con department-specific knowledge

Task 8: Implement FastAPI Main Application
CREATE main.py:
  - PATTERN: FastAPI con async endpoints
  - Webhook endpoint para WhatsApp messages
  - Background task processing
  - Health check endpoint para Render
  - Rate limiting middleware
  - Error handling y logging

Task 9: Create Comprehensive Test Suite
CREATE tests/:
  - test_whatsapp_service.py: Mock WhatsApp Web.js integration
  - test_intent_classifier.py: Intent classification accuracy
  - test_routing_agent.py: Routing decision validation  
  - test_integration.py: End-to-end message flow
  - Mock external services (OpenAI, Redis, WhatsApp)
  - Async testing patterns con pytest-asyncio

Task 10: Setup Render Deployment Configuration
CREATE render.yaml:
  - PATTERN: Web service con auto-deploy
  - Environment variables configuration
  - Health check endpoints
  - Redis addon integration
  - Build y start commands
```

### Integration Points (CRITICAL para WhatsApp chatbot)
```yaml
WHATSAPP WEB.JS INTEGRATION:
  - subprocess: "node whatsapp-bot.js"
  - session_path: "./session" (MUST be persistent)
  - event_handling: message, qr, ready, disconnected
  - rate_limiting: 1.5 seconds between messages minimum

OPENAI FUNCTION CALLING:
  - model: "gpt-4-1106-preview" 
  - functions: intent_classification, entity_extraction
  - max_tokens: 1000 (balance cost vs accuracy)
  - temperature: 0.1 (low for consistent classification)

REDIS SESSION MANAGEMENT:
  - key_pattern: "whatsapp:session:{user_id}"
  - ttl: 7200 seconds (2 hours)
  - data: conversation_context, message_history
  - compression: JSON + gzip para large contexts

FASTAPI BACKGROUND TASKS:
  - webhook_processing: async message handling
  - session_cleanup: periodic Redis cleanup
  - health_monitoring: WhatsApp connection status
```

## Validation Loop (WhatsApp-specific testing)

### Level 1: Syntax & Style
```bash
# Run FIRST - fix errors before proceeding
black . --line-length 88
ruff check . --fix
mypy . --ignore-missing-imports

# Expected: No errors. If errors exist, READ y fix immediately
```

### Level 2: Unit Tests (Mock external services)
```python
# test_intent_classifier.py
@pytest.mark.asyncio
async def test_sales_intent_classification():
    """Test accurate sales intent detection"""
    classifier = IntentClassifier()
    context = ConversationContext(user_id="test", conversation_id="test")
    
    result = await classifier.classify_intent(
        "¿Cuánto cuesta el producto premium?", 
        context
    )
    
    assert result.intent == IntentType.SALES
    assert result.confidence > 0.85
    assert result.routing_department == "ventas"

# test_whatsapp_service.py  
@pytest.mark.asyncio
async def test_whatsapp_message_sending():
    """Test WhatsApp message sending con rate limiting"""
    service = WhatsAppService()
    
    with patch.object(service, '_send_via_subprocess') as mock_send:
        start_time = time.time()
        await service.send_message("1234567890", "Test message")
        end_time = time.time()
        
        # Verify rate limiting delay applied
        assert end_time - start_time >= 1.5
        mock_send.assert_called_once()

# test_context_service.py
@pytest.mark.asyncio  
async def test_redis_session_management():
    """Test Redis session storage y retrieval"""
    service = ContextService()
    context = ConversationContext(user_id="test_user", conversation_id="test_conv")
    
    await service.save_context(context)
    retrieved = await service.get_context("test_user")
    
    assert retrieved.user_id == context.user_id
    assert retrieved.conversation_id == context.conversation_id
```

```bash
# Run iteratively until passing:
pytest tests/ -v --cov=src --cov-report=term-missing

# Target: 80%+ coverage, all tests passing
# If failing: Debug specific test, fix code, re-run
```

### Level 3: Integration Test (End-to-end workflow)
```bash
# Start all services
python main.py &
redis-server &

# Test WhatsApp webhook endpoint
curl -X POST http://localhost:8000/webhook/whatsapp \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_msg_001",
    "from_number": "1234567890", 
    "body": "Hola, tengo una consulta sobre precios de productos",
    "timestamp": "2024-01-01T12:00:00Z"
  }'

# Expected response:
# - Status 200 
# - Background task initiated
# - Intent classified as SALES
# - Context saved to Redis
# - Response generated y queued para WhatsApp

# Verify logs show:
# - Message received y parsed
# - Intent classified con confidence > 0.8
# - Routing decision made
# - Response sent via WhatsApp (mocked)
```

## Final Validation Checklist (WhatsApp chatbot specific)
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check .`
- [ ] No type errors: `mypy .`
- [ ] WhatsApp Web.js connection stable (no disconnections)
- [ ] OpenAI intent classification accuracy > 85%
- [ ] Redis session management working (save/retrieve)
- [ ] FastAPI health endpoint returns 200
- [ ] Rate limiting prevents WhatsApp ban (1.5s+ delays)
- [ ] Context compression prevents token limit exceeded
- [ ] Background tasks handle long-running processing
- [ ] Environment variables documented in .env.example
- [ ] Render deployment configuration valid

---

## Anti-Patterns to Avoid (WhatsApp chatbot specific)
- ❌ Don't ignore WhatsApp session persistence - leads to constant disconnections
- ❌ Don't send messages without rate limiting - WhatsApp will ban the number  
- ❌ Don't store large contexts in memory - use Redis con compression
- ❌ Don't ignore OpenAI rate limits - implement exponential backoff
- ❌ Don't use sync functions in FastAPI - breaks async processing
- ❌ Don't hardcode intent routing rules - make configurable
- ❌ Don't skip health checks - Render requires them for monitoring
- ❌ Don't ignore conversation history limits - implement compression
- ❌ Don't deploy without testing WhatsApp integration locally first

## Confidence Score: 9.0/10

High confidence due to:
- Well-established patterns para WhatsApp Web.js integration
- LangChain/OpenAI combination is battle-tested
- Clear validation loops catch common WhatsApp gotchas
- Render deployment is straightforward para FastAPI apps

Minor uncertainties:
- WhatsApp Web.js stability in production environments
- Specific rate limits for OpenAI function calling in high-volume scenarios