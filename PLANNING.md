# PLANNING.md - Sistema de Chatbot WhatsApp con Detección de Intenciones

## 🎯 **Visión General**
Sistema de chatbot inteligente para WhatsApp que detecta automáticamente las intenciones de los clientes y los enruta al departamento correspondiente usando LangChain/LangGraph con OpenAI.

## 🏗️ **Arquitectura del Sistema**

### **Stack Tecnológico Principal**
- **Backend**: Python + FastAPI 
- **AI Framework**: LangChain + LangGraph para workflows conversacionales
- **LLM**: OpenAI GPT-4 para clasificación de intenciones
- **WhatsApp**: whatsapp-web.js (Node.js subprocess desde Python)
- **Base de Datos**: Redis para sesiones + SQLite para persistencia
- **Deployment**: Render con auto-scaling
- **Testing**: Pytest con cobertura > 80%

### **Componentes Principales**

```
whatsapp-chatbot/
├── src/
│   ├── agents/                 # Agentes LangChain
│   │   ├── intent_classifier.py    # Clasificador principal
│   │   ├── conversation_agent.py   # Agente conversacional
│   │   └── routing_agent.py        # Agente de enrutamiento
│   ├── chains/                 # Cadenas LangChain
│   │   ├── intent_chain.py         # Detección de intenciones
│   │   ├── response_chain.py       # Generación de respuestas
│   │   └── context_chain.py        # Manejo de contexto
│   ├── services/               # Servicios de integración
│   │   ├── whatsapp_service.py     # WhatsApp Web.js wrapper
│   │   ├── llm_service.py          # OpenAI service
│   │   ├── routing_service.py      # Lógica de enrutamiento
│   │   └── context_service.py      # Gestión de contexto
│   ├── models/                 # Modelos Pydantic
│   │   ├── intents.py              # Modelos de intenciones
│   │   ├── messages.py             # Modelos de mensajes
│   │   └── routing.py              # Modelos de enrutamiento
└── main.py                     # FastAPI app principal
```

## 🔄 **Flujo de Procesamiento**

1. **Recepción**: WhatsApp Web.js recibe mensaje → FastAPI webhook
2. **Contexto**: Context Service recupera historial conversacional
3. **Clasificación**: Intent Classifier (LangChain + OpenAI) analiza mensaje
4. **Enrutamiento**: Routing Agent decide acción basada en intención
5. **Respuesta**: Response Chain genera respuesta contextual
6. **Envío**: WhatsApp Service envía respuesta al usuario

## 🎨 **Patrones de Diseño**

### **Patrones LangChain/LangGraph**
- **Agents**: Cada agente tiene responsabilidad específica (clasificación, routing, respuesta)
- **Chains**: Pipelines secuenciales para procesamiento
- **Tools**: Herramientas específicas para cada agente
- **Memory**: Gestión de memoria conversacional con compresión automática
- **Workflows**: LangGraph para flujos condicionales complejos

### **Patrones de Integración**
- **WhatsApp Wrapper**: Asyncio wrapper para subprocess Node.js
- **Rate Limiting**: Por usuario para evitar spam y bans de WhatsApp  
- **Circuit Breaker**: Para calls a OpenAI con fallback
- **Retry Logic**: Exponential backoff para APIs externas
- **Health Checks**: Endpoints para Render monitoring

### **Patrones de Datos**
- **Pydantic Models**: Validación estricta de inputs/outputs
- **Session Management**: Redis con TTL automático  
- **Context Compression**: Resumen automático de conversaciones largas
- **Structured Output**: OpenAI función calling para intenciones

## ⚙️ **Configuraciones Clave**

### **Variables de Entorno**
```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-1106-preview
OPENAI_MAX_TOKENS=1000

# WhatsApp Configuration
WHATSAPP_SESSION_PATH=./session
WHATSAPP_HEADLESS=true

# Redis Configuration  
REDIS_URL=redis://localhost:6379
REDIS_SESSION_TTL=7200

# Rate Limiting
RATE_LIMIT_PER_USER=10
RATE_LIMIT_WINDOW=60
```

### **Tipos de Intenciones**
```python
class IntentType(str, Enum):
    SALES = "ventas"           # Consultas de precios, productos
    SUPPORT = "soporte"        # Problemas técnicos, ayuda
    BILLING = "facturacion"    # Facturación, pagos
    GENERAL = "general"        # Información general
    COMPLAINT = "reclamo"      # Quejas, reclamos
    INFORMATION = "informacion" # Info de empresa
    UNKNOWN = "desconocido"    # No clasificable
```

### **Departamentos de Enrutamiento**
```yaml
ventas@empresa.com: [SALES]
soporte@empresa.com: [SUPPORT] 
facturacion@empresa.com: [BILLING]
info@empresa.com: [GENERAL, INFORMATION]
gerencia@empresa.com: [COMPLAINT] # Alta prioridad
```

## 🧪 **Estrategia de Testing**

### **Niveles de Testing**
- **Unit Tests**: Cada agente, service y model individualmente
- **Integration Tests**: Flujo completo de mensaje a respuesta
- **Contract Tests**: APIs externas (OpenAI, WhatsApp)
- **Load Tests**: Concurrencia y performance bajo carga
- **E2E Tests**: Casos reales de usuario con WhatsApp

### **Patrones de Testing**
```python
# Test de clasificación de intenciones
@pytest.mark.asyncio
async def test_sales_intent_classification():
    classifier = IntentClassifier()
    result = await classifier.classify_intent("¿Cuánto cuesta el producto X?")
    assert result.intent == IntentType.SALES
    assert result.confidence > 0.85

# Test de integración WhatsApp
@pytest.fixture
def mock_whatsapp_client():
    with patch('subprocess.Popen') as mock_popen:
        yield mock_popen
```

## 🚀 **Consideraciones de Deployment**

### **Render Configuration**
- **Service Type**: Web Service con auto-deploy desde Git
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python main.py`
- **Health Check**: `/health` endpoint obligatorio
- **Environment**: Variables seguras via Render dashboard
- **Scaling**: Auto-scaling basado en CPU/memoria

### **Monitoreo y Observabilidad**
- **Logs**: Structured logging con nivel configurable
- **Métricas**: Response time, error rate, intent accuracy  
- **Health Checks**: WhatsApp connection, Redis, OpenAI API
- **Alerts**: Desconexión WhatsApp, rate limit exceeded

## ⚠️ **Gotchas y Consideraciones Críticas**

### **WhatsApp Web.js**
- Requiere sesión persistente (QR code en primera ejecución)
- Puede desconectarse inesperadamente → reconnection logic
- Rate limiting agresivo → delays entre mensajes
- Session storage en filesystem → backup requerido

### **OpenAI API**
- Rate limits estrictos → exponential backoff
- Context window limitado → compresión de historial
- Costos por token → optimización de prompts
- Latencia variable → timeout handling

### **Render Constraints**
- Timeout 30 segundos → workers para procesos largos
- Filesystem no persistente → Redis para todo storage
- Sleep mode en free tier → keep-alive necesario
- Build time limitado → requirements.txt optimizado

### **LangChain Specifics**
- Memory leaks en conversaciones largas → cleanup automático
- Agent loops infinitos → max iterations limit
- Tool calling failures → fallback logic
- Chain complexity → debugging challenging

## 📋 **Checklist de Implementación**

### **Fase 1: Core Infrastructure** ✅ COMPLETADA
- [x] **Configuración Base** - requirements.txt, .env.example, estructura directorios 
- [x] **Modelos Pydantic** - intents.py, messages.py, routing.py completados
- [x] **WhatsApp Service** - whatsapp_service.py con session persistence y rate limiting
- [x] **Redis Context Service** - context_service.py con compression automática
- [x] **OpenAI LLM Service** - llm_service.py con retry logic y structured output
- [x] **Intent Classifier Agent** - intent_classifier.py con LangChain integration
- [x] **FastAPI Main App** - main.py con webhooks, health checks y background tasks
- [x] **Test Suite** - Tests unitarios y de integración con 70%+ coverage target

### **Fase 2: AI Components**
- [ ] Intent Classifier con LangChain
- [ ] Conversation Agent con memory
- [ ] Routing Agent con LangGraph workflow
- [ ] Response generation chains

### **Fase 3: Integration & Testing**
- [ ] End-to-end message flow
- [ ] Rate limiting y error handling  
- [ ] Unit tests para cada componente
- [ ] Integration tests completos

### **Fase 4: Production Ready**
- [ ] Render deployment configuration
- [ ] Monitoring y logging
- [ ] Load testing y optimization
- [ ] Documentation completa

## 🎯 **Criterios de Éxito**
- **Precisión**: 85%+ accuracy en clasificación de intenciones
- **Performance**: <3 segundos tiempo de respuesta
- **Concurrencia**: 100+ conversaciones simultáneas
- **Uptime**: 99%+ availability en producción
- **Cobertura**: 80%+ test coverage