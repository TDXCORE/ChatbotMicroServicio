## FEATURE:

Desarrollar un sistema de chatbot inteligente que detecte automáticamente las intenciones de los clientes en WhatsApp y los dirija al área correspondiente de manera eficiente. El sistema utilizará LangChain/LangGraph con OpenAI para procesamiento de lenguaje natural y se desplegará como microservicio en Render.

**Especificaciones Técnicas:**
- **Backend**: Python + FastAPI para API REST
- **AI Framework**: LangChain + LangGraph para workflows conversacionales
- **LLM**: OpenAI GPT-4 para clasificación de intenciones con 85%+ precisión
- **WhatsApp Integration**: whatsapp-web.js (Node.js subprocess desde Python)
- **Base de Datos**: Redis para sesiones + SQLite para persistencia
- **Deployment**: Render con auto-scaling
- **Performance**: Tiempo de respuesta < 3 segundos, 100+ conversaciones simultáneas

**Funcionalidades Core:**
- Recepción de mensajes de WhatsApp vía webhook
- Análisis de intención del cliente usando LangChain + OpenAI
- Clasificación automática en 7 tipos: SALES, SUPPORT, BILLING, GENERAL, COMPLAINT, INFORMATION, UNKNOWN
- Enrutamiento inteligente al departamento correspondiente
- Mantenimiento de contexto conversacional para seguimiento
- Respuestas automáticas contextuales para consultas frecuentes
- Rate limiting y manejo de errores robusto

## EXAMPLES:

En la carpeta `examples/` encontrarás patrones específicos para implementar el chatbot WhatsApp:

- `examples/langchain_agents/` - Patrones de agentes LangChain para clasificación de intenciones
  - `intent_classifier_agent.py` - Ejemplo de agent que clasifica intenciones con OpenAI
  - `conversation_agent.py` - Agent que maneja memoria conversacional
  - `routing_agent.py` - Agent que toma decisiones de enrutamiento
- `examples/whatsapp_integration/` - Integración con WhatsApp Web.js
  - `whatsapp_wrapper.py` - Wrapper asyncio para subprocess Node.js
  - `message_queue.py` - Queue management para mensajes entrantes/salientes
- `examples/chains/` - Ejemplos de LangChain chains
  - `intent_detection_chain.py` - Pipeline para detección de intenciones
  - `response_generation_chain.py` - Chain para generar respuestas contextuales
- `examples/fastapi_patterns/` - Patrones de API
  - `webhook_handlers.py` - Manejo de webhooks de WhatsApp
  - `background_tasks.py` - Processing async de mensajes
- `examples/testing_patterns/` - Patrones de testing
  - `test_intent_classification.py` - Tests para clasificación
  - `test_whatsapp_integration.py` - Tests de integración WhatsApp
  - `mock_whatsapp.py` - Mocks para testing sin WhatsApp real

## DOCUMENTATION:

### **Documentación Crítica - DEBE Leerse Durante Implementación:**

**LangChain/LangGraph (Core AI Framework):**
- https://python.langchain.com/docs/get_started/introduction - Conceptos fundamentales
- https://langchain-ai.github.io/langgraph/tutorials/introduction/ - Workflows y gestión de estado
- https://python.langchain.com/docs/modules/agents/ - Patrones de agentes conversacionales
- https://python.langchain.com/docs/modules/chains/ - Chain composition patterns

**OpenAI Integration:**
- https://platform.openai.com/docs/guides/text-generation - Text generation y classification
- https://platform.openai.com/docs/guides/function-calling - Structured output para intenciones
- https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models - Function calling patterns

**WhatsApp Web.js:**
- https://github.com/pedroslopez/whatsapp-web.js - Main library documentation
- https://docs.wwebjs.dev/ - API reference completa
- https://wwebjs.dev/guide/authentication.html - Session management crítico

**FastAPI & Async Python:**
- https://fastapi.tiangolo.com/async/ - Async patterns para webhooks
- https://fastapi.tiangolo.com/advanced/background-tasks/ - Background processing
- https://fastapi.tiangolo.com/advanced/middleware/ - Rate limiting middleware

**Render Deployment:**
- https://docs.render.com/deploy-fastapi - FastAPI deployment específico
- https://docs.render.com/environment-variables - Environment configuration
- https://docs.render.com/health-checks - Health check requirements

**Redis & Session Management:**
- https://redis.io/docs/connect/clients/python/ - Redis Python client
- https://redis.io/docs/manual/keyspace-notifications/ - Session expiration handling

## OTHER CONSIDERATIONS:

### **Gotchas Críticos que AI Assistants Suelen Pasar por Alto:**

**WhatsApp Web.js Específicos:**
- Session persistence es CRÍTICO - sin esto el bot se desconecta constantemente
- QR code aparece en primera ejecución y requiere scan manual
- WhatsApp puede banear bots si envían mensajes muy rápido (< 1 segundo entre mensajes)
- puppeteer puede fallar en Render - requiere configuración específica de browser
- Session storage debe ser en filesystem persistente, no en memoria

**OpenAI API Constraints:**
- Rate limits estrictos (3500 requests/minute en tier básico) - DEBE implementar exponential backoff
- Context window de GPT-4 es limitado - conversaciones largas requieren compresión/resumen
- Function calling tiene formato específico que DEBE seguirse exactamente
- Costos se acumulan rápidamente con conversaciones largas - optimizar prompts

**Render Platform Limitaciones:**
- Timeout de 30 segundos para HTTP requests - procesos largos DEBEN usar background tasks
- Filesystem no es persistente entre deploys - todo storage va a Redis
- Free tier tiene sleep mode - bot se desconecta sin traffic
- Environment variables son case-sensitive

**LangChain Memory Management:**
- Memory leaks comunes en conversaciones largas - implementar cleanup automático
- Agent loops infinitos pueden ocurrir - DEBE configurar max_iterations
- Tool calling failures pueden romper el flujo - implementar fallback logic
- Chain complexity hace debugging muy difícil - empezar simple

**Production Readiness:**
- Health checks DEBEN incluir conexión WhatsApp, Redis y OpenAI API
- Logging estructurado es crítico para debugging en producción  
- Rate limiting DEBE ser por usuario, no global
- Error recovery automático para desconexiones de WhatsApp
- Backup strategy para sessions críticas de WhatsApp

### **Variables de Entorno Críticas:**
```bash
# CRÍTICO - Sin estas el sistema no funciona
OPENAI_API_KEY=sk-...
WHATSAPP_SESSION_PATH=./session  # Persistencia obligatoria
REDIS_URL=redis://...

# IMPORTANTE - Para production
ENVIRONMENT=production
RATE_LIMIT_PER_USER=10
WHATSAPP_HEADLESS=true
```

### **Dependencies que DEBEN Estar en requirements.txt:**
```
langchain>=0.1.0
langgraph>=0.0.40
openai>=1.0.0
fastapi>=0.100.0
pydantic>=2.0.0
redis>=4.5.0
python-dotenv>=1.0.0
pytest>=7.0.0
```
