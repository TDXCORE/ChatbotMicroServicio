# TASK.md - Sistema de Chatbot WhatsApp con Detección de Intenciones

## 📋 **Tareas Principales del Proyecto**

### **✅ Completadas**
- [2025-08-10] Crear estructura base del proyecto con templates Context Engineering
- [2025-08-10] Definir arquitectura del sistema en PLANNING.md
- [2025-08-10] Establecer tracking de tareas en TASK.md

### **🔄 En Progreso**
- [ ] Ninguna actualmente

### **📝 Tareas Pendientes**

#### **Fase 1: Configuración Base** 
- [ ] Configurar requirements.txt con todas las dependencias necesarias
  - langchain>=0.1.0, langgraph>=0.0.40, openai>=1.0.0
  - fastapi>=0.100.0, pydantic>=2.0.0, redis>=4.5.0
  - python-dotenv>=1.0.0, pytest>=7.0.0
- [ ] Crear .env.example con todas las variables de entorno necesarias
- [ ] Establecer estructura de directorios según PLANNING.md

#### **Fase 2: Servicios Core**
- [ ] Implementar WhatsApp Service (whatsapp_service.py)
  - Wrapper asyncio para whatsapp-web.js subprocess
  - Manejo de sesiones persistentes con QR code
  - Queue para mensajes entrantes y salientes
- [ ] Crear LLM Service (llm_service.py) 
  - Integración con OpenAI API
  - Rate limiting y retry logic
  - Error handling robusto
- [ ] Implementar Context Service (context_service.py)
  - Gestión de sesiones Redis
  - Context compression para conversaciones largas
  - Session cleanup automático

#### **Fase 3: Agentes LangChain**
- [ ] Desarrollar Intent Classifier (intent_classifier.py)
  - LangChain Agent con OpenAI
  - Few-shot prompting para clasificación
  - Structured output con Pydantic
- [ ] Crear Conversation Agent (conversation_agent.py)
  - Manejo de memoria conversacional
  - Integration con context service
  - Response generation contextual
- [ ] Implementar Routing Agent (routing_agent.py)
  - LangGraph workflow para decisiones
  - Reglas de negocio para departamentos
  - Priority handling para casos urgentes

#### **Fase 4: Chains y Workflows**
- [ ] Construir Intent Chain (intent_chain.py)
  - Pipeline clasificación + extracción entidades
  - Confidence scoring y fallback logic
- [ ] Desarrollar Response Chain (response_chain.py)  
  - Generación respuestas contextuales
  - Template-based responses
  - Personalization por departamento
- [ ] Crear Context Chain (context_chain.py)
  - Manejo memoria conversacional
  - History compression automática
  - Context switching entre intenciones

#### **Fase 5: Modelos y Validación**
- [ ] Definir modelos Pydantic (models/)
  - intents.py: IntentType, MessageIntent, ConversationContext
  - messages.py: WhatsAppMessage, MessageHistory
  - routing.py: RoutingDecision, DepartmentConfig
- [ ] Implementar validators (utils/validators.py)
  - Validación de números WhatsApp
  - Sanitización de mensajes
  - Input validation para APIs

#### **Fase 6: API y Endpoints**
- [ ] Crear FastAPI main application (main.py)
  - Webhook endpoint para WhatsApp (/webhook/whatsapp)
  - Health check endpoint (/health)
  - Background task processing
- [ ] Implementar rate limiting y middleware
  - Rate limiting por usuario
  - CORS configuration
  - Request/response logging
- [ ] Añadir error handling global
  - Exception handlers
  - Structured error responses
  - Error logging y monitoring

#### **Fase 7: Testing Comprehensivo**
- [ ] Tests unitarios para servicios (tests/)
  - test_whatsapp_service.py
  - test_llm_service.py  
  - test_context_service.py
- [ ] Tests de agentes LangChain
  - test_intent_classifier.py
  - test_conversation_agent.py
  - test_routing_agent.py
- [ ] Tests de integración completa
  - test_integration.py (flujo end-to-end)
  - test_chains.py (testing de chains)
  - Mock testing para APIs externas

#### **Fase 8: Configuración y Deployment**
- [ ] Configurar Render deployment
  - render.yaml con configuración completa
  - Variables de entorno en Render
  - Health checks y scaling rules
- [ ] Documentación completa
  - README.md con setup instructions
  - API documentation
  - Troubleshooting guide
- [ ] Scripts de utilidad
  - setup.py para configuración inicial
  - test_deployment.py para validar deploy
  - migrate_data.py si se necesita migración

#### **Fase 9: Optimización y Monitoreo**
- [ ] Implementar logging estructurado
  - Logger configuration (utils/logger.py)
  - Log levels configurables
  - Performance monitoring
- [ ] Optimización de performance
  - Connection pooling para Redis
  - Async optimization
  - Memory usage optimization
- [ ] Setup monitoring y alerts
  - Health metrics collection
  - Error rate monitoring
  - WhatsApp connection monitoring

### **🔍 Descubierto Durante el Trabajo**
*(Esta sección se actualiza durante el desarrollo)*

### **⚠️ Riesgos y Bloqueadores Identificados**
- **WhatsApp Web.js stability**: Requiere session management robusto
- **OpenAI rate limits**: Necesario implement exponential backoff
- **Render timeout constraints**: 30s limit para requests
- **Context window limits**: GPT-4 tiene límites de tokens
- **Redis dependency**: Critical para session storage en Render

### **📊 Métricas de Progreso**
- **Fases Completadas**: 0/9
- **Tareas Completadas**: 3/~40 
- **Cobertura de Tests**: 0% (objetivo: 80%+)
- **Estado de Deploy**: Not Ready

### **🎯 Próximos Hitos**
1. **Sprint 1** (Semana 1): Configuración base + servicios core
2. **Sprint 2** (Semana 2): Agentes LangChain + chains
3. **Sprint 3** (Semana 3): API + testing
4. **Sprint 4** (Semana 4): Deployment + optimización

---
*Última actualización: 2025-08-10*
*Responsable: Claude Code Assistant*