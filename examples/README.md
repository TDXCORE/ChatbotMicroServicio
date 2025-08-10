# 📚 Ejemplos de Patrones para Chatbot WhatsApp

Esta carpeta contiene ejemplos específicos de patrones de implementación para el sistema de chatbot WhatsApp con LangChain/LangGraph y OpenAI.

## 📁 Estructura de Ejemplos

### `/langchain_agents/`
Patrones para agentes LangChain especializados en chatbots WhatsApp.

**📋 `intent_classifier_agent.py`**
- ✅ **Qué muestra**: Clasificación de intenciones con OpenAI function calling
- ✅ **Patrones clave**:
  - Structured output con Pydantic models
  - Few-shot prompting para mejor precisión
  - Confidence scoring y fallback logic
  - Context-aware classification
- ✅ **Cuándo usar**: Para implementar intent_classifier.py en src/agents/
- ✅ **Integración**: Se integra con ConversationContext para multi-turn conversations

### `/whatsapp_integration/`
Patrones para integración con WhatsApp Web.js en producción.

**📱 `whatsapp_wrapper.py`** 
- ✅ **Qué muestra**: Asyncio wrapper para WhatsApp Web.js subprocess
- ✅ **Patrones clave**:
  - Session persistence (CRÍTICO para producción)
  - Rate limiting para evitar WhatsApp bans  
  - Reconnection logic con exponential backoff
  - Event-driven message handling
  - Queue management para async processing
- ✅ **Cuándo usar**: Para implementar whatsapp_service.py en src/services/
- ✅ **Render-ready**: Incluye patterns específicos para deployment en Render

### `/chains/` 
Patrones para cadenas LangChain con manejo de estado conversacional.

**🧠 `conversation_memory_chain.py`**
- ✅ **Qué muestra**: Memory management con Redis persistence 
- ✅ **Patrones clave**:
  - Redis-backed conversation history
  - Automatic context compression para token limits
  - Multi-user session isolation
  - TTL-based cleanup
  - Conversation summarization
- ✅ **Cuándo usar**: Para implementar context_service.py y memory chains
- ✅ **Producción**: Maneja cleanup automático y session persistence

## 🎯 Cómo Usar Estos Ejemplos

### 1. **Para Implementación Nueva**
```bash
# Copia el patrón y adapta para tu uso específico
cp examples/langchain_agents/intent_classifier_agent.py src/agents/intent_classifier.py

# Modifica según tus necesidades:
# - Tipos de intención específicos
# - Routing rules para tus departamentos  
# - Prompts personalizados para tu dominio
```

### 2. **Para Entender Patrones**
- 📖 **Lee los comentarios detallados** - explican el "por qué" de cada decisión
- 🔍 **Busca "CRÍTICO"** - marca considerations específicos de producción  
- ⚡ **Busca "Key patterns"** - identifica los patterns reusables
- 🚨 **Busca "Gotchas"** - evita problemas comunes

### 3. **Para Testing**
Cada ejemplo incluye:
- ✅ **`async def example_usage()`** - muestra cómo usar el component
- ✅ **Logging detallado** - para debugging
- ✅ **Error handling patterns** - para robustez en producción

## 🚨 Consideraciones Críticas

### **WhatsApp Integration** 
```python
# ❌ NUNCA ignores session persistence
WHATSAPP_SESSION_PATH = "./session"  # MUST be persistent filesystem

# ❌ NUNCA envíes messages sin rate limiting  
await asyncio.sleep(1.5)  # MINIMUM delay between messages

# ❌ NUNCA ignores disconnection events
# Implement proper reconnection logic or bot fails in production
```

### **OpenAI Integration**
```python  
# ❌ NUNCA uses high temperature for classification
temperature=0.1  # Low for consistent results

# ❌ NUNCA ignores rate limits
# Implement exponential backoff or API calls fail

# ❌ NUNCA sends full conversation history without compression
# Context window limits will cause failures
```

### **Redis Memory Management**
```python
# ❌ NUNCA stores conversation memory only in app memory
# App restarts will lose all context

# ❌ NUNCA forget TTL configuration  
redis.setex(key, ttl=7200, value=data)  # 2 hours default

# ❌ NUNCA stores raw conversation without compression
# Large conversations consume excessive Redis memory
```

## 🔧 Adaptación para Tu Proyecto

### **Personalización de Intenciones**
1. Modifica `IntentType` enum en `intent_classifier_agent.py`
2. Actualiza `department_routing` mapping
3. Ajusta few-shot examples en prompts
4. Actualiza routing rules en `routing_agent.py`

### **Personalización de WhatsApp**
1. Ajusta `rate_limiter.min_interval` based on your volume
2. Modifica `max_reconnection_attempts` para tu reliability needs
3. Customiza event handlers para tu business logic  
4. Ajusta session TTL based on your customer patterns

### **Personalización de Memory**
1. Ajusta `max_token_limit` based on your model choice
2. Modifica `memory_window` based on conversation length needs
3. Customiza summarization prompts para tu domain
4. Ajusta cleanup schedules based on your traffic

## 📋 Checklist de Implementation

Cuando uses estos ejemplos:

**Setup Phase:**
- [ ] Copy relevant example to your src/ structure
- [ ] Update imports y dependencies 
- [ ] Customize configuration values
- [ ] Update env variables in .env.example

**Integration Phase:**  
- [ ] Test with your OpenAI API key
- [ ] Verify Redis connection works
- [ ] Test WhatsApp session persistence locally
- [ ] Validate rate limiting prevents bans

**Production Phase:**
- [ ] Test on Render with persistent filesystem
- [ ] Monitor memory usage y token consumption  
- [ ] Verify cleanup jobs run properly
- [ ] Test reconnection logic under network issues

## 💡 Pro Tips

1. **Start Small**: Begin with `intent_classifier_agent.py` - it's self-contained
2. **Test Locally First**: WhatsApp authentication is easier locally 
3. **Monitor Token Usage**: OpenAI costs can escalate with poor memory management
4. **Use Logging**: These examples have detailed logging - keep it in production
5. **Handle Edge Cases**: Examples include fallback logic - don't remove it

## 🆘 Common Issues

### **"WhatsApp keeps disconnecting"**
- ✅ Check session persistence is working
- ✅ Verify filesystem is writable en Render
- ✅ Check reconnection logic is implemented

### **"OpenAI API calls failing"**  
- ✅ Check rate limiting implementation
- ✅ Verify function schema matches Pydantic models
- ✅ Check token limits aren't exceeded

### **"Redis memory growing infinitely"**
- ✅ Verify TTL is set on all keys
- ✅ Check cleanup jobs are running
- ✅ Verify compression is working for long conversations

¿Necesitas más ejemplos? Consulta `PLANNING.md` para la arquitectura completa o `TASK.md` para próximas features a implementar.