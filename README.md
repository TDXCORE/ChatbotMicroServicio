# 🤖 Sistema de Chatbot WhatsApp con Detección de Intenciones

Sistema inteligente de chatbot que detecta automáticamente las intenciones de los clientes en WhatsApp y los enruta al departamento correspondiente usando LangChain/LangGraph con OpenAI.

## 🎯 **Características Principales**

- ✅ **Detección de Intenciones**: Clasificación automática con 85%+ precisión usando OpenAI GPT-4
- ✅ **Integración WhatsApp**: Conexión nativa con WhatsApp Web.js
- ✅ **Enrutamiento Inteligente**: Direccionamiento automático a 7 departamentos diferentes  
- ✅ **Contexto Conversacional**: Memoria persistente de conversaciones con Redis
- ✅ **Respuesta Rápida**: Tiempo de respuesta < 3 segundos
- ✅ **Alta Concurrencia**: Soporte para 100+ conversaciones simultáneas
- ✅ **Deploy en Render**: Configuración lista para producción

## 🏗️ **Arquitectura del Sistema**

```
whatsapp-chatbot/
├── src/
│   ├── agents/                 # Agentes LangChain
│   │   ├── intent_classifier.py    # Clasificador de intenciones
│   │   ├── conversation_agent.py   # Agente conversacional
│   │   └── routing_agent.py        # Agente de enrutamiento
│   ├── chains/                 # Cadenas LangChain  
│   │   ├── intent_chain.py         # Pipeline de detección
│   │   ├── response_chain.py       # Generación de respuestas
│   │   └── context_chain.py        # Manejo de contexto
│   ├── services/               # Servicios de integración
│   │   ├── whatsapp_service.py     # WhatsApp Web.js wrapper
│   │   ├── llm_service.py          # OpenAI service
│   │   └── context_service.py      # Gestión Redis
│   ├── models/                 # Modelos Pydantic
│   └── utils/                  # Utilidades y helpers
├── tests/                      # Test suite completo
├── examples/                   # Patrones de implementación
└── main.py                     # FastAPI app principal
```

## 🚀 **Quick Start**

### 1. **Configuración del Entorno**

```bash
# Clonar el repositorio
git clone <your-repo>
cd whatsapp-chatbot

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. **Variables de Entorno**

Crea un archivo `.env` basado en `.env.example`:

```bash
# OpenAI Configuration (OBLIGATORIO)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4-1106-preview
OPENAI_MAX_TOKENS=1000

# WhatsApp Configuration (OBLIGATORIO)
WHATSAPP_SESSION_PATH=./session
WHATSAPP_HEADLESS=true

# Redis Configuration (OBLIGATORIO para producción)
REDIS_URL=redis://localhost:6379
REDIS_SESSION_TTL=7200

# Rate Limiting (RECOMENDADO)
RATE_LIMIT_PER_USER=10
RATE_LIMIT_WINDOW=60

# Render Configuration (para deployment)
PORT=10000
ENVIRONMENT=production
```

### 3. **Configurar WhatsApp**

**Primera vez (desarrollo local):**
```bash
# Ejecutar el bot
python main.py

# 🔍 IMPORTANTE: Aparecerá un QR code en la terminal
# Escanéalo con tu WhatsApp para autenticar la sesión
# La sesión se guardará en ./session/ para futuros usos
```

**En producción (Render):**
- La autenticación debe hacerse localmente primero
- Subir la carpeta `./session/` al repositorio (temporalmente)
- Después del primer deploy, remover del git por seguridad

### 4. **Configurar Redis** 

**Desarrollo local:**
```bash
# Instalar Redis
sudo apt install redis-server  # Ubuntu
brew install redis            # MacOS

# Iniciar Redis
redis-server
```

**Producción (Render):**
- Añadir Redis addon en el dashboard de Render
- La variable `REDIS_URL` se configurará automáticamente

### 5. **Ejecutar el Sistema**

```bash
# Desarrollo
python main.py

# El sistema estará disponible en:
# - API: http://localhost:8000
# - Health Check: http://localhost:8000/health
# - Webhook: http://localhost:8000/webhook/whatsapp
```

## 📋 **Tipos de Intenciones Soportadas**

| Tipo | Ejemplos | Departamento |
|------|----------|--------------|
| **SALES** | "¿Cuánto cuesta?", "Quiero comprar" | ventas@empresa.com |
| **SUPPORT** | "Tengo un problema", "No funciona" | soporte@empresa.com |
| **BILLING** | "Mi factura", "Cobros" | facturacion@empresa.com |
| **GENERAL** | "Horarios", "Ubicación" | info@empresa.com |
| **COMPLAINT** | "Queja", "Reclamo" | gerencia@empresa.com |
| **INFORMATION** | "Información sobre servicios" | info@empresa.com |
| **UNKNOWN** | Mensajes no clasificables | soporte@empresa.com |

## 🧪 **Testing**

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_intent_classifier.py -v
pytest tests/test_whatsapp_service.py -v

# Coverage
pytest tests/ --cov=src --cov-report=term-missing

# Target: 80%+ coverage
```

## 🚀 **Deployment en Render**

### 1. **Configurar render.yaml**
```yaml
services:
  - type: web
    name: whatsapp-chatbot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: REDIS_URL  
        fromDatabase:
          name: whatsapp-redis
          property: connectionString
```

### 2. **Variables de Entorno en Render**
- `OPENAI_API_KEY`: Tu API key de OpenAI
- `WHATSAPP_HEADLESS`: `true`
- `ENVIRONMENT`: `production`
- `PORT`: `10000`

### 3. **Deploy**
```bash
# Commit cambios
git add .
git commit -m "Initial chatbot setup"
git push origin main

# Render auto-deployará desde el push
```

## 🔧 **Desarrollo y Contribución**

### **Estructura de Desarrollo**
```bash
# 1. Siempre leer PLANNING.md para entender arquitectura
# 2. Consultar TASK.md para tareas pendientes  
# 3. Agregar nuevas tareas a TASK.md
# 4. Seguir patrones en examples/
```

### **Patrones de Código**
- **Agentes LangChain**: Un agente por responsabilidad específica
- **Chains**: Pipelines secuenciales para procesamiento
- **Services**: Wrappers para integraciones externas
- **Models**: Validación Pydantic para todos los datos
- **Tests**: Mockear servicios externos, testing async

### **Agregar Nuevo Tipo de Intención**
1. Actualizar `src/models/intents.py`
2. Configurar routing en `src/agents/routing_agent.py`
3. Actualizar prompts en `src/config/prompts.py`
4. Añadir tests en `tests/test_intent_classifier.py`

## 📚 **Recursos y Documentación**

### **LangChain/LangGraph**
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Workflows](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

### **WhatsApp Web.js**
- [GitHub Repository](https://github.com/pedroslopez/whatsapp-web.js)
- [API Documentation](https://docs.wwebjs.dev/)

### **OpenAI**
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Best Practices](https://platform.openai.com/docs/guides/production-best-practices)

## ⚠️ **Troubleshooting**

### **WhatsApp Issues**
```bash
# ❌ "WhatsApp session not found"
# ✅ Eliminar carpeta ./session/ y re-autenticar con QR

# ❌ "WhatsApp disconnected frequently"  
# ✅ Verificar WHATSAPP_SESSION_PATH es persistente

# ❌ Bot enviando mensajes muy lento
# ✅ Revisar rate limiting, WhatsApp tiene límites agresivos
```

### **OpenAI Issues**
```bash
# ❌ "Rate limit exceeded"
# ✅ Implementado exponential backoff automático

# ❌ "Context length exceeded"  
# ✅ Sistema comprime automáticamente conversaciones largas

# ❌ "Invalid function call format"
# ✅ Verificar modelos Pydantic están bien definidos
```

### **Redis Issues**
```bash
# ❌ "Redis connection failed"
# ✅ Verificar REDIS_URL y que Redis server esté corriendo

# ❌ "Session data lost"
# ✅ Configurar Redis TTL apropiadamente en variables entorno
```

### **Render Issues**
```bash
# ❌ "Build failed"
# ✅ Verificar requirements.txt tiene todas las dependencias

# ❌ "Health check failed"  
# ✅ Endpoint /health debe retornar status 200

# ❌ "App sleeping"
# ✅ Considerar plan paid o implementar keep-alive
```

## 📊 **Monitoring y Métricas**

- **Response Time**: < 3 segundos objetivo
- **Intent Accuracy**: 85%+ precisión objetivo  
- **Uptime**: 99%+ availability objetivo
- **Concurrent Users**: 100+ conversaciones simultáneas

## 📞 **Soporte**

¿Problemas con la configuración? ¿Bugs encontrados? 

1. Consulta la sección **Troubleshooting** arriba
2. Revisa los [Issues del proyecto](issues)
3. Para bugs nuevos, crea un issue con:
   - Descripción del problema
   - Logs relevantes  
   - Pasos para reproducir
   - Configuración de entorno

---

## 📄 **Licencia**

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---

*🚀 ¡Happy chatbot building!*