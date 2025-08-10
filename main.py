"""
FastAPI Main Application - WhatsApp Chatbot con Detección de Intenciones

Aplicación principal que maneja webhooks de WhatsApp, procesa mensajes
con LangChain/OpenAI y enruta a departamentos correspondientes.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.models.intents import IntentClassificationRequest, IntentType
from src.models.messages import WhatsAppMessage, MessageType, MessageStatus
from src.services.whatsapp_service import WhatsAppService
from src.services.llm_service import LLMService
from src.services.context_service import ContextService
from src.agents.intent_classifier import IntentClassifierAgent
from src.utils.config import get_settings
from src.utils.logger import get_logger, LoggingMiddleware, log_message_routing

# Configuración
settings = get_settings()
logger = get_logger(__name__)

# Servicios globales - inicializados en startup
whatsapp_service: Optional[WhatsAppService] = None
llm_service: Optional[LLMService] = None
context_service: Optional[ContextService] = None
intent_classifier: Optional[IntentClassifierAgent] = None

# Estadísticas globales
app_stats = {
    "messages_received": 0,
    "messages_processed": 0,
    "messages_failed": 0,
    "intents_classified": 0,
    "uptime_start": time.time(),
    "last_activity": time.time()
}


class WebhookMessage(BaseModel):
    """Modelo para webhooks entrantes de WhatsApp."""
    id: str
    from_number: str
    body: str
    timestamp: str
    message_type: str = "text"


class HealthResponse(BaseModel):
    """Respuesta del endpoint de health check."""
    status: str
    timestamp: float
    uptime_seconds: float
    services: Dict[str, Any]
    stats: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager para la aplicación.
    
    Inicializa servicios en startup y los limpia en shutdown.
    """
    global whatsapp_service, llm_service, context_service, intent_classifier
    
    logger.info("🚀 Iniciando WhatsApp Chatbot Application...")
    
    try:
        # Inicializar servicios
        logger.info("📦 Inicializando servicios...")
        
        # Context Service (Redis)
        context_service = ContextService()
        if not await context_service.initialize():
            logger.error("❌ Error inicializando Context Service")
            raise RuntimeError("Context Service initialization failed")
        
        # LLM Service (OpenAI)
        llm_service = LLMService()
        
        # Intent Classifier Agent
        intent_classifier = IntentClassifierAgent(
            llm_service=llm_service,
            context_service=context_service
        )
        
        # WhatsApp Service
        whatsapp_service = WhatsAppService()
        
        # Registrar event handler para mensajes entrantes
        whatsapp_service.register_event_handler(
            'message_received',
            handle_whatsapp_message
        )
        
        # Iniciar WhatsApp service
        if not await whatsapp_service.start():
            logger.error("❌ Error iniciando WhatsApp Service")
            raise RuntimeError("WhatsApp Service initialization failed")
        
        logger.info("✅ Todos los servicios iniciados correctamente")
        
        yield  # Aquí la app está corriendo
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise
    
    finally:
        # Cleanup en shutdown
        logger.info("🛑 Cerrando aplicación...")
        
        if whatsapp_service:
            await whatsapp_service.stop()
        
        if context_service:
            await context_service.close()
        
        logger.info("✅ Aplicación cerrada correctamente")


# Crear aplicación FastAPI
app = FastAPI(
    title="WhatsApp Chatbot con Detección de Intenciones",
    description="Sistema inteligente que detecta intenciones y enruta mensajes WhatsApp",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Añadir logging middleware
app.add_middleware(LoggingMiddleware, logger=logger)


async def handle_whatsapp_message(message_data: Dict[str, Any]):
    """
    Handler para mensajes entrantes de WhatsApp.
    
    Procesa mensajes usando background tasks para evitar timeouts.
    """
    try:
        # Crear mensaje WhatsApp
        message = WhatsAppMessage(
            id=message_data.get('id', ''),
            from_number=message_data.get('from', ''),
            to_number=message_data.get('to', ''),
            body=message_data.get('body', ''),
            message_type=MessageType(message_data.get('type', 'text')),
            metadata=message_data
        )
        
        # Procesar en background
        asyncio.create_task(process_incoming_message(message))
        
        app_stats["messages_received"] += 1
        app_stats["last_activity"] = time.time()
        
        logger.info(f"📨 Mensaje WhatsApp encolado para procesamiento: {message.id}")
        
    except Exception as e:
        logger.error(f"❌ Error manejando mensaje WhatsApp: {e}")


async def process_incoming_message(message: WhatsAppMessage):
    """
    Procesa mensaje entrante usando intent classification y routing.
    
    Args:
        message: Mensaje WhatsApp a procesar
    """
    try:
        logger.info(f"⚙️ Procesando mensaje de {message.from_number}: {message.body[:50]}...")
        
        # Actualizar estado del mensaje
        message.status = MessageStatus.PROCESSING
        
        # Añadir mensaje al historial conversacional
        await context_service.add_message_to_history(
            user_id=message.from_number,
            message=message
        )
        
        # Clasificar intención
        classification_request = IntentClassificationRequest(
            message=message.body,
            user_id=message.from_number,
            timestamp=message.timestamp
        )
        
        classification_response = await intent_classifier.classify_intent(classification_request)
        
        # Actualizar estadísticas
        app_stats["intents_classified"] += 1
        
        # Generar respuesta automática
        auto_response = await llm_service.generate_response(
            intent_type=classification_response.message_intent.intent,
            user_message=message.body,
            context=classification_response.context_updated
        )
        
        # Enviar respuesta automática
        success = await whatsapp_service.send_message(
            to_number=message.from_number,
            message=auto_response
        )
        
        if success:
            message.status = MessageStatus.RESPONDED
            
            # Log routing decision
            log_message_routing(
                logger=logger,
                user_id=message.from_number,
                message_id=message.id,
                intent_type=classification_response.message_intent.intent.value,
                department=classification_response.message_intent.routing_department,
                routing_reason=classification_response.message_intent.reasoning
            )
            
            logger.info(
                f"✅ Mensaje procesado exitosamente: {classification_response.message_intent.intent} "
                f"-> {classification_response.message_intent.routing_department}"
            )
        else:
            message.status = MessageStatus.FAILED
            logger.error(f"❌ Error enviando respuesta automática para mensaje {message.id}")
        
        app_stats["messages_processed"] += 1
        
    except Exception as e:
        message.status = MessageStatus.FAILED
        app_stats["messages_failed"] += 1
        
        logger.error(f"❌ Error procesando mensaje {message.id}: {e}", exc_info=True)
        
        # Enviar mensaje de error genérico
        try:
            await whatsapp_service.send_message(
                to_number=message.from_number,
                message="Disculpa, hubo un error procesando tu mensaje. Te responderemos pronto."
            )
        except Exception as send_error:
            logger.error(f"❌ Error enviando mensaje de error: {send_error}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de health check para monitoreo.
    
    Verifica estado de todos los servicios y retorna métricas.
    """
    try:
        uptime = time.time() - app_stats["uptime_start"]
        
        # Verificar servicios
        services_status = {}
        
        # WhatsApp Service
        if whatsapp_service:
            whatsapp_status = await whatsapp_service.get_status()
            services_status["whatsapp"] = {
                "status": "healthy" if whatsapp_service.is_connected else "unhealthy",
                "details": whatsapp_status
            }
        else:
            services_status["whatsapp"] = {"status": "not_initialized"}
        
        # Context Service (Redis)
        if context_service:
            context_health = await context_service.health_check()
            services_status["context_redis"] = context_health
        else:
            services_status["context_redis"] = {"status": "not_initialized"}
        
        # LLM Service (OpenAI)
        if llm_service:
            llm_health = await llm_service.health_check()
            services_status["llm_openai"] = llm_health
        else:
            services_status["llm_openai"] = {"status": "not_initialized"}
        
        # Intent Classifier
        if intent_classifier:
            classifier_health = await intent_classifier.health_check()
            services_status["intent_classifier"] = classifier_health
        else:
            services_status["intent_classifier"] = {"status": "not_initialized"}
        
        # Determinar estado general
        all_healthy = all(
            service.get("status") == "healthy" 
            for service in services_status.values()
        )
        
        overall_status = "healthy" if all_healthy else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=time.time(),
            uptime_seconds=uptime,
            services=services_status,
            stats=app_stats
        )
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            uptime_seconds=time.time() - app_stats["uptime_start"],
            services={"error": str(e)},
            stats=app_stats
        )


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    webhook_data: WebhookMessage,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Webhook endpoint para recibir mensajes de WhatsApp.
    
    Procesa mensajes usando background tasks para evitar timeouts.
    """
    try:
        logger.info(f"📥 Webhook WhatsApp recibido de {webhook_data.from_number}")
        
        # Validación básica
        if not webhook_data.body.strip():
            logger.warning("⚠️ Mensaje vacío recibido")
            return {"status": "ignored", "reason": "empty_message"}
        
        # Crear mensaje WhatsApp
        message = WhatsAppMessage(
            id=webhook_data.id,
            from_number=webhook_data.from_number,
            to_number="chatbot",  # Nuestro bot
            body=webhook_data.body,
            message_type=MessageType(webhook_data.message_type),
            status=MessageStatus.RECEIVED
        )
        
        # Procesar en background para evitar timeout
        background_tasks.add_task(process_incoming_message, message)
        
        app_stats["messages_received"] += 1
        app_stats["last_activity"] = time.time()
        
        return {
            "status": "accepted",
            "message_id": webhook_data.id,
            "processing": "background"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en webhook WhatsApp: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Webhook processing error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Endpoint para obtener estadísticas detalladas del sistema.
    
    Returns:
        Estadísticas de todos los servicios
    """
    try:
        stats = {
            "app_stats": app_stats,
            "uptime_seconds": time.time() - app_stats["uptime_start"],
        }
        
        # Añadir stats de servicios si están disponibles
        if whatsapp_service:
            stats["whatsapp_service"] = whatsapp_service.get_stats() if hasattr(whatsapp_service, 'get_stats') else {}
        
        if llm_service:
            stats["llm_service"] = llm_service.get_stats()
        
        if context_service:
            stats["context_service"] = context_service.get_stats()
        
        if intent_classifier:
            stats["intent_classifier"] = intent_classifier.get_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.post("/classify")
async def classify_message(request: IntentClassificationRequest):
    """
    Endpoint para clasificar intenciones manualmente.
    
    Útil para testing y debugging.
    """
    try:
        if not intent_classifier:
            raise HTTPException(status_code=503, detail="Intent classifier not available")
        
        response = await intent_classifier.classify_intent(request)
        
        return {
            "intent": response.message_intent.intent.value,
            "confidence": response.message_intent.confidence,
            "department": response.message_intent.routing_department,
            "reasoning": response.message_intent.reasoning,
            "processing_time_ms": response.processing_time_ms,
            "fallback_used": response.fallback_used
        }
        
    except Exception as e:
        logger.error(f"❌ Error en clasificación manual: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/")
async def root():
    """Endpoint raíz con información básica."""
    return {
        "service": "WhatsApp Chatbot con Detección de Intenciones",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": time.time() - app_stats["uptime_start"],
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook/whatsapp", 
            "stats": "/stats",
            "classify": "/classify"
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para excepciones no manejadas."""
    logger.error(f"❌ Excepción global no manejada: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    # Configuración para desarrollo local y producción
    import os
    
    # Render sets PORT as environment variable
    port = int(os.environ.get("PORT", settings.PORT))
    host = os.environ.get("HOST", settings.HOST)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )