"""
FastAPI Main Application - WhatsApp Chatbot con Detección de Intenciones

Aplicación principal que maneja webhooks de WhatsApp, procesa mensajes
con LangChain/OpenAI y enruta a departamentos correspondientes.
"""

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, WebSocket, WebSocketDisconnect
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

# WhatsApp QR Integration - Store para QR codes y WebSockets por usuario
user_qr_codes: Dict[str, str] = {}
user_websockets: Dict[str, list] = {}
user_whatsapp_services: Dict[str, WhatsAppService] = {}  # WhatsApp service por usuario


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
        
        # Context Service (Redis) - Optional in production/Render
        # Detect if we're running on Render platform
        is_render = os.environ.get("RENDER") == "true" or os.environ.get("RENDER_EXTERNAL_URL") is not None
        logger.info(f"🔍 Environment detection: production={settings.is_production}, render={is_render}")
        
        context_service = ContextService()
        try:
            if not await context_service.initialize():
                if settings.is_production or is_render:
                    logger.warning("⚠️ Context Service disabled in production/Render (Redis not available)")
                    context_service = None
                else:
                    logger.error("❌ Error inicializando Context Service")
                    raise RuntimeError("Context Service initialization failed")
        except Exception as e:
            if settings.is_production or is_render:
                logger.warning(f"⚠️ Context Service disabled in production/Render: {e}")
                context_service = None
            else:
                logger.error(f"❌ Error inicializando Context Service: {e}")
                raise RuntimeError("Context Service initialization failed")
        
        # LLM Service (OpenAI) - Optional in production/Render without API key
        try:
            llm_service = LLMService()
        except Exception as e:
            if settings.is_production or is_render:
                logger.warning(f"⚠️ LLM Service disabled in production/Render: {e}")
                logger.warning("Intent classification will use fallback logic only")
                llm_service = None
            else:
                logger.error(f"❌ Error inicializando LLM Service: {e}")
                raise RuntimeError(f"LLM Service initialization failed: {e}")
        
        # Intent Classifier Agent
        intent_classifier = IntentClassifierAgent(
            llm_service=llm_service,
            context_service=context_service  # Can be None in production
        )
        
        # WhatsApp Service - Optional in production/Render
        whatsapp_service = WhatsAppService()
        
        if settings.is_production or is_render:
            logger.warning("⚠️ WhatsApp Service disabled in production/Render")
            logger.warning("Core API functionality available without WhatsApp integration")
        else:
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

# Configurar CORS - Permisivo para WhatsApp QR Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes temporalmente
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
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
            "classify": "/classify",
            "whatsapp_qr": {
                "init": "POST /whatsapp/init/{user_id}",
                "qr": "GET /whatsapp/qr/{user_id}",
                "status": "GET /whatsapp/status/{user_id}",
                "disconnect": "POST /whatsapp/disconnect/{user_id}",
                "stats": "GET /whatsapp/stats/{user_id}",
                "websocket": "WS /ws/{user_id}"
            }
        }
    }


# ================================
# WhatsApp QR Integration Endpoints
# ================================

@app.post("/whatsapp/init/{user_id}")
async def init_whatsapp_session(user_id: str):
    """
    Inicializar sesión WhatsApp para usuario específico.
    
    Genera un código QR REAL usando WhatsApp Web.js para que el usuario pueda conectar su WhatsApp.
    """
    try:
        logger.info(f"🔄 Iniciando sesión WhatsApp REAL para usuario: {user_id}")
        
        # Crear WhatsApp service dedicado para este usuario
        user_wa_service = WhatsAppService()
        
        # Configurar session path único por usuario
        user_session_path = user_wa_service.session_path.parent / f"session_{user_id}"
        user_wa_service.session_path = user_session_path
        user_session_path.mkdir(exist_ok=True, parents=True)
        
        # Registrar handler para QR code
        qr_received = False
        qr_code_data = None

        def on_qr_received(qr_data):
            nonlocal qr_received, qr_code_data
            qr_received = True
            qr_code_data = qr_data.get('qr') if isinstance(qr_data, dict) else qr_data
            logger.info(f"📱 QR Code REAL generado para usuario {user_id}: {qr_code_data[:50]}...")

            # Almacenar QR real
            user_qr_codes[user_id] = qr_code_data

            # Notificar via WebSocket
            if user_id in user_websockets:
                for ws in user_websockets[user_id][:]:
                    try:
                        asyncio.create_task(ws.send_json({
                            "type": "qr_generated",
                            "qr": qr_code_data,
                            "user_id": user_id,
                            "message": "Real WhatsApp QR code generated"
                        }))
                    except:
                        user_websockets[user_id].remove(ws)

        def on_authenticated(session_data):
            logger.info(f"✅ Usuario {user_id} autenticado en WhatsApp: {session_data}")

            # Notificar autenticación via WebSocket
            if user_id in user_websockets:
                for ws in user_websockets[user_id][:]:
                    try:
                        asyncio.create_task(ws.send_json({
                            "type": "authenticated",
                            "phone_number": session_data.get('phone_number') if isinstance(session_data, dict) else 'unknown',
                            "user_id": user_id,
                            "message": "WhatsApp authenticated successfully"
                        }))
                    except:
                        user_websockets[user_id].remove(ws)

        # Registrar eventos
        user_wa_service.register_event_handler('qr_code', on_qr_received)
        user_wa_service.register_event_handler('authenticated', on_authenticated)
        
        # Iniciar WhatsApp service específicamente para QR generation
        try:
            # Usar método especializado para QR generation que funciona en producción
            qr_start_success = await user_wa_service.start_for_qr_generation()
            
            if qr_start_success:
                user_whatsapp_services[user_id] = user_wa_service
                
                # Esperar hasta 45 segundos por el QR code (más tiempo para producción)
                for i in range(45):
                    if qr_received:
                        break
                    await asyncio.sleep(1)
                
                if qr_received and qr_code_data:
                    return {
                        "status": "success", 
                        "message": "WhatsApp session initialized with REAL QR code",
                        "qr_code": qr_code_data,
                        "user_id": user_id,
                        "qr_type": "real_whatsapp_web",
                        "environment": "production" if settings.is_production else "development"
                    }
                else:
                    # Si no se genera QR en 45 segundos, devolver error
                    await user_wa_service.stop()
                    if user_id in user_whatsapp_services:
                        del user_whatsapp_services[user_id]
                    raise HTTPException(status_code=408, detail="Timeout waiting for WhatsApp QR code generation")
            else:
                # Si no se puede iniciar el servicio para QR, usar fallback
                raise Exception("WhatsApp service could not start for QR generation")
                
        except Exception as wa_error:
            logger.error(f"❌ Error con WhatsApp Web.js para {user_id}: {wa_error}")
            
            # Fallback: generar QR simulado con aviso
            qr_data = f"DEMO-whatsapp-session-{user_id}-{int(time.time())}"
            user_qr_codes[user_id] = qr_data
            
            # Notificar via WebSocket
            if user_id in user_websockets:
                for ws in user_websockets[user_id][:]:
                    try:
                        await ws.send_json({
                            "type": "qr_generated",
                            "qr": qr_data,
                            "user_id": user_id,
                            "message": "Demo QR code - WhatsApp Web.js not available",
                            "demo_mode": True
                        })
                    except:
                        user_websockets[user_id].remove(ws)
            
            return {
                "status": "success",
                "message": "Demo mode - WhatsApp Web.js not available, showing demo QR",
                "qr_code": qr_data,
                "user_id": user_id,
                "qr_type": "demo_fallback",
                "warning": "This is a demo QR code. Real WhatsApp integration requires Node.js environment."
            }
            
    except Exception as e:
        logger.error(f"❌ Error inicializando WhatsApp para {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/whatsapp/qr/{user_id}")  
async def get_whatsapp_qr(user_id: str):
    """
    Obtener código QR para usuario específico.
    
    Returns el QR code generado previamente, si existe.
    """
    if user_id not in user_qr_codes:
        raise HTTPException(status_code=404, detail="QR code not found for user. Please initialize session first.")
    
    return {
        "qr_code": user_qr_codes[user_id],
        "user_id": user_id,
        "status": "pending_scan",
        "expires_in": "300 seconds"
    }


@app.get("/whatsapp/status/{user_id}")
async def get_whatsapp_status(user_id: str):
    """
    Obtener estado de conexión WhatsApp del usuario.
    
    Returns el estado actual de la sesión WhatsApp.
    """
    # Determinar estado basado en si hay QR generado
    if user_id in user_qr_codes:
        status = "qr_generated"
        connected = False
    else:
        status = "disconnected" 
        connected = False
    
    # Verificar si hay WebSocket activo (indica sesión activa)
    websocket_active = user_id in user_websockets and len(user_websockets[user_id]) > 0
    
    return {
        "user_id": user_id,
        "status": status,
        "connected": connected,
        "websocket_active": websocket_active,
        "phone_number": None,  # Placeholder para futura implementación
        "last_seen": None,
        "qr_available": user_id in user_qr_codes
    }


@app.post("/whatsapp/disconnect/{user_id}")
async def disconnect_whatsapp(user_id: str):
    """
    Desconectar WhatsApp del usuario específico.
    
    Limpia la sesión y notifica via WebSocket.
    """
    try:
        logger.info(f"🔌 Desconectando WhatsApp para usuario: {user_id}")
        
        # Limpiar QR code
        if user_id in user_qr_codes:
            del user_qr_codes[user_id]
        
        # Notificar desconexión via WebSocket
        if user_id in user_websockets:
            for ws in user_websockets[user_id][:]:  # Copy list
                try:
                    await ws.send_json({
                        "type": "disconnected",
                        "user_id": user_id,
                        "message": "WhatsApp session disconnected"
                    })
                except:
                    # Remover WebSocket si no funciona
                    user_websockets[user_id].remove(ws)
            
            # Limpiar lista si está vacía
            if not user_websockets[user_id]:
                del user_websockets[user_id]
        
        return {
            "status": "success",
            "message": "WhatsApp disconnected successfully",
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"❌ Error desconectando WhatsApp para {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/whatsapp/stats/{user_id}")
async def get_whatsapp_stats(user_id: str):
    """
    Obtener estadísticas WhatsApp del usuario específico.
    
    Returns métricas de uso de WhatsApp para el usuario.
    """
    # Por ahora devolvemos estadísticas mock
    # En el futuro se integrarán con el WhatsApp service real
    return {
        "user_id": user_id,
        "messages_sent": 0,
        "messages_received": 0,
        "session_active": user_id in user_qr_codes,
        "websocket_connections": len(user_websockets.get(user_id, [])),
        "connected_since": None,
        "last_activity": None,
        "total_sessions": 1 if user_id in user_qr_codes else 0
    }


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint para updates en tiempo real por usuario.
    
    Mantiene conexión activa y envía notificaciones de estado WhatsApp.
    """
    await websocket.accept()
    logger.info(f"🔗 WebSocket conectado para usuario: {user_id}")
    
    # Registrar WebSocket para este usuario
    if user_id not in user_websockets:
        user_websockets[user_id] = []
    user_websockets[user_id].append(websocket)
    
    try:
        # Enviar estado inicial si hay QR disponible
        if user_id in user_qr_codes:
            await websocket.send_json({
                "type": "qr_generated",
                "qr": user_qr_codes[user_id],
                "user_id": user_id,
                "message": "QR code available"
            })
        else:
            await websocket.send_json({
                "type": "status_update",
                "status": "disconnected",
                "user_id": user_id,
                "message": "No active WhatsApp session"
            })
        
        # Mantener conexión viva y escuchar mensajes
        while True:
            try:
                # Recibir mensaje del cliente (heartbeat o comandos)
                data = await websocket.receive_text()
                
                # Echo para confirmar conexión activa
                await websocket.send_json({
                    "type": "ping_response",
                    "message": "WebSocket connection active",
                    "user_id": user_id,
                    "timestamp": time.time()
                })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"⚠️ Error en WebSocket para {user_id}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket desconectado para usuario: {user_id}")
    finally:
        # Limpiar WebSocket cuando se desconecta
        if user_id in user_websockets:
            try:
                user_websockets[user_id].remove(websocket)
                # Eliminar lista vacía
                if not user_websockets[user_id]:
                    del user_websockets[user_id]
            except ValueError:
                pass  # WebSocket ya no estaba en la lista


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
