"""
Sistema de logging centralizado y estructurado.

Configura logging con formato estructurado, niveles apropiados
y rotación de archivos para el sistema de chatbot WhatsApp.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

from .config import get_settings

# Configuración global
_loggers: Dict[str, logging.Logger] = {}
_log_initialized = False


class StructuredFormatter(logging.Formatter):
    """
    Formatter que produce logs estructurados en JSON.
    
    Útil para parsing automático y agregación de logs en producción.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formatea record como JSON estructurado.
        
        Args:
            record: LogRecord a formatear
            
        Returns:
            String JSON con información estructurada
        """
        
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Añadir información de excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Añadir campos extras si existen
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'message_id'):
            log_data["message_id"] = record.message_id
        if hasattr(record, 'department'):
            log_data["department"] = record.department
        if hasattr(record, 'intent_type'):
            log_data["intent_type"] = record.intent_type
        if hasattr(record, 'processing_time'):
            log_data["processing_time_ms"] = record.processing_time
        
        return json.dumps(log_data, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Formatter readable para desarrollo y debugging.
    
    Produce output más legible para desarrollo local.
    """
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea record de manera legible."""
        formatted = super().format(record)
        
        # Añadir información extra si existe
        extras = []
        if hasattr(record, 'user_id'):
            extras.append(f"user:{record.user_id}")
        if hasattr(record, 'message_id'):
            extras.append(f"msg:{record.message_id[:8]}...")
        if hasattr(record, 'intent_type'):
            extras.append(f"intent:{record.intent_type}")
        if hasattr(record, 'processing_time'):
            extras.append(f"{record.processing_time}ms")
        
        if extras:
            formatted += f" [{', '.join(extras)}]"
        
        return formatted


def setup_logging():
    """
    Configura sistema de logging global.
    
    Establece handlers, formatters y niveles apropiados según el entorno.
    """
    global _log_initialized
    
    if _log_initialized:
        return
    
    settings = get_settings()
    
    # Limpiar configuración existente
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Nivel de logging global
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    root_logger.setLevel(log_level)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formatter según entorno
    if settings.is_production:
        # JSON estructurado para producción
        console_handler.setFormatter(StructuredFormatter())
    else:
        # Formato legible para desarrollo
        console_handler.setFormatter(HumanReadableFormatter())
    
    root_logger.addHandler(console_handler)
    
    # Handler para archivo de logs (solo si no es testing)
    if not settings.MOCK_EXTERNAL_SERVICES:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Rotación de archivos por tamaño
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "whatsapp_chatbot.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)  # Archivo siempre INFO+
        file_handler.setFormatter(StructuredFormatter())
        
        root_logger.addHandler(file_handler)
        
        # Error log separado
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10*1024*1024,  # 10MB 
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        root_logger.addHandler(error_handler)
    
    # Configurar loggers específicos de bibliotecas externas
    _configure_external_loggers()
    
    _log_initialized = True
    
    # Log inicial
    logger = logging.getLogger(__name__)
    logger.info(f"📋 Logging configurado - Nivel: {settings.LOG_LEVEL}, Entorno: {settings.ENVIRONMENT}")


def _configure_external_loggers():
    """Configura loggers de bibliotecas externas para reducir ruido."""
    
    # Reducir verbosidad de bibliotecas externas
    external_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'openai',
        'httpx',
        'asyncio',
        'subprocess'
    ]
    
    for logger_name in external_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
    
    # Redis específico - solo errores
    logging.getLogger('redis').setLevel(logging.ERROR)
    logging.getLogger('aioredis').setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene logger configurado para un módulo.
    
    Args:
        name: Nombre del módulo (típicamente __name__)
        
    Returns:
        Logger configurado
    """
    global _loggers
    
    if not _log_initialized:
        setup_logging()
    
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


@contextmanager
def log_context(**kwargs):
    """
    Context manager para añadir información contextual a logs.
    
    Args:
        **kwargs: Campos contextuales a añadir
        
    Usage:
        with log_context(user_id="123", intent_type="SALES"):
            logger.info("Procesando mensaje")
    """
    
    # Crear adapter que añade contexto
    class ContextLoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs_inner):
            # Fusionar kwargs del contexto con los del log
            for key, value in kwargs.items():
                if key not in kwargs_inner:
                    setattr(self.logger, key, value)
            return msg, kwargs_inner
    
    # Aplicar contexto a todos los loggers activos
    original_loggers = {}
    for name, logger in _loggers.items():
        original_loggers[name] = logger
        _loggers[name] = ContextLoggerAdapter(logger, kwargs)
    
    try:
        yield
    finally:
        # Restaurar loggers originales
        _loggers.update(original_loggers)


def log_whatsapp_event(logger: logging.Logger, event_type: str, data: Dict[str, Any]):
    """
    Log especializado para eventos de WhatsApp.
    
    Args:
        logger: Logger a usar
        event_type: Tipo de evento WhatsApp
        data: Datos del evento
    """
    
    logger.info(
        f"WhatsApp event: {event_type}",
        extra={
            'event_type': event_type,
            'whatsapp_data': data
        }
    )


def log_intent_classification(
    logger: logging.Logger,
    user_id: str,
    message: str,
    intent_type: str,
    confidence: float,
    processing_time: float
):
    """
    Log especializado para clasificación de intenciones.
    
    Args:
        logger: Logger a usar
        user_id: ID del usuario
        message: Mensaje clasificado
        intent_type: Tipo de intención detectada
        confidence: Nivel de confianza
        processing_time: Tiempo de procesamiento en ms
    """
    
    logger.info(
        f"Intent classified: {intent_type} (confidence: {confidence:.2f})",
        extra={
            'user_id': user_id,
            'intent_type': intent_type,
            'confidence_score': confidence,
            'processing_time': processing_time,
            'message_preview': message[:50] + '...' if len(message) > 50 else message
        }
    )


def log_message_routing(
    logger: logging.Logger,
    user_id: str,
    message_id: str,
    intent_type: str,
    department: str,
    routing_reason: str
):
    """
    Log especializado para routing de mensajes.
    
    Args:
        logger: Logger a usar
        user_id: ID del usuario
        message_id: ID del mensaje
        intent_type: Tipo de intención
        department: Departamento destino
        routing_reason: Razón del routing
    """
    
    logger.info(
        f"Message routed to {department}",
        extra={
            'user_id': user_id,
            'message_id': message_id,
            'intent_type': intent_type,
            'department': department,
            'routing_reason': routing_reason
        }
    )


def log_api_call(
    logger: logging.Logger,
    service: str,
    endpoint: str,
    duration_ms: float,
    success: bool,
    error: Optional[str] = None
):
    """
    Log especializado para llamadas API externas.
    
    Args:
        logger: Logger a usar
        service: Nombre del servicio (OpenAI, Redis, etc.)
        endpoint: Endpoint llamado
        duration_ms: Duración en milisegundos
        success: Si fue exitosa
        error: Mensaje de error si falló
    """
    
    level = logging.INFO if success else logging.ERROR
    message = f"{service} API call {'succeeded' if success else 'failed'}: {endpoint}"
    
    extra_data = {
        'service': service,
        'endpoint': endpoint,
        'duration_ms': duration_ms,
        'success': success
    }
    
    if error:
        extra_data['error_message'] = error
    
    logger.log(level, message, extra=extra_data)


class LoggingMiddleware:
    """
    Middleware para logging de requests HTTP en FastAPI.
    
    Registra información de requests/responses para debugging.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def __call__(self, request, call_next):
        """Procesa request y response logging."""
        
        start_time = datetime.now()
        
        # Log request
        self.logger.info(
            f"HTTP {request.method} {request.url.path}",
            extra={
                'http_method': request.method,
                'http_path': request.url.path,
                'client_ip': request.client.host if request.client else 'unknown',
                'user_agent': request.headers.get('user-agent', 'unknown')
            }
        )
        
        # Procesar request
        try:
            response = await call_next(request)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log response
            self.logger.info(
                f"HTTP {response.status_code} - {duration:.2f}ms",
                extra={
                    'http_status': response.status_code,
                    'response_time_ms': duration,
                    'http_method': request.method,
                    'http_path': request.url.path
                }
            )
            
            return response
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log error
            self.logger.error(
                f"HTTP request failed - {duration:.2f}ms",
                extra={
                    'http_method': request.method,
                    'http_path': request.url.path,
                    'error_message': str(e),
                    'response_time_ms': duration
                },
                exc_info=True
            )
            
            raise


# Auto-inicializar logging cuando se importa el módulo
setup_logging()