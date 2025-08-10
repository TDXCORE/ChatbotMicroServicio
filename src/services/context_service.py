"""
Context Service - Gestión de contexto conversacional con Redis

Servicio para manejar memoria conversacional, sesiones de usuario
y compresión automática de historial usando Redis como backend.

Basado en: examples/chains/conversation_memory_chain.py
"""

import asyncio
import json
import gzip
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

from ..models.intents import ConversationContext, MessageIntent
from ..models.messages import WhatsAppMessage, MessageHistory
from ..utils.config import get_settings
from ..utils.logger import get_logger, log_api_call

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ConversationSummary:
    """Resumen comprimido de conversación."""
    user_id: str
    session_id: str
    summary: str
    key_points: List[str]
    last_intent: Optional[str] = None
    created_at: datetime = None
    message_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class RedisConnectionManager:
    """
    Manager para conexiones Redis con pooling y health monitoring.
    
    Maneja reconexiones automáticas y monitoring de salud de la conexión.
    """
    
    def __init__(self):
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        self.last_health_check = datetime.now()
        self.connection_attempts = 0
        self.max_connection_attempts = 5
    
    async def initialize(self) -> bool:
        """
        Inicializa conexión Redis con pooling.
        
        Returns:
            True si conexión exitosa
        """
        try:
            # Crear pool de conexiones
            self.pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30
            )
            
            self.client = redis.Redis(connection_pool=self.pool, decode_responses=True)
            
            # Test conectividad
            await self.client.ping()
            
            self.is_connected = True
            self.connection_attempts = 0
            
            logger.info("✅ Redis connection establecida")
            return True
            
        except Exception as e:
            self.is_connected = False
            self.connection_attempts += 1
            
            logger.error(f"❌ Error conectando a Redis: {e}")
            return False
    
    async def ensure_connected(self) -> bool:
        """
        Asegura que la conexión esté activa.
        
        Returns:
            True si conexión está disponible
        """
        if not self.is_connected or not self.client:
            return await self.initialize()
        
        # Health check periódico
        now = datetime.now()
        if (now - self.last_health_check).seconds > 30:
            try:
                await self.client.ping()
                self.last_health_check = now
                return True
            except Exception:
                self.is_connected = False
                return await self.initialize()
        
        return True
    
    async def close(self):
        """Cierra conexiones Redis."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        
        self.is_connected = False
        logger.info("🔌 Redis connection cerrada")


class ContextService:
    """
    Servicio principal para gestión de contexto conversacional.
    
    Features:
    - Redis-backed session storage con TTL
    - Compresión automática de historial largo
    - Multi-user session isolation
    - Cleanup automático de sesiones expiradas
    - Context compression para evitar token limits
    - Performance monitoring
    """
    
    def __init__(self):
        self.redis_manager = RedisConnectionManager()
        self.compression_threshold = settings.MAX_MEMORY_MESSAGES
        self.max_context_tokens = settings.MAX_CONTEXT_TOKENS
        
        # Cache local para reducir calls a Redis
        self._local_cache: Dict[str, ConversationContext] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._cache_max_age = timedelta(minutes=5)  # Cache 5 minutos
        
        # Estadísticas
        self.stats = {
            "contexts_loaded": 0,
            "contexts_saved": 0,
            "compressions_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "redis_errors": 0
        }
    
    async def initialize(self) -> bool:
        """
        Inicializa el servicio.
        
        Returns:
            True si inicialización exitosa
        """
        success = await self.redis_manager.initialize()
        if success:
            # Iniciar background task para cleanup
            asyncio.create_task(self._periodic_cleanup())
        return success
    
    def _get_context_key(self, user_id: str, session_id: str = "default") -> str:
        """Generate unique Redis key para contexto."""
        return f"whatsapp:context:{user_id}:{session_id}"
    
    def _get_history_key(self, user_id: str, session_id: str = "default") -> str:
        """Generate unique Redis key para historial."""
        return f"whatsapp:history:{user_id}:{session_id}"
    
    def _get_summary_key(self, user_id: str, session_id: str = "default") -> str:
        """Generate unique Redis key para resumen.""" 
        return f"whatsapp:summary:{user_id}:{session_id}"
    
    async def get_context(
        self, 
        user_id: str, 
        session_id: str = "default"
    ) -> Optional[ConversationContext]:
        """
        Obtiene contexto conversacional para usuario.
        
        Args:
            user_id: ID del usuario WhatsApp
            session_id: ID de sesión (default para conversación única)
            
        Returns:
            ConversationContext o None si no existe
        """
        
        cache_key = f"{user_id}:{session_id}"
        
        # Check cache local primero
        if self._is_cache_valid(cache_key):
            self.stats["cache_hits"] += 1
            return self._local_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Asegurar conexión Redis
        if not await self.redis_manager.ensure_connected():
            logger.error("❌ Redis no disponible para obtener contexto")
            return None
        
        try:
            context_key = self._get_context_key(user_id, session_id)
            
            # Obtener datos de Redis
            context_data = await self.redis_manager.client.get(context_key)
            
            if not context_data:
                logger.debug(f"📭 No hay contexto para {user_id}")
                return None
            
            # Deserializar contexto
            context_dict = json.loads(context_data)
            context = ConversationContext(**context_dict)
            
            # Actualizar cache local
            self._update_local_cache(cache_key, context)
            
            # Cargar resumen si existe
            await self._load_conversation_summary(user_id, session_id, context)
            
            self.stats["contexts_loaded"] += 1
            logger.debug(f"📥 Contexto cargado para {user_id}")
            
            return context
            
        except Exception as e:
            self.stats["redis_errors"] += 1
            logger.error(f"❌ Error obteniendo contexto para {user_id}: {e}")
            return None
    
    async def save_context(
        self,
        context: ConversationContext,
        session_id: str = "default"
    ) -> bool:
        """
        Guarda contexto conversacional en Redis.
        
        Args:
            context: Contexto a guardar
            session_id: ID de sesión
            
        Returns:
            True si guardado exitosamente
        """
        
        # Asegurar conexión Redis
        if not await self.redis_manager.ensure_connected():
            logger.error("❌ Redis no disponible para guardar contexto")
            return False
        
        try:
            context_key = self._get_context_key(context.user_id, session_id)
            
            # Actualizar timestamp
            context.last_activity = datetime.now()
            
            # Serializar contexto
            context_data = context.model_dump_json()
            
            # Guardar con TTL
            await self.redis_manager.client.setex(
                context_key,
                settings.REDIS_SESSION_TTL,
                context_data
            )
            
            # Actualizar cache local
            cache_key = f"{context.user_id}:{session_id}"
            self._update_local_cache(cache_key, context)
            
            self.stats["contexts_saved"] += 1
            logger.debug(f"💾 Contexto guardado para {context.user_id}")
            
            return True
            
        except Exception as e:
            self.stats["redis_errors"] += 1
            logger.error(f"❌ Error guardando contexto para {context.user_id}: {e}")
            return False
    
    async def add_message_to_history(
        self,
        user_id: str,
        message: WhatsAppMessage,
        session_id: str = "default"
    ) -> bool:
        """
        Añade mensaje al historial conversacional.
        
        Args:
            user_id: ID del usuario
            message: Mensaje a añadir
            session_id: ID de sesión
            
        Returns:
            True si añadido exitosamente
        """
        
        try:
            # Obtener o crear contexto
            context = await self.get_context(user_id, session_id)
            if not context:
                context = ConversationContext(
                    user_id=user_id,
                    conversation_id=f"{user_id}_{int(time.time())}"
                )
            
            # Añadir mensaje al historial
            message_text = f"{'Usuario' if not message.is_from_me else 'Asistente'}: {message.body}"
            context.message_history.append(message_text)
            
            # Check si necesita compresión
            if len(context.message_history) > self.compression_threshold:
                await self._compress_conversation_history(context, session_id)
            
            # Guardar contexto actualizado
            return await self.save_context(context, session_id)
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo mensaje al historial para {user_id}: {e}")
            return False
    
    async def update_intent(
        self,
        user_id: str,
        intent: MessageIntent,
        session_id: str = "default"
    ) -> bool:
        """
        Actualiza intención actual en el contexto.
        
        Args:
            user_id: ID del usuario
            intent: Nueva intención detectada
            session_id: ID de sesión
            
        Returns:
            True si actualizado exitosamente
        """
        
        try:
            context = await self.get_context(user_id, session_id)
            if not context:
                # Crear nuevo contexto con la intención
                context = ConversationContext(
                    user_id=user_id,
                    conversation_id=f"{user_id}_{int(time.time())}",
                    current_intent=intent
                )
            else:
                context.current_intent = intent
            
            return await self.save_context(context, session_id)
            
        except Exception as e:
            logger.error(f"❌ Error actualizando intención para {user_id}: {e}")
            return False
    
    async def get_conversation_summary(
        self,
        user_id: str,
        session_id: str = "default",
        max_messages: int = 10
    ) -> str:
        """
        Obtiene resumen formateado de la conversación para contexto LLM.
        
        Args:
            user_id: ID del usuario
            session_id: ID de sesión
            max_messages: Número máximo de mensajes a incluir
            
        Returns:
            String formateado para usar como contexto
        """
        
        try:
            context = await self.get_context(user_id, session_id)
            if not context:
                return "No hay historial de conversación previo."
            
            # Verificar si hay resumen comprimido
            summary_key = self._get_summary_key(user_id, session_id)
            
            if await self.redis_manager.ensure_connected():
                summary_data = await self.redis_manager.client.get(summary_key)
                
                if summary_data:
                    summary = ConversationSummary(**json.loads(summary_data))
                    context_lines = [f"[RESUMEN PREVIO]: {summary.summary}"]
                    
                    if summary.key_points:
                        context_lines.append("Puntos clave:")
                        for point in summary.key_points:
                            context_lines.append(f"- {point}")
                    
                    context_lines.append("---")
            
            # Añadir mensajes recientes
            recent_messages = context.message_history[-max_messages:] if context.message_history else []
            context_lines.extend(recent_messages)
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo resumen para {user_id}: {e}")
            return "Error obteniendo historial de conversación."
    
    async def _load_conversation_summary(
        self,
        user_id: str,
        session_id: str,
        context: ConversationContext
    ):
        """Carga resumen de conversación si existe."""
        
        try:
            if not await self.redis_manager.ensure_connected():
                return
            
            summary_key = self._get_summary_key(user_id, session_id)
            summary_data = await self.redis_manager.client.get(summary_key)
            
            if summary_data:
                summary = ConversationSummary(**json.loads(summary_data))
                
                # Si el resumen es reciente (< 4 horas), añadirlo al contexto
                if summary.created_at > datetime.now() - timedelta(hours=4):
                    # Añadir resumen al inicio del historial
                    summary_text = f"[RESUMEN]: {summary.summary}"
                    if summary_text not in context.message_history:
                        context.message_history.insert(0, summary_text)
                    
                    logger.debug(f"📋 Resumen cargado para {user_id}")
                    
        except Exception as e:
            logger.error(f"❌ Error cargando resumen para {user_id}: {e}")
    
    async def _compress_conversation_history(
        self,
        context: ConversationContext,
        session_id: str = "default"
    ):
        """
        Comprime historial de conversación cuando es muy largo.
        
        Mantiene mensajes recientes y crea resumen del resto.
        """
        
        try:
            messages_to_compress = context.message_history[:-8]  # Comprimir todo excepto últimos 8
            messages_to_keep = context.message_history[-8:]  # Mantener últimos 8
            
            if len(messages_to_compress) < 5:  # No comprimir si son muy pocos
                return
            
            # Crear texto para compresión
            conversation_text = "\n".join(messages_to_compress)
            
            # TODO: Integrar con LLM service para generar resumen inteligente
            # Por ahora usar compresión simple
            summary_text = self._create_simple_summary(messages_to_compress)
            
            # Extraer puntos clave
            key_points = self._extract_key_points(messages_to_compress)
            
            # Crear objeto de resumen
            summary = ConversationSummary(
                user_id=context.user_id,
                session_id=session_id,
                summary=summary_text,
                key_points=key_points,
                message_count=len(messages_to_compress),
                created_at=datetime.now()
            )
            
            # Guardar resumen en Redis
            await self._save_conversation_summary(summary)
            
            # Actualizar contexto con mensajes comprimidos
            context.message_history = [f"[RESUMEN]: {summary_text}"] + messages_to_keep
            
            self.stats["compressions_performed"] += 1
            logger.info(f"🗜️ Historial comprimido para {context.user_id} ({len(messages_to_compress)} mensajes)")
            
        except Exception as e:
            logger.error(f"❌ Error comprimiendo historial para {context.user_id}: {e}")
    
    def _create_simple_summary(self, messages: List[str]) -> str:
        """Crea resumen simple basado en patterns comunes."""
        
        # Contar tipos de mensaje
        user_messages = [msg for msg in messages if msg.startswith("Usuario:")]
        assistant_messages = [msg for msg in messages if msg.startswith("Asistente:")]
        
        # Detectar temas comunes
        all_text = " ".join(messages).lower()
        topics = []
        
        if any(word in all_text for word in ["precio", "cuesta", "comprar"]):
            topics.append("consultas de precios")
        if any(word in all_text for word in ["problema", "error", "no funciona"]):
            topics.append("problemas técnicos")
        if any(word in all_text for word in ["factura", "pago", "cobro"]):
            topics.append("facturación")
        
        # Construir resumen
        summary_parts = [
            f"Conversación de {len(user_messages)} mensajes del usuario",
            f"y {len(assistant_messages)} respuestas"
        ]
        
        if topics:
            summary_parts.append(f"sobre {', '.join(topics)}")
        
        return ". ".join(summary_parts) + "."
    
    def _extract_key_points(self, messages: List[str]) -> List[str]:
        """Extrae puntos clave de mensajes."""
        
        key_points = []
        
        # Buscar números importantes (teléfonos, precios, códigos)
        import re
        for message in messages[-5:]:  # Solo últimos 5 mensajes
            # Números de teléfono
            phones = re.findall(r'\b\d{10,}\b', message)
            for phone in phones:
                if len(phone) >= 10:
                    key_points.append(f"Número mencionado: {phone}")
            
            # Precios
            prices = re.findall(r'\$[\d,]+', message)
            for price in prices:
                key_points.append(f"Precio mencionado: {price}")
        
        return key_points[:5]  # Máximo 5 puntos clave
    
    async def _save_conversation_summary(self, summary: ConversationSummary):
        """Guarda resumen de conversación en Redis."""
        
        try:
            if not await self.redis_manager.ensure_connected():
                return
            
            summary_key = self._get_summary_key(summary.user_id, summary.session_id)
            summary_data = json.dumps(summary.__dict__, default=str)
            
            # Guardar con TTL
            await self.redis_manager.client.setex(
                summary_key,
                settings.REDIS_SESSION_TTL,
                summary_data
            )
            
        except Exception as e:
            logger.error(f"❌ Error guardando resumen: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica si entrada de cache es válida."""
        
        if cache_key not in self._local_cache:
            return False
        
        if cache_key not in self._cache_ttl:
            return False
        
        return datetime.now() - self._cache_ttl[cache_key] < self._cache_max_age
    
    def _update_local_cache(self, cache_key: str, context: ConversationContext):
        """Actualiza cache local."""
        
        self._local_cache[cache_key] = context
        self._cache_ttl[cache_key] = datetime.now()
    
    async def clear_session(
        self,
        user_id: str,
        session_id: str = "default"
    ) -> bool:
        """
        Limpia sesión completa para usuario.
        
        Args:
            user_id: ID del usuario
            session_id: ID de sesión
            
        Returns:
            True si limpiado exitosamente
        """
        
        try:
            if not await self.redis_manager.ensure_connected():
                return False
            
            # Limpiar todas las keys relacionadas
            keys_to_delete = [
                self._get_context_key(user_id, session_id),
                self._get_history_key(user_id, session_id),
                self._get_summary_key(user_id, session_id)
            ]
            
            for key in keys_to_delete:
                await self.redis_manager.client.delete(key)
            
            # Limpiar cache local
            cache_key = f"{user_id}:{session_id}"
            self._local_cache.pop(cache_key, None)
            self._cache_ttl.pop(cache_key, None)
            
            logger.info(f"🗑️ Sesión limpiada para {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error limpiando sesión para {user_id}: {e}")
            return False
    
    async def _periodic_cleanup(self):
        """Background task para cleanup periódico."""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Cada hora
                
                if await self.redis_manager.ensure_connected():
                    # Limpiar keys expiradas (Redis lo hace automáticamente pero esto es extra)
                    pattern = "whatsapp:*"
                    
                    async for key in self.redis_manager.client.scan_iter(match=pattern):
                        ttl = await self.redis_manager.client.ttl(key)
                        if ttl == -1:  # No TTL set
                            await self.redis_manager.client.expire(key, settings.REDIS_SESSION_TTL)
                
                # Limpiar cache local expirado
                now = datetime.now()
                expired_keys = [
                    key for key, timestamp in self._cache_ttl.items()
                    if now - timestamp > self._cache_max_age
                ]
                
                for key in expired_keys:
                    self._local_cache.pop(key, None)
                    self._cache_ttl.pop(key, None)
                
                if expired_keys:
                    logger.debug(f"🧹 {len(expired_keys)} entradas de cache local limpiadas")
                    
            except Exception as e:
                logger.error(f"❌ Error en cleanup periódico: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servicio."""
        
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total_cache_requests) * 100
        
        return {
            **self.stats,
            "cache_hit_rate_percent": cache_hit_rate,
            "local_cache_size": len(self._local_cache),
            "redis_connected": self.redis_manager.is_connected,
            "connection_attempts": self.redis_manager.connection_attempts
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica salud del servicio.
        
        Returns:
            Estado de salud y métricas
        """
        
        try:
            start_time = time.time()
            
            # Test Redis connectivity
            if await self.redis_manager.ensure_connected():
                await self.redis_manager.client.ping()
                response_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "healthy",
                    "redis_connected": True,
                    "response_time_ms": response_time,
                    "stats": self.get_stats()
                }
            else:
                return {
                    "status": "unhealthy",
                    "redis_connected": False,
                    "error": "No se pudo conectar a Redis",
                    "stats": self.get_stats()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy", 
                "redis_connected": False,
                "error": str(e),
                "stats": self.get_stats()
            }
    
    async def close(self):
        """Cierra el servicio y conexiones."""
        await self.redis_manager.close()
        logger.info("🔌 Context Service cerrado")