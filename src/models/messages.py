"""
Modelos Pydantic para mensajes de WhatsApp y historial conversacional.

Define las estructuras de datos para mensajes entrantes, salientes
y manejo del historial de conversaciones.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class MessageType(str, Enum):
    """
    Tipos de mensaje soportados por WhatsApp Web.js.
    
    Basado en la documentación oficial de whatsapp-web.js.
    """
    TEXT = "text"                # Mensaje de texto regular  
    IMAGE = "image"              # Imagen con opcional caption
    VIDEO = "video"              # Video con opcional caption
    AUDIO = "audio"              # Mensaje de voz
    DOCUMENT = "document"        # Documento/archivo
    STICKER = "sticker"          # Sticker/emoji animado
    LOCATION = "location"        # Ubicación compartida
    CONTACT = "contact"          # Contacto compartido
    UNKNOWN = "unknown"          # Tipo no reconocido


class MessageStatus(str, Enum):
    """
    Estados de procesamiento de mensajes.
    
    Rastrea el lifecycle completo del mensaje en el sistema.
    """
    RECEIVED = "received"        # Mensaje recibido desde WhatsApp
    PROCESSING = "processing"    # En proceso de clasificación
    CLASSIFIED = "classified"    # Intención clasificada exitosamente  
    ROUTED = "routed"           # Enrutado a departamento
    RESPONDED = "responded"      # Respuesta automática enviada
    ESCALATED = "escalated"      # Escalado a humano
    FAILED = "failed"           # Falló en algún punto del proceso
    IGNORED = "ignored"         # Mensaje ignorado (spam, etc.)


class WhatsAppMessage(BaseModel):
    """
    Mensaje de WhatsApp recibido o enviado.
    
    Estructura completa que incluye metadata y estado de procesamiento.
    """
    id: str = Field(description="ID único del mensaje (desde WhatsApp Web.js)")
    
    from_number: str = Field(
        description="Número de teléfono del remitente (formato: 1234567890@c.us)"
    )
    
    to_number: str = Field(
        description="Número de teléfono destino (nuestro bot)"
    )
    
    body: str = Field(
        default="",
        description="Contenido de texto del mensaje"
    )
    
    message_type: MessageType = Field(
        default=MessageType.TEXT,
        description="Tipo de mensaje"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp del mensaje"
    )
    
    is_from_me: bool = Field(
        default=False,
        description="Si el mensaje fue enviado por nuestro bot"
    )
    
    status: MessageStatus = Field(
        default=MessageStatus.RECEIVED,
        description="Estado actual de procesamiento"
    )
    
    media_url: Optional[str] = Field(
        default=None,
        description="URL del archivo multimedia (si aplica)"
    )
    
    media_mime_type: Optional[str] = Field(
        default=None,
        description="MIME type del archivo multimedia"
    )
    
    location_latitude: Optional[float] = Field(
        default=None,
        description="Latitud (para mensajes de ubicación)"
    )
    
    location_longitude: Optional[float] = Field(
        default=None,
        description="Longitud (para mensajes de ubicación)"
    )
    
    contact_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Información de contacto (para mensajes de contacto)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata adicional del mensaje"
    )
    
    processing_errors: List[str] = Field(
        default_factory=list,
        description="Lista de errores encontrados durante procesamiento"
    )
    
    @field_validator('from_number', 'to_number')
    @classmethod
    def validate_whatsapp_number(cls, v):
        """Valida formato de número WhatsApp."""
        if not v.endswith('@c.us') and not v.endswith('@g.us'):
            # Auto-corregir formato si es necesario
            if '@' not in v:
                return f"{v}@c.us"  # Asume chat individual por defecto
        return v
    
    @field_validator('body')
    @classmethod
    def validate_body_length(cls, v):
        """Valida longitud del mensaje."""
        if len(v) > 4096:  # Límite realista para WhatsApp
            raise ValueError("Mensaje demasiado largo")
        return v


class MessageHistory(BaseModel):
    """
    Historial de mensajes para una conversación.
    
    Mantiene registro completo con compresión automática para evitar
    exceder límites de memoria y tokens.
    """
    user_id: str = Field(description="ID del usuario (número WhatsApp)")
    
    conversation_id: str = Field(description="ID único de la conversación")
    
    messages: List[WhatsAppMessage] = Field(
        default_factory=list,
        description="Lista de mensajes en orden cronológico"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de creación del historial"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Última actualización del historial"
    )
    
    total_messages: int = Field(
        default=0,
        description="Contador total de mensajes (incluye comprimidos)"
    )
    
    compressed_summary: Optional[str] = Field(
        default=None,
        description="Resumen de mensajes comprimidos más antiguos"
    )
    
    is_compressed: bool = Field(
        default=False,
        description="Si el historial ha sido comprimido"
    )
    
    compression_threshold: int = Field(
        default=30,
        description="Número de mensajes antes de comprimir"
    )
    
    def add_message(self, message: WhatsAppMessage) -> None:
        """
        Añade un mensaje al historial.
        
        Args:
            message: Mensaje a añadir
        """
        self.messages.append(message)
        self.total_messages += 1
        self.last_updated = datetime.now()
        
        # Trigger compresión si es necesario
        if len(self.messages) > self.compression_threshold:
            self._trigger_compression_needed()
    
    def _trigger_compression_needed(self) -> None:
        """Marca que se necesita compresión (será procesada por ContextService)."""
        self.metadata = getattr(self, 'metadata', {})
        self.metadata['needs_compression'] = True
    
    def get_recent_messages(self, limit: int = 10) -> List[WhatsAppMessage]:
        """
        Obtiene los mensajes más recientes.
        
        Args:
            limit: Número máximo de mensajes a retornar
            
        Returns:
            Lista de mensajes recientes
        """
        return self.messages[-limit:] if self.messages else []
    
    def get_messages_by_type(self, message_type: MessageType) -> List[WhatsAppMessage]:
        """
        Filtra mensajes por tipo.
        
        Args:
            message_type: Tipo de mensaje a filtrar
            
        Returns:
            Lista de mensajes del tipo especificado
        """
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_text_content_for_context(self, limit: int = 20) -> str:
        """
        Genera contenido de texto para contexto LLM.
        
        Args:
            limit: Número máximo de mensajes a incluir
            
        Returns:
            String formateado para usar como contexto
        """
        recent_messages = self.get_recent_messages(limit)
        text_lines = []
        
        # Incluir resumen si existe
        if self.compressed_summary:
            text_lines.append(f"[RESUMEN PREVIO]: {self.compressed_summary}")
            text_lines.append("---")
        
        # Incluir mensajes recientes
        for msg in recent_messages:
            if msg.message_type == MessageType.TEXT and msg.body.strip():
                role = "Asistente" if msg.is_from_me else "Usuario"
                text_lines.append(f"{role}: {msg.body}")
        
        return "\n".join(text_lines)


class OutgoingMessage(BaseModel):
    """
    Mensaje que será enviado por el bot a WhatsApp.
    
    Incluye configuración para delivery y tracking.
    """
    to_number: str = Field(description="Número destino (formato WhatsApp)")
    
    body: str = Field(description="Contenido del mensaje")
    
    message_type: MessageType = Field(
        default=MessageType.TEXT,
        description="Tipo de mensaje a enviar"
    )
    
    priority: int = Field(
        ge=1,
        le=5, 
        default=3,
        description="Prioridad de envío (1=baja, 5=urgente)"
    )
    
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp para envío programado (None = inmediato)"
    )
    
    retry_attempts: int = Field(
        default=0,
        description="Número de intentos de reenvío realizados"
    )
    
    max_retries: int = Field(
        default=3,
        description="Número máximo de intentos de reenvío"
    )
    
    media_path: Optional[str] = Field(
        default=None,
        description="Path local del archivo multimedia a enviar"
    )
    
    reply_to_message_id: Optional[str] = Field(
        default=None,
        description="ID del mensaje al que responde (para threading)"
    )
    
    template_name: Optional[str] = Field(
        default=None,
        description="Nombre del template usado para generar el mensaje"
    )
    
    context_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Datos de contexto usados para el mensaje"
    )
    
    tracking_enabled: bool = Field(
        default=True,
        description="Si debe trackear delivery y lectura"
    )


class MessageQueue(BaseModel):
    """
    Cola de mensajes pendientes de procesar o enviar.
    
    Maneja priorización y rate limiting automático.
    """
    queue_id: str = Field(description="ID único de la cola")
    
    incoming_messages: List[WhatsAppMessage] = Field(
        default_factory=list,
        description="Mensajes entrantes pendientes de procesar"
    )
    
    outgoing_messages: List[OutgoingMessage] = Field(
        default_factory=list,
        description="Mensajes pendientes de envío"
    )
    
    processing_active: bool = Field(
        default=True,
        description="Si la cola está procesando activamente"
    )
    
    last_send_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp del último mensaje enviado (para rate limiting)"
    )
    
    rate_limit_delay: float = Field(
        default=1.5,
        description="Delay mínimo entre envíos (segundos)"
    )
    
    failed_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Mensajes que fallaron en procesamiento"
    )
    
    def add_incoming(self, message: WhatsAppMessage) -> None:
        """Añade mensaje entrante a la cola."""
        self.incoming_messages.append(message)
    
    def add_outgoing(self, message: OutgoingMessage) -> None:
        """Añade mensaje saliente a la cola con ordenamiento por prioridad."""
        self.outgoing_messages.append(message)
        # Reordenar por prioridad (mayor prioridad primero)
        self.outgoing_messages.sort(key=lambda x: x.priority, reverse=True)
    
    def get_next_outgoing(self) -> Optional[OutgoingMessage]:
        """
        Obtiene el próximo mensaje a enviar respetando rate limiting.
        
        Returns:
            Mensaje a enviar o None si debe esperar por rate limiting
        """
        if not self.outgoing_messages:
            return None
        
        # Verificar rate limiting
        if self.last_send_time:
            time_since_last = (datetime.now() - self.last_send_time).total_seconds()
            if time_since_last < self.rate_limit_delay:
                return None  # Debe esperar más tiempo
        
        return self.outgoing_messages.pop(0)
    
    def mark_message_sent(self) -> None:
        """Marca timestamp del último envío para rate limiting."""
        self.last_send_time = datetime.now()