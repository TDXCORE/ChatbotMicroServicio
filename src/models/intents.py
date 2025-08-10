"""
Modelos Pydantic para clasificación de intenciones.

Este módulo define los tipos de intención, estructuras de clasificación
y contexto conversacional para el sistema de chatbot WhatsApp.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """
    Tipos de intención soportados por el sistema.
    
    Cada intención mapea a un departamento específico para enrutamiento.
    """
    SALES = "ventas"              # Consultas de precios, productos, cotizaciones
    SUPPORT = "soporte"           # Problemas técnicos, ayuda, averías
    BILLING = "facturacion"       # Facturación, pagos, cobros, cuentas
    GENERAL = "general"           # Información general, horarios, ubicación
    COMPLAINT = "reclamo"         # Quejas, reclamos, insatisfacción
    INFORMATION = "informacion"   # Información sobre servicios, empresa
    UNKNOWN = "desconocido"       # Mensajes no clasificables


class MessageIntent(BaseModel):
    """
    Resultado de clasificación de intención para un mensaje.
    
    Contiene la intención detectada, nivel de confianza y entidades extraídas.
    """
    intent: IntentType = Field(description="Tipo de intención clasificada")
    
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Nivel de confianza de la clasificación (0.0-1.0)"
    )
    
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entidades extraídas del mensaje (productos, números, fechas, etc.)"
    )
    
    routing_department: str = Field(description="Email del departamento destino")
    
    priority: int = Field(
        ge=1, 
        le=5, 
        default=3,
        description="Prioridad del mensaje (1=baja, 5=crítica)"
    )
    
    reasoning: str = Field(
        default="",
        description="Explicación breve de por qué se clasificó así"
    )
    
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de cuando se procesó la clasificación"
    )


class ConversationContext(BaseModel):
    """
    Contexto de una conversación WhatsApp en curso.
    
    Mantiene historial, estado actual e información de sesión.
    """
    user_id: str = Field(description="Identificador único del usuario (número WhatsApp)")
    
    conversation_id: str = Field(description="ID único de la conversación")
    
    message_history: List[str] = Field(
        default_factory=list,
        max_items=50,  # Límite para evitar exceso de memoria
        description="Historial de mensajes recientes"
    )
    
    current_intent: Optional[MessageIntent] = Field(
        default=None,
        description="Última intención detectada en la conversación"
    )
    
    session_start: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de inicio de la sesión"
    )
    
    last_activity: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de última actividad"
    )
    
    is_active: bool = Field(
        default=True,
        description="Si la conversación está activa"
    )
    
    language: str = Field(
        default="es",
        description="Idioma detectado de la conversación"
    )
    
    department_transferred: Optional[str] = Field(
        default=None,
        description="Departamento al que se transfirió (si aplica)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos adicionales de la conversación"
    )


class IntentClassificationRequest(BaseModel):
    """
    Request para clasificación de intención de un mensaje.
    
    Incluye el mensaje y contexto necesario para clasificación precisa.
    """
    message: str = Field(
        min_length=1,
        max_length=2000,  # Límite realista para mensajes WhatsApp
        description="Texto del mensaje a clasificar"
    )
    
    user_id: str = Field(description="ID del usuario que envió el mensaje")
    
    context: Optional[ConversationContext] = Field(
        default=None,
        description="Contexto conversacional previo (opcional)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp del mensaje"
    )


class IntentClassificationResponse(BaseModel):
    """
    Respuesta de clasificación de intención.
    
    Contiene la clasificación y información adicional para procesamiento.
    """
    message_intent: MessageIntent = Field(description="Resultado de clasificación")
    
    context_updated: ConversationContext = Field(description="Contexto actualizado")
    
    processing_time_ms: float = Field(
        ge=0.0,
        description="Tiempo de procesamiento en milisegundos"
    )
    
    model_used: str = Field(
        default="gpt-4-1106-preview",
        description="Modelo usado para clasificación"
    )
    
    fallback_used: bool = Field(
        default=False,
        description="Si se usó lógica de fallback (no LLM)"
    )


class DepartmentConfig(BaseModel):
    """
    Configuración de un departamento para enrutamiento.
    
    Define cómo se maneja el enrutamiento a cada departamento.
    """
    name: str = Field(description="Nombre del departamento")
    
    email: str = Field(description="Email de contacto del departamento")
    
    intent_types: List[IntentType] = Field(description="Tipos de intención que maneja")
    
    priority_threshold: int = Field(
        ge=1,
        le=5,
        default=3,
        description="Threshold de prioridad para escalamiento automático"
    )
    
    auto_response_enabled: bool = Field(
        default=True,
        description="Si debe enviar respuesta automática inicial"
    )
    
    auto_response_template: str = Field(
        default="Gracias por contactarnos. Tu mensaje ha sido enviado al departamento correspondiente.",
        description="Template para respuesta automática"
    )
    
    working_hours: Dict[str, Any] = Field(
        default_factory=dict,
        description="Horarios de atención del departamento"
    )
    
    escalation_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reglas de escalamiento para casos complejos"
    )


# Configuraciones predefinidas de departamentos
DEFAULT_DEPARTMENT_CONFIGS = {
    IntentType.SALES: DepartmentConfig(
        name="Ventas",
        email="ventas@empresa.com",
        intent_types=[IntentType.SALES],
        priority_threshold=3,
        auto_response_template="¡Hola! Tu consulta de ventas ha sido recibida. Un ejecutivo te contactará pronto.",
        working_hours={"weekdays": "9:00-18:00", "weekends": "10:00-14:00"}
    ),
    
    IntentType.SUPPORT: DepartmentConfig(
        name="Soporte Técnico",
        email="soporte@empresa.com",
        intent_types=[IntentType.SUPPORT],
        priority_threshold=4,  # Soporte tiene mayor prioridad
        auto_response_template="Tu solicitud de soporte ha sido registrada. Te ayudaremos lo antes posible.",
        working_hours={"weekdays": "8:00-20:00", "weekends": "closed"}
    ),
    
    IntentType.BILLING: DepartmentConfig(
        name="Facturación",
        email="facturacion@empresa.com", 
        intent_types=[IntentType.BILLING],
        priority_threshold=3,
        auto_response_template="Tu consulta de facturación ha sido recibida. Te responderemos dentro de 24 horas.",
        working_hours={"weekdays": "9:00-17:00", "weekends": "closed"}
    ),
    
    IntentType.GENERAL: DepartmentConfig(
        name="Información General",
        email="info@empresa.com",
        intent_types=[IntentType.GENERAL, IntentType.INFORMATION],
        priority_threshold=2,
        auto_response_template="Gracias por tu consulta. Te proporcionaremos la información solicitada pronto.",
        working_hours={"weekdays": "9:00-18:00", "weekends": "10:00-16:00"}
    ),
    
    IntentType.COMPLAINT: DepartmentConfig(
        name="Gerencia - Reclamos",
        email="gerencia@empresa.com",
        intent_types=[IntentType.COMPLAINT],
        priority_threshold=5,  # Máxima prioridad para reclamos
        auto_response_template="Tu reclamo es importante para nosotros. Será atendido por gerencia dentro de 4 horas.",
        working_hours={"weekdays": "8:00-19:00", "weekends": "emergency_only"}
    )
}