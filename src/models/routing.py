"""
Modelos Pydantic para decisiones de enrutamiento y gestión de departamentos.

Define las estructuras para routing decisions, configuración de departamentos
y reglas de escalamiento automático.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, time
from enum import Enum
from pydantic import BaseModel, Field

from .intents import IntentType, MessageIntent


class RoutingAction(str, Enum):
    """
    Acciones disponibles para el routing de mensajes.
    
    Define qué hacer con un mensaje después de clasificar su intención.
    """
    ROUTE_TO_DEPARTMENT = "route_to_department"      # Enviar a departamento específico
    AUTO_RESPOND = "auto_respond"                    # Respuesta automática 
    ESCALATE_TO_HUMAN = "escalate_to_human"          # Escalar a agente humano
    REQUEST_MORE_INFO = "request_more_info"          # Solicitar información adicional
    TRANSFER_CONVERSATION = "transfer_conversation"   # Transferir conversación completa
    QUEUE_FOR_LATER = "queue_for_later"             # Encolar para procesamiento posterior
    IGNORE_MESSAGE = "ignore_message"                # Ignorar (spam, irrelevante)


class RoutingDecision(BaseModel):
    """
    Decisión de enrutamiento tomada por el sistema.
    
    Contiene la acción a tomar y toda la información necesaria para ejecutarla.
    """
    message_id: str = Field(description="ID del mensaje que se está enrutando")
    
    user_id: str = Field(description="ID del usuario que envió el mensaje")
    
    detected_intent: MessageIntent = Field(description="Intención detectada del mensaje")
    
    action: RoutingAction = Field(description="Acción a tomar")
    
    target_department: Optional[str] = Field(
        default=None,
        description="Email del departamento destino (si aplica)"
    )
    
    assigned_agent: Optional[str] = Field(
        default=None,
        description="ID del agente humano asignado (si aplica)"
    )
    
    auto_response_message: Optional[str] = Field(
        default=None,
        description="Mensaje de respuesta automática a enviar"
    )
    
    priority: int = Field(
        ge=1,
        le=5,
        default=3,
        description="Prioridad de la decisión (1=baja, 5=crítica)"
    )
    
    estimated_resolution_time: Optional[int] = Field(
        default=None,
        description="Tiempo estimado de resolución en minutos"
    )
    
    requires_immediate_attention: bool = Field(
        default=False,
        description="Si requiere atención inmediata"
    )
    
    routing_reason: str = Field(
        default="",
        description="Explicación de por qué se tomó esta decisión"
    )
    
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confianza en la decisión de routing"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de la decisión"
    )
    
    additional_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contexto adicional para la decisión"
    )


class DepartmentAvailability(BaseModel):
    """
    Estado de disponibilidad de un departamento.
    
    Maneja horarios de atención, carga de trabajo y disponibilidad en tiempo real.
    """
    department_email: str = Field(description="Email identificador del departamento")
    
    is_online: bool = Field(
        default=True,
        description="Si el departamento está disponible actualmente"
    )
    
    current_workload: int = Field(
        ge=0,
        default=0,
        description="Número actual de casos asignados"
    )
    
    max_capacity: int = Field(
        ge=1,
        default=50,
        description="Capacidad máxima de casos simultáneos"
    )
    
    average_response_time: float = Field(
        ge=0.0,
        default=30.0,
        description="Tiempo promedio de respuesta en minutos"
    )
    
    working_hours_start: time = Field(
        default=time(9, 0),
        description="Hora de inicio de atención"
    )
    
    working_hours_end: time = Field(
        default=time(18, 0), 
        description="Hora de fin de atención"
    )
    
    working_days: List[int] = Field(
        default=[0, 1, 2, 3, 4],  # Lunes a viernes
        description="Días de la semana que trabaja (0=lunes, 6=domingo)"
    )
    
    timezone: str = Field(
        default="America/Mexico_City",
        description="Zona horaria del departamento"
    )
    
    emergency_contact: Optional[str] = Field(
        default=None,
        description="Contacto de emergencia fuera de horario"
    )
    
    auto_responder_enabled: bool = Field(
        default=True,
        description="Si tiene auto-responder activado"
    )
    
    last_status_update: datetime = Field(
        default_factory=datetime.now,
        description="Última actualización de estado"
    )
    
    @property
    def is_available_now(self) -> bool:
        """
        Verifica si el departamento está disponible en este momento.
        
        Returns:
            True si está disponible, False si no
        """
        if not self.is_online:
            return False
        
        # Verificar capacidad
        if self.current_workload >= self.max_capacity:
            return False
        
        # Verificar horarios (simplificado - no considera timezone aún)
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()
        
        if current_day not in self.working_days:
            return False
        
        if not (self.working_hours_start <= current_time <= self.working_hours_end):
            return False
        
        return True
    
    @property
    def workload_percentage(self) -> float:
        """Porcentaje actual de carga de trabajo."""
        return (self.current_workload / self.max_capacity) * 100


class RoutingRule(BaseModel):
    """
    Regla de enrutamiento configurable.
    
    Define cómo se deben enrutar mensajes basado en condiciones específicas.
    """
    rule_id: str = Field(description="ID único de la regla")
    
    name: str = Field(description="Nombre descriptivo de la regla")
    
    description: str = Field(description="Descripción de qué hace la regla")
    
    intent_types: List[IntentType] = Field(description="Tipos de intención que activan la regla")
    
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Condiciones adicionales para activar la regla"
    )
    
    action: RoutingAction = Field(description="Acción a tomar cuando se activa")
    
    target_department: Optional[str] = Field(
        default=None,
        description="Departamento destino (si aplica)"
    )
    
    priority_boost: int = Field(
        ge=0,
        le=5,
        default=0,
        description="Boost de prioridad a aplicar"
    )
    
    auto_response_template: Optional[str] = Field(
        default=None,
        description="Template de respuesta automática"
    )
    
    is_active: bool = Field(
        default=True,
        description="Si la regla está activa"
    )
    
    execution_order: int = Field(
        ge=0,
        default=100,
        description="Orden de ejecución (menor = primero)"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de creación"
    )
    
    last_modified: datetime = Field(
        default_factory=datetime.now,
        description="Última modificación"
    )
    
    usage_count: int = Field(
        ge=0,
        default=0,
        description="Número de veces que se ha aplicado"
    )


class RoutingConfiguration(BaseModel):
    """
    Configuración completa del sistema de enrutamiento.
    
    Centraliza todas las reglas, departamentos y configuraciones de routing.
    """
    departments: Dict[str, DepartmentAvailability] = Field(
        default_factory=dict,
        description="Configuración de disponibilidad por departamento"
    )
    
    routing_rules: List[RoutingRule] = Field(
        default_factory=list,
        description="Lista de reglas de enrutamiento ordenadas"
    )
    
    default_department: str = Field(
        default="info@empresa.com",
        description="Departamento por defecto para casos no clasificables"
    )
    
    escalation_threshold_minutes: int = Field(
        ge=1,
        default=60,
        description="Minutos después de los cuales escalar automáticamente"
    )
    
    high_priority_intents: List[IntentType] = Field(
        default=[IntentType.COMPLAINT, IntentType.SUPPORT],
        description="Intenciones que se consideran alta prioridad"
    )
    
    auto_response_enabled: bool = Field(
        default=True,
        description="Si está habilitado el auto-response global"
    )
    
    business_hours_override: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Override global de horarios de negocio"
    )
    
    holiday_schedule: List[datetime] = Field(
        default_factory=list,
        description="Fechas de días festivos donde no hay atención"
    )
    
    load_balancing_enabled: bool = Field(
        default=True,
        description="Si está habilitado el balanceo de carga automático"
    )
    
    max_routing_attempts: int = Field(
        ge=1,
        default=3,
        description="Número máximo de intentos de enrutamiento"
    )
    
    fallback_contact: str = Field(
        default="gerencia@empresa.com",
        description="Contacto de fallback cuando todo falla"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de creación de la configuración"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Última actualización de la configuración"
    )


class RoutingMetrics(BaseModel):
    """
    Métricas de performance del sistema de enrutamiento.
    
    Rastrea estadísticas para optimización y monitoreo.
    """
    total_messages_routed: int = Field(
        ge=0,
        default=0,
        description="Total de mensajes enrutados"
    )
    
    successful_routings: int = Field(
        ge=0,
        default=0,
        description="Enrutamientos exitosos"
    )
    
    failed_routings: int = Field(
        ge=0,
        default=0,
        description="Enrutamientos fallidos"
    )
    
    average_routing_time_ms: float = Field(
        ge=0.0,
        default=0.0,
        description="Tiempo promedio de enrutamiento en milisegundos"
    )
    
    routing_accuracy: float = Field(
        ge=0.0,
        le=100.0,
        default=0.0,
        description="Precisión del enrutamiento como porcentaje"
    )
    
    intent_distribution: Dict[IntentType, int] = Field(
        default_factory=dict,
        description="Distribución de mensajes por tipo de intención"
    )
    
    department_workload: Dict[str, int] = Field(
        default_factory=dict,
        description="Carga de trabajo actual por departamento"
    )
    
    peak_hour_distribution: Dict[int, int] = Field(
        default_factory=dict,
        description="Distribución de mensajes por hora del día"
    )
    
    escalation_rate: float = Field(
        ge=0.0,
        le=100.0,
        default=0.0,
        description="Porcentaje de casos que requieren escalamiento"
    )
    
    auto_resolution_rate: float = Field(
        ge=0.0,
        le=100.0,
        default=0.0,
        description="Porcentaje de casos resueltos automáticamente"
    )
    
    last_calculated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp del último cálculo de métricas"
    )
    
    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito del enrutamiento."""
        total = self.successful_routings + self.failed_routings
        if total == 0:
            return 0.0
        return (self.successful_routings / total) * 100


# Configuración por defecto del sistema de enrutamiento
DEFAULT_ROUTING_CONFIG = RoutingConfiguration(
    departments={
        "ventas@empresa.com": DepartmentAvailability(
            department_email="ventas@empresa.com",
            max_capacity=30,
            working_hours_start=time(9, 0),
            working_hours_end=time(18, 0)
        ),
        "soporte@empresa.com": DepartmentAvailability(
            department_email="soporte@empresa.com", 
            max_capacity=50,
            working_hours_start=time(8, 0),
            working_hours_end=time(20, 0)
        ),
        "facturacion@empresa.com": DepartmentAvailability(
            department_email="facturacion@empresa.com",
            max_capacity=20,
            working_hours_start=time(9, 0), 
            working_hours_end=time(17, 0)
        ),
        "info@empresa.com": DepartmentAvailability(
            department_email="info@empresa.com",
            max_capacity=40,
            working_hours_start=time(9, 0),
            working_hours_end=time(18, 0),
            working_days=[0, 1, 2, 3, 4, 5]  # Incluye sábados
        ),
        "gerencia@empresa.com": DepartmentAvailability(
            department_email="gerencia@empresa.com",
            max_capacity=10,
            working_hours_start=time(8, 0),
            working_hours_end=time(19, 0),
            emergency_contact="emergency@empresa.com"
        )
    },
    
    routing_rules=[
        RoutingRule(
            rule_id="high_priority_complaints",
            name="Reclamos Alta Prioridad",
            description="Reclamos con palabras clave críticas van directo a gerencia",
            intent_types=[IntentType.COMPLAINT],
            conditions={"keywords": ["urgente", "abogado", "demanda", "terrible"]},
            action=RoutingAction.ESCALATE_TO_HUMAN,
            target_department="gerencia@empresa.com", 
            priority_boost=2,
            execution_order=1
        ),
        
        RoutingRule(
            rule_id="after_hours_auto_respond",
            name="Respuesta Automática Fuera de Horario",
            description="Auto-respuesta cuando todos los departamentos están cerrados",
            intent_types=list(IntentType),
            conditions={"time_condition": "outside_business_hours"},
            action=RoutingAction.AUTO_RESPOND,
            auto_response_template="Gracias por contactarnos. Hemos recibido tu mensaje fuera de horario de atención. Te responderemos el próximo día hábil.",
            execution_order=10
        ),
        
        RoutingRule(
            rule_id="capacity_overflow",
            name="Desbordamiento de Capacidad",
            description="Enrutar a departamento alternativo si el principal está saturado",
            intent_types=[IntentType.GENERAL, IntentType.INFORMATION],
            conditions={"department_capacity": ">90%"},
            action=RoutingAction.QUEUE_FOR_LATER,
            execution_order=5
        )
    ]
)