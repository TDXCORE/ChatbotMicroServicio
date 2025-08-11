"""
Intent Classifier Agent - LangChain Agent para clasificación de intenciones

Agente especializado en clasificar intenciones de mensajes WhatsApp usando
LangChain con OpenAI function calling y fallback logic robusto.

Basado en: examples/langchain_agents/intent_classifier_agent.py
"""

import time
from typing import Dict, Optional, List, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..models.intents import (
    IntentType, MessageIntent, ConversationContext, 
    IntentClassificationRequest, IntentClassificationResponse,
    DEFAULT_DEPARTMENT_CONFIGS
)
from ..services.llm_service import LLMService
from ..services.context_service import ContextService
from ..utils.config import get_settings
from ..utils.logger import get_logger, log_intent_classification

logger = get_logger(__name__)
settings = get_settings()


class IntentClassificationTool(BaseTool):
    """
    Tool de LangChain para clasificación de intenciones.
    
    Integra con el LLM Service para structured output.
    """
    
    name: str = "classify_intent"
    description: str = "Clasifica la intención de un mensaje de WhatsApp del usuario"
    llm_service: LLMService = Field(description="LLM Service for classification")
    
    def __init__(self, llm_service: LLMService, **kwargs):
        super().__init__(llm_service=llm_service, **kwargs)
    
    def _run(self, message: str, context: str = "") -> str:
        """Run tool synchronously (no usado)."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, message: str, context: str = "") -> Dict[str, Any]:
        """
        Run tool asynchronously.
        
        Args:
            message: Mensaje a clasificar
            context: Contexto conversacional
            
        Returns:
            Diccionario con clasificación
        """
        
        try:
            # Parse context si está disponible
            conversation_context = None
            if context:
                # TODO: Parse context from string representation
                pass
            
            # Clasificar usando LLM service
            intent = await self.llm_service.classify_intent(
                message=message,
                context=conversation_context
            )
            
            return {
                "intent": intent.intent.value,
                "confidence": intent.confidence,
                "department": intent.routing_department,
                "reasoning": intent.reasoning,
                "entities": intent.entities
            }
            
        except Exception as e:
            logger.error(f"❌ Error en tool de clasificación: {e}")
            return {
                "intent": IntentType.UNKNOWN.value,
                "confidence": 0.0,
                "department": settings.SUPPORT_EMAIL,
                "reasoning": f"Error en clasificación: {str(e)}",
                "entities": {}
            }


class ContextAnalysisTool(BaseTool):
    """
    Tool para analizar contexto conversacional.
    
    Proporciona información sobre el historial y estado de la conversación.
    """
    
    name: str = "analyze_context"
    description: str = "Analiza el contexto conversacional para mejorar clasificación"
    context_service: ContextService = Field(description="Context Service for conversation history")
    
    def __init__(self, context_service: ContextService, **kwargs):
        super().__init__(context_service=context_service, **kwargs)
    
    def _run(self, user_id: str) -> str:
        """Run tool synchronously (no usado)."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, user_id: str) -> Dict[str, Any]:
        """
        Analiza contexto para usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Información contextual
        """
        
        try:
            context = await self.context_service.get_context(user_id)
            
            if not context:
                return {
                    "has_history": False,
                    "message_count": 0,
                    "last_intent": None,
                    "conversation_age_minutes": 0
                }
            
            # Calcular edad de conversación
            age_minutes = (datetime.now() - context.session_start).total_seconds() / 60
            
            return {
                "has_history": len(context.message_history) > 0,
                "message_count": len(context.message_history),
                "last_intent": context.current_intent.intent.value if context.current_intent else None,
                "last_intent_confidence": context.current_intent.confidence if context.current_intent else 0.0,
                "conversation_age_minutes": age_minutes,
                "is_active": context.is_active,
                "language": context.language
            }
            
        except Exception as e:
            logger.error(f"❌ Error analizando contexto: {e}")
            return {
                "has_history": False,
                "message_count": 0,
                "last_intent": None,
                "conversation_age_minutes": 0,
                "error": str(e)
            }


class IntentClassifierAgent:
    """
    Agente LangChain especializado en clasificación de intenciones.
    
    Features:
    - Integración con LLM Service para structured output
    - Context-aware classification usando historial conversacional
    - Fallback logic robusto para errores
    - Confidence thresholding configurable
    - Performance monitoring y logging
    - Multi-departamento routing logic
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        context_service: Optional[ContextService] = None,
        confidence_threshold: float = None
    ):
        self.llm_service = llm_service
        self.context_service = context_service
        self.confidence_threshold = confidence_threshold or settings.INTENT_CONFIDENCE_THRESHOLD
        
        # Configurar LangChain LLM (opcional)
        self.llm = None
        if llm_service is not None:
            try:
                self.llm = ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL,
                    temperature=settings.OPENAI_TEMPERATURE,
                    max_tokens=settings.OPENAI_MAX_TOKENS
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize OpenAI LLM: {e}")
        
        # Configurar tools (servicios opcionales)
        self.tools = []
        
        if llm_service is not None:
            self.tools.append(IntentClassificationTool(llm_service))
        else:
            logger.warning("⚠️ LLM Service not available - using fallback classification only")
        
        if context_service is not None:
            self.tools.append(ContextAnalysisTool(context_service))
        else:
            logger.warning("⚠️ Context Service not available - running without conversation history")
        
        # Configurar agent (opcional)
        self.agent = None
        if self.llm is not None and len(self.tools) > 0:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True if settings.DEBUG else False,
                max_iterations=3,  # Límite para evitar loops infinitos
                handle_parsing_errors=True
            )
        else:
            logger.warning("⚠️ Agent not initialized - running in fallback-only mode")
        
        # Routing mapping
        self.department_configs = DEFAULT_DEPARTMENT_CONFIGS
        
        # Statistics
        self.stats = {
            "classifications_performed": 0,
            "successful_classifications": 0,
            "fallback_classifications": 0,
            "high_confidence_classifications": 0,
            "low_confidence_classifications": 0,
            "average_processing_time_ms": 0.0,
            "intent_distribution": {intent.value: 0 for intent in IntentType}
        }
    
    async def classify_intent(
        self,
        request: IntentClassificationRequest
    ) -> IntentClassificationResponse:
        """
        Clasifica intención de un mensaje con contexto completo.
        
        Args:
            request: Request con mensaje y contexto
            
        Returns:
            Response con clasificación y contexto actualizado
        """
        
        start_time = time.time()
        
        try:
            # Obtener o crear contexto
            context = request.context
            if not context:
                context = await self.context_service.get_context(request.user_id)
                if not context:
                    context = ConversationContext(
                        user_id=request.user_id,
                        conversation_id=f"{request.user_id}_{int(time.time())}"
                    )
            
            # Usar LLM service directamente (más eficiente)
            message_intent = await self.llm_service.classify_intent(
                message=request.message,
                context=context
            )
            
            # Verificar threshold de confianza
            if message_intent.confidence < self.confidence_threshold:
                logger.warning(
                    f"⚠️ Baja confianza en clasificación: {message_intent.confidence:.2f} "
                    f"(threshold: {self.confidence_threshold})"
                )
                
                # Aplicar fallback si confianza es muy baja
                if message_intent.confidence < 0.5:
                    message_intent = await self._apply_fallback_classification(
                        request.message, context
                    )
            
            # Actualizar contexto
            context.current_intent = message_intent
            context.last_activity = datetime.now()
            
            # Añadir mensaje al historial
            user_message_text = f"Usuario: {request.message}"
            context.message_history.append(user_message_text)
            
            # Guardar contexto actualizado
            await self.context_service.save_context(context)
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - start_time) * 1000
            
            # Actualizar estadísticas
            self._update_stats(message_intent, processing_time, fallback_used=False)
            
            # Log clasificación
            log_intent_classification(
                logger=logger,
                user_id=request.user_id,
                message=request.message,
                intent_type=message_intent.intent.value,
                confidence=message_intent.confidence,
                processing_time=processing_time
            )
            
            # Crear respuesta
            response = IntentClassificationResponse(
                message_intent=message_intent,
                context_updated=context,
                processing_time_ms=processing_time,
                model_used=settings.OPENAI_MODEL,
                fallback_used=False
            )
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            logger.error(f"❌ Error en clasificación de intenciones: {e}", exc_info=True)
            
            # Fallback completo
            fallback_intent = await self._apply_fallback_classification(
                request.message, 
                request.context or ConversationContext(
                    user_id=request.user_id,
                    conversation_id=f"{request.user_id}_fallback"
                )
            )
            
            # Actualizar estadísticas
            self._update_stats(fallback_intent, processing_time, fallback_used=True)
            
            return IntentClassificationResponse(
                message_intent=fallback_intent,
                context_updated=request.context or ConversationContext(
                    user_id=request.user_id,
                    conversation_id=f"{request.user_id}_fallback"
                ),
                processing_time_ms=processing_time,
                model_used="fallback_rules",
                fallback_used=True
            )
    
    async def _apply_fallback_classification(
        self,
        message: str,
        context: ConversationContext
    ) -> MessageIntent:
        """
        Aplica clasificación fallback basada en reglas.
        
        Args:
            message: Mensaje a clasificar
            context: Contexto conversacional
            
        Returns:
            MessageIntent con clasificación rule-based
        """
        
        logger.info("🔄 Aplicando clasificación fallback rule-based")
        
        message_lower = message.lower()
        
        # Reglas mejoradas por palabras clave
        sales_keywords = [
            "precio", "cuesta", "comprar", "cotización", "descuento", 
            "oferta", "promoción", "vender", "adquirir", "cuánto"
        ]
        
        support_keywords = [
            "problema", "error", "no funciona", "falla", "ayuda", 
            "soporte", "técnico", "bug", "roto", "arreglar"
        ]
        
        billing_keywords = [
            "factura", "pago", "cobro", "cuenta", "billing", 
            "cargo", "débito", "crédito", "dinero", "costo"
        ]
        
        complaint_keywords = [
            "queja", "reclamo", "molesto", "pésimo", "malo", 
            "terrible", "enojado", "insatisfecho", "demanda"
        ]
        
        general_keywords = [
            "horario", "ubicación", "dirección", "contacto", 
            "teléfono", "email", "donde", "cuándo", "cómo"
        ]
        
        # Scoring por keywords
        scores = {
            IntentType.SALES: sum(1 for kw in sales_keywords if kw in message_lower),
            IntentType.SUPPORT: sum(1 for kw in support_keywords if kw in message_lower),
            IntentType.BILLING: sum(1 for kw in billing_keywords if kw in message_lower),
            IntentType.COMPLAINT: sum(1 for kw in complaint_keywords if kw in message_lower),
            IntentType.GENERAL: sum(1 for kw in general_keywords if kw in message_lower)
        }
        
        # Considerar contexto previo
        if context.current_intent:
            # Boost para intención previa (contexto conversacional)
            scores[context.current_intent.intent] += 1
        
        # Encontrar intención con mayor score
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # Si no hay matches claros, usar UNKNOWN
        if best_score == 0:
            best_intent = IntentType.UNKNOWN
            confidence = 0.3
        else:
            # Confidence basado en score relativo
            total_score = sum(scores.values())
            confidence = min(0.6, best_score / max(total_score, 1) * 0.8)  # Max 0.6 para fallback
        
        # Mapear a departamento
        department_mapping = {
            IntentType.SALES: settings.SALES_EMAIL,
            IntentType.SUPPORT: settings.SUPPORT_EMAIL,
            IntentType.BILLING: settings.BILLING_EMAIL,
            IntentType.GENERAL: settings.GENERAL_EMAIL,
            IntentType.COMPLAINT: settings.COMPLAINT_EMAIL,
            IntentType.INFORMATION: settings.INFORMATION_EMAIL,
            IntentType.UNKNOWN: settings.SUPPORT_EMAIL
        }
        
        return MessageIntent(
            intent=best_intent,
            confidence=confidence,
            entities=self._extract_simple_entities(message),
            routing_department=department_mapping[best_intent],
            reasoning=f"Clasificación rule-based: {best_score} matches para {best_intent.value}",
            processed_at=datetime.now()
        )
    
    def _extract_simple_entities(self, message: str) -> Dict[str, Any]:
        """
        Extrae entidades simples usando regex.
        
        Args:
            message: Mensaje a analizar
            
        Returns:
            Diccionario con entidades encontradas
        """
        
        import re
        entities = {}
        
        # Números de teléfono
        phone_pattern = r'\b\d{10,15}\b'
        phones = re.findall(phone_pattern, message)
        if phones:
            entities["phone_numbers"] = phones
        
        # Emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        if emails:
            entities["emails"] = emails
        
        # Precios/montos
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        money = re.findall(money_pattern, message)
        if money:
            entities["amounts"] = money
        
        # Fechas simples
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, message)
        if dates:
            entities["dates"] = dates
        
        return entities
    
    def _update_stats(
        self, 
        intent: MessageIntent, 
        processing_time: float,
        fallback_used: bool
    ):
        """Actualiza estadísticas del agente."""
        
        self.stats["classifications_performed"] += 1
        
        if fallback_used:
            self.stats["fallback_classifications"] += 1
        else:
            self.stats["successful_classifications"] += 1
        
        if intent.confidence >= 0.8:
            self.stats["high_confidence_classifications"] += 1
        else:
            self.stats["low_confidence_classifications"] += 1
        
        # Actualizar distribución de intenciones
        self.stats["intent_distribution"][intent.intent.value] += 1
        
        # Actualizar tiempo promedio
        total_classifications = self.stats["classifications_performed"]
        current_avg = self.stats["average_processing_time_ms"]
        
        self.stats["average_processing_time_ms"] = (
            (current_avg * (total_classifications - 1) + processing_time) / total_classifications
        )
    
    def get_intent_suggestions(self, partial_message: str) -> List[Dict[str, Any]]:
        """
        Proporciona sugerencias de intención para mensaje parcial.
        
        Args:
            partial_message: Mensaje parcial
            
        Returns:
            Lista de posibles intenciones con scores
        """
        
        message_lower = partial_message.lower()
        suggestions = []
        
        # Evaluar cada tipo de intención
        for intent_type in IntentType:
            if intent_type == IntentType.UNKNOWN:
                continue
            
            # Get department config
            dept_config = self.department_configs.get(intent_type)
            if not dept_config:
                continue
            
            # Simple keyword matching score
            score = 0
            keywords = self._get_keywords_for_intent(intent_type)
            
            for keyword in keywords:
                if keyword in message_lower:
                    score += 1
            
            if score > 0:
                suggestions.append({
                    "intent": intent_type.value,
                    "department": dept_config.email,
                    "score": score,
                    "confidence_estimate": min(score * 0.2, 0.9)
                })
        
        # Ordenar por score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:3]  # Top 3 sugerencias
    
    def _get_keywords_for_intent(self, intent_type: IntentType) -> List[str]:
        """Obtiene keywords para tipo de intención."""
        
        keyword_mapping = {
            IntentType.SALES: ["precio", "comprar", "vender", "cotización"],
            IntentType.SUPPORT: ["problema", "error", "ayuda", "soporte"],
            IntentType.BILLING: ["factura", "pago", "cobro", "dinero"],
            IntentType.COMPLAINT: ["queja", "reclamo", "molesto", "malo"],
            IntentType.GENERAL: ["horario", "ubicación", "contacto", "información"],
            IntentType.INFORMATION: ["servicios", "productos", "empresa", "información"]
        }
        
        return keyword_mapping.get(intent_type, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del agente."""
        
        success_rate = 0.0
        if self.stats["classifications_performed"] > 0:
            success_rate = (
                self.stats["successful_classifications"] / 
                self.stats["classifications_performed"]
            ) * 100
        
        fallback_rate = 0.0
        if self.stats["classifications_performed"] > 0:
            fallback_rate = (
                self.stats["fallback_classifications"] / 
                self.stats["classifications_performed"]
            ) * 100
        
        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "fallback_rate_percent": fallback_rate,
            "confidence_threshold": self.confidence_threshold
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica salud del agente.
        
        Returns:
            Estado de salud y métricas
        """
        
        try:
            # Test clasificación simple
            test_request = IntentClassificationRequest(
                message="Test de conectividad",
                user_id="health_check_user"
            )
            
            start_time = time.time()
            response = await self.classify_intent(test_request)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "test_classification": {
                    "intent": response.message_intent.intent.value,
                    "confidence": response.message_intent.confidence
                },
                "stats": self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check falló: {e}")
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }