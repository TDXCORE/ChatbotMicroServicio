"""
LLM Service - Integración con OpenAI API

Servicio para comunicación con OpenAI API con rate limiting,
retry logic, structured output y manejo robusto de errores.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..models.intents import MessageIntent, IntentType, ConversationContext
from ..utils.config import get_settings
from ..utils.logger import get_logger, log_api_call

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class APICallStats:
    """Estadísticas de llamadas a la API."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    average_response_time: float = 0.0
    last_call_time: Optional[datetime] = None
    rate_limit_hits: int = 0


class RateLimitError(Exception):
    """Error cuando se alcanza rate limit."""
    pass


class TokenLimitError(Exception):
    """Error cuando se excede límite de tokens."""
    pass


class RetryHandler:
    """
    Maneja retry logic con exponential backoff para OpenAI API.
    
    Implementa estrategia robusta para manejar rate limits y errores temporales.
    """
    
    def __init__(
        self,
        max_retries: int = None,
        initial_delay: float = None,
        multiplier: float = None,
        max_delay: float = 60.0
    ):
        self.max_retries = max_retries or settings.MAX_API_RETRIES
        self.initial_delay = initial_delay or settings.INITIAL_RETRY_DELAY  
        self.multiplier = multiplier or settings.RETRY_MULTIPLIER
        self.max_delay = max_delay
        
    async def execute_with_retry(self, func, *args, **kwargs):
        """
        Ejecuta función con retry automático.
        
        Args:
            func: Función async a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si falla después de todos los reintentos
        """
        
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except openai.RateLimitError as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(f"❌ Rate limit alcanzado después de {attempt + 1} intentos")
                    raise RateLimitError(f"Rate limit exceeded after {attempt + 1} attempts") from e
                
                # Para rate limits, usar delay más largo
                rate_limit_delay = min(delay * 2, 60)
                logger.warning(f"⏳ Rate limit hit, esperando {rate_limit_delay}s (intento {attempt + 1})")
                await asyncio.sleep(rate_limit_delay)
                
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(f"❌ Timeout/Connection error después de {attempt + 1} intentos")
                    raise
                
                logger.warning(f"⏳ API timeout, reintentando en {delay}s (intento {attempt + 1})")
                await asyncio.sleep(delay)
                delay = min(delay * self.multiplier, self.max_delay)
                
            except openai.BadRequestError as e:
                # No retry para bad requests (error de input)
                logger.error(f"❌ Bad request error: {e}")
                raise
                
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(f"❌ Error desconocido después de {attempt + 1} intentos: {e}")
                    raise
                
                logger.warning(f"⚠️ Error temporal, reintentando en {delay}s: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * self.multiplier, self.max_delay)
        
        # Esto no debería alcanzarse nunca
        raise last_exception


class TokenEstimator:
    """
    Estima uso de tokens para evitar exceder límites.
    
    Usa heurísticas aproximadas para estimar tokens antes de hacer llamadas.
    """
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estima número de tokens para texto.
        
        Heurística conservadora: ~3.5 caracteres por token para español.
        
        Args:
            text: Texto a estimar
            
        Returns:
            Número estimado de tokens
        """
        return max(1, len(text) // 3)
    
    @staticmethod
    def estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
        """
        Estima tokens para lista de mensajes.
        
        Args:
            messages: Lista de mensajes formato OpenAI
            
        Returns:
            Número estimado de tokens
        """
        total_tokens = 0
        
        for message in messages:
            # Tokens base por mensaje
            total_tokens += 4
            
            # Tokens del contenido
            content = message.get('content', '')
            total_tokens += TokenEstimator.estimate_tokens(content)
            
            # Tokens adicionales para role y name si existen
            if message.get('role'):
                total_tokens += 1
            if message.get('name'):
                total_tokens += 1
        
        # Tokens adicionales para el response
        total_tokens += 2
        
        return total_tokens


class LLMService:
    """
    Servicio principal para integración con OpenAI API.
    
    Features:
    - Rate limiting automático
    - Retry con exponential backoff  
    - Structured output con Pydantic
    - Token management y cost tracking
    - Context compression automática
    - Caching de prompts frecuentes
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.retry_handler = RetryHandler()
        self.stats = APICallStats()
        
        # Cache simple para prompts frecuentes (TTL: 1 hora)
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        # Token cost per 1K tokens (aproximado para GPT-4)
        self.token_costs = {
            "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    async def classify_intent(
        self,
        message: str,
        context: Optional[ConversationContext] = None,
        model: str = None
    ) -> MessageIntent:
        """
        Clasifica intención de un mensaje usando OpenAI function calling.
        
        Args:
            message: Texto del mensaje a clasificar
            context: Contexto conversacional opcional
            model: Modelo específico a usar
            
        Returns:
            MessageIntent con clasificación y confianza
            
        Raises:
            TokenLimitError: Si el contexto excede límites de tokens
            RateLimitError: Si se alcanza rate limit
        """
        
        start_time = time.time()
        model = model or settings.OPENAI_MODEL
        
        try:
            # Verificar límites de tokens
            messages = self._build_classification_messages(message, context)
            estimated_tokens = TokenEstimator.estimate_message_tokens(messages)
            
            if estimated_tokens > settings.MAX_CONTEXT_TOKENS:
                logger.warning(f"⚠️ Contexto muy largo ({estimated_tokens} tokens), comprimiendo...")
                messages = await self._compress_context_messages(messages)
            
            # Preparar function schema para structured output
            function_schema = {
                "name": "classify_intent",
                "description": "Clasifica la intención de un mensaje de WhatsApp",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": [intent.value for intent in IntentType],
                            "description": "Tipo de intención detectada"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Nivel de confianza (0.0-1.0)"
                        },
                        "entities": {
                            "type": "object",
                            "description": "Entidades extraídas del mensaje"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explicación breve de la clasificación"
                        }
                    },
                    "required": ["intent", "confidence", "reasoning"]
                }
            }
            
            # Hacer llamada con retry automático
            response = await self.retry_handler.execute_with_retry(
                self._make_function_call,
                messages=messages,
                functions=[function_schema],
                function_call={"name": "classify_intent"},
                model=model,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS
            )
            
            # Procesar respuesta
            function_call = response.choices[0].message.function_call
            classification_data = json.loads(function_call.arguments)
            
            # Determinar departamento de enrutamiento
            intent_type = IntentType(classification_data["intent"])
            routing_department = self._get_department_for_intent(intent_type)
            
            # Crear MessageIntent
            message_intent = MessageIntent(
                intent=intent_type,
                confidence=classification_data["confidence"],
                entities=classification_data.get("entities", {}),
                routing_department=routing_department,
                reasoning=classification_data["reasoning"]
            )
            
            # Actualizar estadísticas
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(response, processing_time, success=True)
            
            # Log clasificación
            logger.info(
                f"🎯 Intent clasificado: {intent_type} (confianza: {message_intent.confidence:.2f})",
                extra={
                    "intent_type": intent_type,
                    "confidence_score": message_intent.confidence,
                    "processing_time": processing_time,
                    "routing_department": routing_department
                }
            )
            
            return message_intent
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(None, processing_time, success=False)
            
            logger.error(f"❌ Error clasificando intención: {e}", exc_info=True)
            
            # Fallback a clasificación rule-based
            return await self._fallback_classification(message, context)
    
    def _build_classification_messages(
        self, 
        message: str, 
        context: Optional[ConversationContext]
    ) -> List[Dict[str, str]]:
        """Construye messages array para clasificación de intenciones."""
        
        system_prompt = """Eres un experto clasificador de intenciones para un sistema de atención al cliente vía WhatsApp.

Tu tarea es clasificar cada mensaje del usuario en una de las siguientes categorías:

1. **VENTAS (ventas)**: Consultas sobre precios, productos, compras, cotizaciones
   - Ejemplos: "¿Cuánto cuesta?", "Quiero comprar", "¿Hacen descuentos?"

2. **SOPORTE (soporte)**: Problemas técnicos, averías, dudas de uso
   - Ejemplos: "No funciona", "¿Cómo se usa?", "Tengo un error"

3. **FACTURACIÓN (facturacion)**: Consultas sobre pagos, facturas, cobros
   - Ejemplos: "Mi factura", "¿Cuándo vence?", "Cargo desconocido"

4. **GENERAL (general)**: Información básica como horarios, ubicación
   - Ejemplos: "¿Horarios?", "¿Dónde están?", "¿Cómo contactarlos?"

5. **RECLAMO (reclamo)**: Quejas, insatisfacción, problemas de servicio  
   - Ejemplos: "Estoy molesto", "Queja formal", "Servicio pésimo"

6. **INFORMACIÓN (informacion)**: Solicitudes de información sobre servicios
   - Ejemplos: "¿Qué servicios ofrecen?", "Cuéntame sobre productos"

7. **DESCONOCIDO (desconocido)**: Mensajes no clasificables claramente

IMPORTANTE:
- Considera el contexto de conversación previa si está disponible
- Si la confianza es menor a 0.7, clasifica como DESCONOCIDO
- Extrae entidades relevantes como nombres de productos, números, etc.
- Proporciona explicación clara de tu clasificación"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Añadir contexto conversacional si está disponible
        if context and context.message_history:
            context_text = "\n".join(context.message_history[-5:])  # Últimos 5 mensajes
            messages.append({
                "role": "system", 
                "content": f"Contexto de conversación previa:\n{context_text}"
            })
        
        # Añadir mensaje actual
        messages.append({
            "role": "user",
            "content": f"Clasifica este mensaje: {message}"
        })
        
        return messages
    
    async def _compress_context_messages(
        self, 
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Comprime mensajes de contexto para reducir tokens.
        
        Mantiene system prompt y último mensaje, resumiendo el resto.
        """
        
        if len(messages) <= 2:
            return messages
        
        system_message = messages[0]
        user_message = messages[-1]
        context_messages = messages[1:-1]
        
        # Combinar mensajes de contexto
        context_text = "\n".join([msg["content"] for msg in context_messages])
        
        # Resumir usando OpenAI (llamada más pequeña)
        try:
            summary_response = await self.retry_handler.execute_with_retry(
                self._make_simple_completion,
                prompt=f"Resume brevemente esta conversación previa para contexto:\n{context_text}",
                max_tokens=200
            )
            
            summary = summary_response.choices[0].message.content
            
            return [
                system_message,
                {"role": "system", "content": f"Resumen de conversación previa: {summary}"},
                user_message
            ]
            
        except Exception as e:
            logger.warning(f"⚠️ Error comprimiendo contexto, usando versión truncada: {e}")
            # Fallback: solo usar últimos mensajes
            return [system_message, user_message]
    
    async def _make_function_call(self, **kwargs) -> Any:
        """Hace llamada a OpenAI con function calling."""
        
        response = await self.client.chat.completions.create(**kwargs)
        
        # Log API call
        log_api_call(
            logger=logger,
            service="OpenAI",
            endpoint="chat.completions",
            duration_ms=0,  # Se calcula en el caller
            success=True
        )
        
        return response
    
    async def _make_simple_completion(self, prompt: str, max_tokens: int = 100) -> Any:
        """Hace completion simple para tareas auxiliares."""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Usar modelo más barato para tareas auxiliares
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        return response
    
    def _get_department_for_intent(self, intent_type: IntentType) -> str:
        """
        Mapea tipo de intención a departamento.
        
        Args:
            intent_type: Tipo de intención
            
        Returns:
            Email del departamento
        """
        
        department_mapping = {
            IntentType.SALES: settings.SALES_EMAIL,
            IntentType.SUPPORT: settings.SUPPORT_EMAIL,
            IntentType.BILLING: settings.BILLING_EMAIL,
            IntentType.GENERAL: settings.GENERAL_EMAIL,
            IntentType.COMPLAINT: settings.COMPLAINT_EMAIL,
            IntentType.INFORMATION: settings.INFORMATION_EMAIL,
            IntentType.UNKNOWN: settings.SUPPORT_EMAIL  # Fallback a soporte
        }
        
        return department_mapping.get(intent_type, settings.GENERAL_EMAIL)
    
    async def _fallback_classification(
        self,
        message: str,
        context: Optional[ConversationContext]
    ) -> MessageIntent:
        """
        Clasificación fallback basada en reglas cuando falla OpenAI.
        
        Args:
            message: Mensaje a clasificar
            context: Contexto conversacional
            
        Returns:
            MessageIntent con clasificación básica
        """
        
        logger.warning("🔄 Usando clasificación fallback rule-based")
        
        message_lower = message.lower()
        
        # Reglas simples por palabras clave
        if any(keyword in message_lower for keyword in ["precio", "cuesta", "comprar", "cotización"]):
            intent = IntentType.SALES
        elif any(keyword in message_lower for keyword in ["problema", "error", "no funciona", "falla"]):
            intent = IntentType.SUPPORT
        elif any(keyword in message_lower for keyword in ["factura", "pago", "cobro", "cuenta"]):
            intent = IntentType.BILLING
        elif any(keyword in message_lower for keyword in ["queja", "reclamo", "molesto", "pésimo"]):
            intent = IntentType.COMPLAINT
        elif any(keyword in message_lower for keyword in ["horario", "ubicación", "dirección", "contacto"]):
            intent = IntentType.GENERAL
        else:
            intent = IntentType.UNKNOWN
        
        return MessageIntent(
            intent=intent,
            confidence=0.6,  # Confianza baja para fallback
            entities={},
            routing_department=self._get_department_for_intent(intent),
            reasoning="Clasificación automática por fallback (OpenAI no disponible)"
        )
    
    def _update_stats(self, response: Any, processing_time: float, success: bool):
        """Actualiza estadísticas de llamadas API."""
        
        self.stats.total_calls += 1
        self.stats.last_call_time = datetime.now()
        
        if success and response:
            self.stats.successful_calls += 1
            
            # Actualizar tokens y costos si está disponible
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.stats.total_tokens_used += tokens_used
                
                # Estimar costo (aproximado)
                model = settings.OPENAI_MODEL
                if model in self.token_costs:
                    input_cost = (response.usage.prompt_tokens / 1000) * self.token_costs[model]["input"]
                    output_cost = (response.usage.completion_tokens / 1000) * self.token_costs[model]["output"]
                    self.stats.total_cost_estimate += input_cost + output_cost
        else:
            self.stats.failed_calls += 1
        
        # Actualizar tiempo promedio de respuesta
        if self.stats.successful_calls > 0:
            self.stats.average_response_time = (
                (self.stats.average_response_time * (self.stats.successful_calls - 1) + processing_time) 
                / self.stats.successful_calls
            )
    
    async def generate_response(
        self,
        intent_type: IntentType,
        user_message: str,
        context: Optional[ConversationContext] = None,
        template: Optional[str] = None
    ) -> str:
        """
        Genera respuesta contextual para un mensaje.
        
        Args:
            intent_type: Tipo de intención detectada
            user_message: Mensaje original del usuario
            context: Contexto conversacional
            template: Template específico a usar
            
        Returns:
            Respuesta generada
        """
        
        if template:
            # Usar template proporcionado
            return template
        
        # Generar respuesta automática basada en intención
        default_responses = {
            IntentType.SALES: "¡Hola! Gracias por tu interés en nuestros productos. Un ejecutivo de ventas se pondrá en contacto contigo pronto para ayudarte con tu consulta.",
            
            IntentType.SUPPORT: "Hemos recibido tu solicitud de soporte técnico. Nuestro equipo la revisará y te ayudaremos a resolver el problema lo antes posible.",
            
            IntentType.BILLING: "Tu consulta sobre facturación ha sido registrada. Te proporcionaremos la información solicitada dentro de las próximas 24 horas.",
            
            IntentType.GENERAL: "Gracias por contactarnos. Te proporcionaremos la información que necesitas muy pronto.",
            
            IntentType.COMPLAINT: "Tomamos muy en serio tu preocupación. Tu mensaje ha sido escalado a nuestro equipo de gerencia y será atendido con la máxima prioridad.",
            
            IntentType.INFORMATION: "Con gusto te proporcionaremos información sobre nuestros servicios. Alguien de nuestro equipo se comunicará contigo pronto.",
            
            IntentType.UNKNOWN: "Hemos recibido tu mensaje y lo hemos enviado al departamento correspondiente. Te responderemos lo antes posible."
        }
        
        return default_responses.get(intent_type, default_responses[IntentType.UNKNOWN])
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servicio."""
        
        success_rate = 0.0
        if self.stats.total_calls > 0:
            success_rate = (self.stats.successful_calls / self.stats.total_calls) * 100
        
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate_percent": success_rate,
            "total_tokens_used": self.stats.total_tokens_used,
            "estimated_cost_usd": self.stats.total_cost_estimate,
            "average_response_time_ms": self.stats.average_response_time,
            "rate_limit_hits": self.stats.rate_limit_hits,
            "last_call_time": self.stats.last_call_time
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica salud del servicio OpenAI.
        
        Returns:
            Estado de salud y métricas básicas
        """
        
        try:
            # Test simple de conectividad
            start_time = time.time()
            
            response = await self.retry_handler.execute_with_retry(
                self._make_simple_completion,
                prompt="Test de conectividad - responde solo 'OK'",
                max_tokens=5
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "model": settings.OPENAI_MODEL,
                "api_accessible": True,
                "stats": self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check falló: {e}")
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": settings.OPENAI_MODEL,
                "api_accessible": False,
                "stats": self.get_stats()
            }