"""
Intent Classification Agent - WhatsApp Chatbot Pattern

Este ejemplo muestra cómo implementar un agente LangChain para clasificar
intenciones de mensajes de WhatsApp usando OpenAI function calling.

Key patterns:
- Structured output con Pydantic models
- Few-shot prompting para mejorar precision
- Confidence scoring para handling edge cases
- Context awareness para conversaciones multi-turn
"""

from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json


class IntentType(str, Enum):
    SALES = "ventas"
    SUPPORT = "soporte" 
    BILLING = "facturacion"
    GENERAL = "general"
    COMPLAINT = "reclamo"
    INFORMATION = "informacion"
    UNKNOWN = "desconocido"


class MessageIntent(BaseModel):
    """
    Structured output model para intent classification.
    
    OpenAI function calling requires precise schema definitions.
    This model defines exactly what we expect back.
    """
    intent: IntentType = Field(description="Classified intent category")
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence score between 0 and 1"
    )
    entities: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities like product names, account numbers"
    )
    routing_department: str = Field(description="Target department for routing")
    reasoning: str = Field(description="Brief explanation of classification")


class ConversationContext(BaseModel):
    """Context from previous messages in conversation."""
    user_id: str
    message_history: list[str] = Field(max_items=10)  # Limit to prevent token overflow
    previous_intent: Optional[IntentType] = None
    conversation_start: datetime = Field(default_factory=datetime.now)


class IntentClassifierAgent:
    """
    LangChain agent para clasificar intenciones en mensajes WhatsApp.
    
    Key features:
    - Uses OpenAI function calling for structured output
    - Context-aware (considers conversation history)  
    - Fallback to rule-based classification on low confidence
    - Configurable confidence thresholds
    """
    
    def __init__(
        self, 
        openai_api_key: str,
        model_name: str = "gpt-4-1106-preview",
        confidence_threshold: float = 0.7
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=500
        )
        self.confidence_threshold = confidence_threshold
        self.parser = PydanticOutputParser(pydantic_object=MessageIntent)
        
        # Department routing mapping
        self.department_routing = {
            IntentType.SALES: "ventas@empresa.com",
            IntentType.SUPPORT: "soporte@empresa.com", 
            IntentType.BILLING: "facturacion@empresa.com",
            IntentType.GENERAL: "info@empresa.com",
            IntentType.COMPLAINT: "gerencia@empresa.com",
            IntentType.INFORMATION: "info@empresa.com",
            IntentType.UNKNOWN: "soporte@empresa.com"  # Default fallback
        }
        
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """
        Setup few-shot prompt template para intent classification.
        
        Few-shot examples significantly improve classification accuracy.
        Include examples for each intent type con typical variations.
        """
        
        self.system_message = """Eres un experto clasificador de intenciones para un sistema de atención al cliente via WhatsApp.

Tu tarea es clasificar cada mensaje del usuario en una de las siguientes categorías:

1. VENTAS (ventas): Consultas sobre precios, productos, compras, cotizaciones
   - "¿Cuánto cuesta el producto X?"
   - "Quiero comprar esto"
   - "¿Hacen descuentos por cantidad?"

2. SOPORTE (soporte): Problemas técnicos, averías, dudas de uso
   - "El producto no funciona"
   - "¿Cómo se usa esto?"
   - "Tengo un error en la aplicación"

3. FACTURACIÓN (facturacion): Consultas sobre pagos, facturas, cobros
   - "No me llegó la factura"
   - "¿Cuándo vence mi pago?"
   - "Tengo un cargo desconocido"

4. GENERAL (general): Información básica como horarios, ubicación, contactos
   - "¿Cuáles son sus horarios?"
   - "¿Dónde están ubicados?"
   - "¿Cómo los contacto?"

5. RECLAMO (reclamo): Quejas, insatisfacción, problemas de servicio
   - "Estoy muy molesto con el servicio"
   - "Quiero hacer una queja formal"
   - "El trato fue pésimo"

6. INFORMACIÓN (informacion): Solicitudes de información sobre servicios
   - "¿Qué servicios ofrecen?"
   - "Cuéntame sobre sus productos"
   - "¿Trabajan con empresas?"

7. DESCONOCIDO (unknown): Mensajes que no se pueden clasificar claramente

IMPORTANTE: 
- Considera el CONTEXTO de la conversación previa
- Si la confianza es menor a 0.7, clasifica como UNKNOWN
- Extrae entidades relevantes (nombres productos, números cuenta, etc.)
- Proporciona una explicación breve de tu clasificación

Responde SIEMPRE en el formato JSON especificado."""

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", """Mensaje del usuario: "{message}"

Contexto de la conversación:
- Historial de mensajes: {message_history}
- Intención previa: {previous_intent}
- Usuario: {user_id}

Clasifica este mensaje y extrae las entidades relevantes.

{format_instructions}""")
        ])

    async def classify_intent(
        self, 
        message: str, 
        context: ConversationContext
    ) -> MessageIntent:
        """
        Classify intent of WhatsApp message using conversation context.
        
        Args:
            message: User message text
            context: Conversation context with history
            
        Returns:
            MessageIntent with classification and confidence
        """
        
        # Prepare context for prompt
        message_history = " | ".join(context.message_history[-5:])  # Last 5 messages
        previous_intent = context.previous_intent.value if context.previous_intent else "None"
        
        # Format prompt with context
        prompt = self.prompt_template.format_messages(
            message=message,
            message_history=message_history,
            previous_intent=previous_intent,
            user_id=context.user_id,
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            # Call OpenAI with function calling
            response = await self.llm.ainvoke(prompt)
            
            # Parse structured output
            intent_result = self.parser.parse(response.content)
            
            # Add routing department
            intent_result.routing_department = self.department_routing[intent_result.intent]
            
            # Apply confidence threshold check
            if intent_result.confidence < self.confidence_threshold:
                intent_result = self._fallback_classification(message, context)
            
            return intent_result
            
        except Exception as e:
            # Fallback on any error
            print(f"Intent classification error: {e}")
            return self._fallback_classification(message, context)
    
    def _fallback_classification(
        self, 
        message: str, 
        context: ConversationContext
    ) -> MessageIntent:
        """
        Rule-based fallback when LLM classification fails or low confidence.
        
        Simple keyword matching as backup para ensure system reliability.
        """
        
        message_lower = message.lower()
        
        # Rule-based keyword matching
        sales_keywords = ["precio", "cuesta", "comprar", "cotización", "descuento"]
        support_keywords = ["problema", "error", "no funciona", "ayuda", "falla"]
        billing_keywords = ["factura", "pago", "cobro", "cuenta", "cargo"]
        complaint_keywords = ["queja", "reclamo", "molesto", "malo", "pésimo"]
        
        if any(keyword in message_lower for keyword in sales_keywords):
            intent = IntentType.SALES
        elif any(keyword in message_lower for keyword in support_keywords):
            intent = IntentType.SUPPORT
        elif any(keyword in message_lower for keyword in billing_keywords):
            intent = IntentType.BILLING
        elif any(keyword in message_lower for keyword in complaint_keywords):
            intent = IntentType.COMPLAINT
        else:
            intent = IntentType.UNKNOWN
        
        return MessageIntent(
            intent=intent,
            confidence=0.6,  # Lower confidence for rule-based
            entities={},
            routing_department=self.department_routing[intent],
            reasoning="Fallback rule-based classification"
        )


# Example usage
async def example_usage():
    """
    Example de cómo usar el IntentClassifierAgent en un chatbot WhatsApp.
    """
    
    import os
    
    # Initialize agent
    classifier = IntentClassifierAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        confidence_threshold=0.75
    )
    
    # Simulate conversation context
    context = ConversationContext(
        user_id="whatsapp_1234567890",
        message_history=[
            "Hola, buenos días",
            "Estoy interesado en sus productos"
        ],
        previous_intent=IntentType.GENERAL
    )
    
    # Classify new message
    user_message = "¿Cuánto cuesta el plan premium?"
    
    result = await classifier.classify_intent(user_message, context)
    
    print(f"Intent: {result.intent}")
    print(f"Confidence: {result.confidence}")
    print(f"Department: {result.routing_department}")
    print(f"Reasoning: {result.reasoning}")
    
    # Expected output:
    # Intent: IntentType.SALES
    # Confidence: 0.92
    # Department: ventas@empresa.com
    # Reasoning: User asking about pricing for premium plan


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())