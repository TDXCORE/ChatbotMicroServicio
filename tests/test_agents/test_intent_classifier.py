"""
Tests para Intent Classifier Agent.

Cubre clasificación de intenciones, fallback logic, y context awareness.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from src.agents.intent_classifier import IntentClassifierAgent
from src.models.intents import (
    IntentType, MessageIntent, ConversationContext,
    IntentClassificationRequest, IntentClassificationResponse
)
from src.models.messages import WhatsAppMessage, MessageType


class TestIntentClassifierAgent:
    """Tests para IntentClassifierAgent."""
    
    @pytest.fixture
    async def agent(self, mock_llm_service, mock_context_service):
        """Agente configurado para tests."""
        return IntentClassifierAgent(
            llm_service=mock_llm_service,
            context_service=mock_context_service,
            confidence_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_successful_intent_classification(self, agent, mock_llm_service):
        """Test clasificación exitosa de intención."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="¿Cuánto cuesta el producto premium?",
            user_id="1234567890@c.us"
        )
        
        expected_intent = MessageIntent(
            intent=IntentType.SALES,
            confidence=0.9,
            entities={"product": "premium"},
            routing_department="ventas@empresa.com",
            reasoning="Usuario pregunta sobre precios"
        )
        
        mock_llm_service.classify_intent.return_value = expected_intent
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert isinstance(response, IntentClassificationResponse)
        assert response.message_intent.intent == IntentType.SALES
        assert response.message_intent.confidence >= 0.7
        assert response.message_intent.routing_department == "ventas@empresa.com"
        assert not response.fallback_used
        
        # Verificar que se llamó al LLM service
        mock_llm_service.classify_intent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_support_intent_classification(self, agent, mock_llm_service):
        """Test clasificación de intención de soporte."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="La aplicación no funciona, tengo un error",
            user_id="1234567890@c.us"
        )
        
        expected_intent = MessageIntent(
            intent=IntentType.SUPPORT,
            confidence=0.85,
            entities={},
            routing_department="soporte@empresa.com",
            reasoning="Usuario reporta problema técnico"
        )
        
        mock_llm_service.classify_intent.return_value = expected_intent
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert response.message_intent.intent == IntentType.SUPPORT
        assert response.message_intent.routing_department == "soporte@empresa.com"
        assert response.message_intent.confidence >= 0.7
    
    @pytest.mark.asyncio
    async def test_billing_intent_classification(self, agent, mock_llm_service):
        """Test clasificación de intención de facturación."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="No me llegó la factura del mes pasado",
            user_id="1234567890@c.us"
        )
        
        expected_intent = MessageIntent(
            intent=IntentType.BILLING,
            confidence=0.8,
            entities={"period": "mes pasado"},
            routing_department="facturacion@empresa.com",
            reasoning="Usuario consulta sobre facturación"
        )
        
        mock_llm_service.classify_intent.return_value = expected_intent
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert response.message_intent.intent == IntentType.BILLING
        assert response.message_intent.routing_department == "facturacion@empresa.com"
    
    @pytest.mark.asyncio
    async def test_complaint_intent_classification(self, agent, mock_llm_service):
        """Test clasificación de intención de reclamo."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="Estoy muy molesto con el servicio, quiero hacer una queja formal",
            user_id="1234567890@c.us"
        )
        
        expected_intent = MessageIntent(
            intent=IntentType.COMPLAINT,
            confidence=0.95,
            entities={},
            routing_department="gerencia@empresa.com",
            reasoning="Usuario expresa insatisfacción y quiere hacer queja"
        )
        
        mock_llm_service.classify_intent.return_value = expected_intent
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert response.message_intent.intent == IntentType.COMPLAINT
        assert response.message_intent.routing_department == "gerencia@empresa.com"
        assert response.message_intent.confidence >= 0.9
    
    @pytest.mark.asyncio
    async def test_low_confidence_triggers_fallback(self, agent, mock_llm_service):
        """Test que baja confianza activa fallback logic."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="mensaje confuso y ambiguo xyx",
            user_id="1234567890@c.us"
        )
        
        # LLM retorna baja confianza
        low_confidence_intent = MessageIntent(
            intent=IntentType.GENERAL,
            confidence=0.4,  # Menor que threshold (0.7)
            entities={},
            routing_department="info@empresa.com",
            reasoning="Clasificación incierta"
        )
        
        mock_llm_service.classify_intent.return_value = low_confidence_intent
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        # Debe usar fallback y reclasificar
        assert response.fallback_used or response.message_intent.confidence < 0.7
        assert response.message_intent.intent in [IntentType.UNKNOWN, IntentType.GENERAL]
    
    @pytest.mark.asyncio
    async def test_fallback_classification_sales_keywords(self, agent, mock_llm_service):
        """Test fallback classification con keywords de ventas."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="precio comprar cotización",
            user_id="1234567890@c.us"
        )
        
        # Simular error en LLM service para activar fallback
        mock_llm_service.classify_intent.side_effect = Exception("OpenAI API error")
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert response.fallback_used
        assert response.message_intent.intent == IntentType.SALES
        assert response.message_intent.confidence <= 0.6  # Fallback tiene menor confianza
        assert "fallback" in response.message_intent.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_fallback_classification_support_keywords(self, agent, mock_llm_service):
        """Test fallback classification con keywords de soporte."""
        
        # Arrange 
        request = IntentClassificationRequest(
            message="problema error no funciona ayuda",
            user_id="1234567890@c.us"
        )
        
        # Simular error en LLM service
        mock_llm_service.classify_intent.side_effect = Exception("API error")
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert response.fallback_used
        assert response.message_intent.intent == IntentType.SUPPORT
        assert response.message_intent.routing_department == "soporte@empresa.com"
    
    @pytest.mark.asyncio
    async def test_fallback_unknown_for_unclear_message(self, agent, mock_llm_service):
        """Test fallback classification para mensajes poco claros."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="abc xyz 123",
            user_id="1234567890@c.us"
        )
        
        # Simular error en LLM service
        mock_llm_service.classify_intent.side_effect = Exception("API error")
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        assert response.fallback_used
        assert response.message_intent.intent == IntentType.UNKNOWN
        assert response.message_intent.routing_department == "soporte@empresa.com"  # Fallback dept
    
    @pytest.mark.asyncio 
    async def test_context_awareness(self, agent, mock_llm_service, mock_context_service):
        """Test que el agente usa contexto conversacional."""
        
        # Arrange
        existing_context = ConversationContext(
            user_id="1234567890@c.us",
            conversation_id="test_conv",
            message_history=[
                "Usuario: Hola",
                "Asistente: ¿En qué puedo ayudarte?", 
                "Usuario: Tengo preguntas sobre productos"
            ],
            current_intent=MessageIntent(
                intent=IntentType.SALES,
                confidence=0.8,
                entities={},
                routing_department="ventas@empresa.com",
                reasoning="Contexto previo"
            )
        )
        
        mock_context_service.get_context.return_value = existing_context
        
        request = IntentClassificationRequest(
            message="¿También tienen descuentos?",
            user_id="1234567890@c.us"
        )
        
        expected_intent = MessageIntent(
            intent=IntentType.SALES,
            confidence=0.9,
            entities={},
            routing_department="ventas@empresa.com",
            reasoning="Continúa conversación sobre ventas"
        )
        
        mock_llm_service.classify_intent.return_value = expected_intent
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        # Verificar que se obtuvo y actualizó el contexto
        mock_context_service.get_context.assert_called_once_with("1234567890@c.us")
        mock_context_service.save_context.assert_called_once()
        
        # El contexto debe incluir el nuevo mensaje
        saved_context = mock_context_service.save_context.call_args[0][0]
        assert "Usuario: ¿También tienen descuentos?" in saved_context.message_history[-1]
    
    @pytest.mark.asyncio
    async def test_entity_extraction_in_fallback(self, agent, mock_llm_service):
        """Test extracción de entidades en fallback mode."""
        
        # Arrange
        request = IntentClassificationRequest(
            message="Mi número es 1234567890 y mi email es test@example.com, precio $500",
            user_id="1234567890@c.us"
        )
        
        # Simular error para activar fallback
        mock_llm_service.classify_intent.side_effect = Exception("API error")
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert
        entities = response.message_intent.entities
        assert "phone_numbers" in entities
        assert "1234567890" in entities["phone_numbers"]
        assert "emails" in entities  
        assert "test@example.com" in entities["emails"]
        assert "amounts" in entities
        assert "$500" in entities["amounts"]
    
    @pytest.mark.asyncio
    async def test_intent_suggestions(self, agent):
        """Test sugerencias de intención para mensaje parcial."""
        
        # Act
        suggestions = agent.get_intent_suggestions("quiero comprar producto")
        
        # Assert
        assert len(suggestions) > 0
        assert any(s["intent"] == IntentType.SALES.value for s in suggestions)
        assert all("score" in s for s in suggestions)
        assert all("confidence_estimate" in s for s in suggestions)
        
        # Las sugerencias deben estar ordenadas por score
        scores = [s["score"] for s in suggestions]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, agent, mock_llm_service):
        """Test que las estadísticas se actualizan correctamente."""
        
        # Arrange
        requests = [
            IntentClassificationRequest(message="venta producto", user_id="user1"),
            IntentClassificationRequest(message="problema técnico", user_id="user2"),
            IntentClassificationRequest(message="mensaje confuso", user_id="user3")
        ]
        
        # Configure mock responses
        mock_responses = [
            MessageIntent(intent=IntentType.SALES, confidence=0.9, entities={}, routing_department="ventas@empresa.com", reasoning="test"),
            MessageIntent(intent=IntentType.SUPPORT, confidence=0.85, entities={}, routing_department="soporte@empresa.com", reasoning="test"),
            MessageIntent(intent=IntentType.UNKNOWN, confidence=0.3, entities={}, routing_department="info@empresa.com", reasoning="test")
        ]
        
        mock_llm_service.classify_intent.side_effect = mock_responses
        
        # Act
        for request in requests:
            await agent.classify_intent(request)
        
        # Assert
        stats = agent.get_stats()
        assert stats["classifications_performed"] == 3
        assert stats["successful_classifications"] >= 2  # Primeros 2 exitosos
        assert stats["high_confidence_classifications"] >= 2
        assert stats["average_processing_time_ms"] > 0
        
        # Verificar distribución de intenciones
        assert stats["intent_distribution"][IntentType.SALES.value] >= 1
        assert stats["intent_distribution"][IntentType.SUPPORT.value] >= 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent, mock_llm_service, mock_context_service):
        """Test health check del agente."""
        
        # Arrange
        mock_llm_service.classify_intent.return_value = MessageIntent(
            intent=IntentType.GENERAL,
            confidence=0.8,
            entities={},
            routing_department="info@empresa.com",
            reasoning="Health check test"
        )
        
        # Act
        health = await agent.health_check()
        
        # Assert
        assert health["status"] == "healthy"
        assert "response_time_ms" in health
        assert "test_classification" in health
        assert health["test_classification"]["intent"] == IntentType.GENERAL.value
        assert "stats" in health
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, agent, mock_llm_service):
        """Test health check cuando hay errores."""
        
        # Arrange
        mock_llm_service.classify_intent.side_effect = Exception("Service unavailable")
        
        # Act
        health = await agent.health_check()
        
        # Assert
        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "Service unavailable" in health["error"]
    
    def test_keywords_for_intents(self, agent):
        """Test que se obtienen keywords correctos para cada intención."""
        
        # Test sales keywords
        sales_keywords = agent._get_keywords_for_intent(IntentType.SALES)
        assert "precio" in sales_keywords
        assert "comprar" in sales_keywords
        
        # Test support keywords
        support_keywords = agent._get_keywords_for_intent(IntentType.SUPPORT) 
        assert "problema" in support_keywords
        assert "error" in support_keywords
        
        # Test billing keywords
        billing_keywords = agent._get_keywords_for_intent(IntentType.BILLING)
        assert "factura" in billing_keywords
        assert "pago" in billing_keywords