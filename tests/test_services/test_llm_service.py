"""
Tests para LLM Service.

Cubre clasificación de intenciones, retry logic, y rate limiting.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch
from datetime import datetime

import openai
from src.services.llm_service import LLMService, RateLimitError, TokenLimitError, RetryHandler
from src.models.intents import IntentType, MessageIntent, ConversationContext
from src.utils.config import get_settings


class TestRetryHandler:
    """Tests para RetryHandler."""
    
    @pytest.fixture
    def retry_handler(self):
        """Retry handler con configuración rápida para tests."""
        return RetryHandler(
            max_retries=2,
            initial_delay=0.01,  # 10ms para tests rápidos
            multiplier=2.0
        )
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, retry_handler):
        """Test ejecución exitosa sin retries."""
        
        # Arrange
        async def success_func():
            return "success"
        
        # Act
        result = await retry_handler.execute_with_retry(success_func)
        
        # Assert
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self, retry_handler):
        """Test retry en timeout errors."""
        
        # Arrange
        call_count = 0
        
        async def timeout_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.APITimeoutError("Timeout")
            return "success_after_retry"
        
        # Act
        result = await retry_handler.execute_with_retry(timeout_then_success)
        
        # Assert
        assert result == "success_after_retry"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, retry_handler):
        """Test retry en connection errors."""
        
        # Arrange
        call_count = 0
        
        async def connection_then_success():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise openai.APIConnectionError("Connection failed")
            return "success_after_retry"
        
        # Act
        result = await retry_handler.execute_with_retry(connection_then_success)
        
        # Assert
        assert result == "success_after_retry"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, retry_handler):
        """Test manejo específico de rate limit errors."""
        
        # Arrange
        call_count = 0
        
        async def rate_limit_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.RateLimitError("Rate limit exceeded")
            return "success_after_rate_limit"
        
        # Act
        result = await retry_handler.execute_with_retry(rate_limit_then_success)
        
        # Assert
        assert result == "success_after_rate_limit"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, retry_handler):
        """Test que se respeta el límite máximo de retries."""
        
        # Arrange
        async def always_fail():
            raise openai.APITimeoutError("Always timeout")
        
        # Act & Assert
        with pytest.raises(openai.APITimeoutError):
            await retry_handler.execute_with_retry(always_fail)
    
    @pytest.mark.asyncio
    async def test_no_retry_on_bad_request(self, retry_handler):
        """Test que no se reintenta en bad request errors."""
        
        # Arrange
        async def bad_request():
            raise openai.BadRequestError("Bad request")
        
        # Act & Assert
        with pytest.raises(openai.BadRequestError):
            await retry_handler.execute_with_retry(bad_request)


class TestLLMService:
    """Tests para LLMService."""
    
    @pytest.fixture
    def llm_service(self):
        """LLM service configurado para tests."""
        return LLMService()
    
    @pytest.fixture
    def mock_openai_client(self, llm_service):
        """Mock del cliente OpenAI."""
        with patch.object(llm_service, 'client') as mock_client:
            yield mock_client
    
    @pytest.mark.asyncio
    async def test_successful_intent_classification(self, llm_service, mock_openai_client):
        """Test clasificación exitosa de intención."""
        
        # Arrange
        message = "¿Cuánto cuesta el producto premium?"
        
        # Mock response de OpenAI
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.function_call.arguments = json.dumps({
            "intent": "ventas",
            "confidence": 0.9,
            "entities": {"product": "premium"},
            "reasoning": "Usuario pregunta sobre precios"
        })
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = await llm_service.classify_intent(message)
        
        # Assert
        assert isinstance(result, MessageIntent)
        assert result.intent == IntentType.SALES
        assert result.confidence == 0.9
        assert result.entities["product"] == "premium"
        assert result.routing_department == get_settings().SALES_EMAIL
        assert "precios" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_support_intent_classification(self, llm_service, mock_openai_client):
        """Test clasificación de intención de soporte."""
        
        # Arrange
        message = "La aplicación no funciona correctamente, necesito ayuda"
        
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.function_call.arguments = json.dumps({
            "intent": "soporte",
            "confidence": 0.95,
            "entities": {"issue": "application_not_working"},
            "reasoning": "Usuario reporta problema con aplicación"
        })
        mock_response.usage.total_tokens = 120
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 40
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = await llm_service.classify_intent(message)
        
        # Assert
        assert result.intent == IntentType.SUPPORT
        assert result.confidence == 0.95
        assert result.routing_department == get_settings().SUPPORT_EMAIL
    
    @pytest.mark.asyncio
    async def test_context_aware_classification(self, llm_service, mock_openai_client):
        """Test clasificación con contexto conversacional."""
        
        # Arrange
        message = "Sí, me interesa comprarlo"
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_history=[
                "Usuario: Hola",
                "Asistente: ¿En qué puedo ayudarte?",
                "Usuario: ¿Tienen productos disponibles?",
                "Asistente: Sí, tenemos varios productos. ¿Te interesa alguno en particular?"
            ]
        )
        
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.function_call.arguments = json.dumps({
            "intent": "ventas",
            "confidence": 0.85,
            "entities": {},
            "reasoning": "Usuario confirma interés en compra basado en contexto previo"
        })
        mock_response.usage.total_tokens = 180
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Act
        result = await llm_service.classify_intent(message, context)
        
        # Assert
        assert result.intent == IntentType.SALES
        assert "contexto" in result.reasoning or "previo" in result.reasoning
        
        # Verificar que se incluyó el contexto en la llamada
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) > 2  # System + context + user message
    
    @pytest.mark.asyncio
    async def test_fallback_on_openai_error(self, llm_service, mock_openai_client):
        """Test fallback cuando OpenAI API falla."""
        
        # Arrange
        message = "problema con factura pago"
        
        # Simular error de OpenAI
        mock_openai_client.chat.completions.create.side_effect = openai.APIConnectionError("Connection failed")
        
        # Act
        result = await llm_service.classify_intent(message)
        
        # Assert
        assert isinstance(result, MessageIntent)
        assert result.intent == IntentType.BILLING  # Debería detectar keywords
        assert result.confidence <= 0.6  # Fallback tiene menor confianza
        assert "fallback" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_token_limit_handling(self, llm_service, mock_openai_client):
        """Test manejo de límites de tokens."""
        
        # Arrange
        # Crear contexto muy largo para exceder límites
        long_context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_history=["Mensaje muy largo " * 100] * 50  # Muchos mensajes largos
        )
        
        message = "Test message"
        
        # Mock para compression call
        compression_response = AsyncMock()
        compression_response.choices = [AsyncMock()]
        compression_response.choices[0].message.content = "Resumen comprimido de conversación previa"
        
        # Mock para classification call
        classification_response = AsyncMock()
        classification_response.choices = [AsyncMock()]
        classification_response.choices[0].message.function_call.arguments = json.dumps({
            "intent": "general",
            "confidence": 0.8,
            "entities": {},
            "reasoning": "Mensaje general"
        })
        classification_response.usage.total_tokens = 100
        
        mock_openai_client.chat.completions.create.side_effect = [
            compression_response,  # Primera llamada para compresión
            classification_response  # Segunda llamada para clasificación
        ]
        
        # Act
        result = await llm_service.classify_intent(message, long_context)
        
        # Assert
        assert isinstance(result, MessageIntent)
        assert result.intent == IntentType.GENERAL
        
        # Verificar que se hicieron 2 llamadas (compresión + clasificación)
        assert mock_openai_client.chat.completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_response_generation(self, llm_service):
        """Test generación de respuestas automáticas."""
        
        # Act & Assert para diferentes tipos de intención
        test_cases = [
            (IntentType.SALES, "ventas"),
            (IntentType.SUPPORT, "soporte"),
            (IntentType.BILLING, "facturación"),
            (IntentType.COMPLAINT, "gerencia"),
            (IntentType.GENERAL, "información")
        ]
        
        for intent_type, expected_keyword in test_cases:
            response = await llm_service.generate_response(
                intent_type=intent_type,
                user_message="Test message"
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Verificar que la respuesta es contextual al tipo de intención
            assert any(keyword in response.lower() for keyword in [expected_keyword, "contacto", "ayuda", "responder"])
    
    @pytest.mark.asyncio
    async def test_custom_template_response(self, llm_service):
        """Test generación con template personalizado."""
        
        # Arrange
        custom_template = "Respuesta personalizada para el usuario"
        
        # Act
        response = await llm_service.generate_response(
            intent_type=IntentType.SALES,
            user_message="Test",
            template=custom_template
        )
        
        # Assert
        assert response == custom_template
    
    def test_stats_tracking(self, llm_service):
        """Test tracking de estadísticas."""
        
        # Act
        stats = llm_service.get_stats()
        
        # Assert
        expected_keys = [
            "total_calls", "successful_calls", "failed_calls",
            "success_rate_percent", "total_tokens_used", "estimated_cost_usd",
            "average_response_time_ms", "rate_limit_hits"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert isinstance(stats["total_calls"], int)
        assert isinstance(stats["success_rate_percent"], float)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service, mock_openai_client):
        """Test health check exitoso."""
        
        # Arrange
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "OK"
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Act
        health = await llm_service.health_check()
        
        # Assert
        assert health["status"] == "healthy"
        assert health["api_accessible"] is True
        assert "response_time_ms" in health
        assert health["model"] == get_settings().OPENAI_MODEL
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service, mock_openai_client):
        """Test health check cuando API falla."""
        
        # Arrange
        mock_openai_client.chat.completions.create.side_effect = openai.APIConnectionError("API unavailable")
        
        # Act
        health = await llm_service.health_check()
        
        # Assert
        assert health["status"] == "unhealthy"
        assert health["api_accessible"] is False
        assert "error" in health
        assert "API unavailable" in health["error"]
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, llm_service, mock_openai_client):
        """Test manejo de rate limit errors."""
        
        # Arrange
        message = "test message"
        
        # Simular rate limit error
        mock_openai_client.chat.completions.create.side_effect = openai.RateLimitError("Rate limit exceeded")
        
        # Act
        result = await llm_service.classify_intent(message)
        
        # Assert
        # Debe usar fallback
        assert isinstance(result, MessageIntent)
        assert result.confidence <= 0.6  # Fallback confidence
        assert "fallback" in result.reasoning.lower()
    
    def test_department_mapping(self, llm_service):
        """Test mapeo correcto de intenciones a departamentos."""
        
        # Test todos los tipos de intención
        test_cases = [
            (IntentType.SALES, get_settings().SALES_EMAIL),
            (IntentType.SUPPORT, get_settings().SUPPORT_EMAIL),
            (IntentType.BILLING, get_settings().BILLING_EMAIL),
            (IntentType.GENERAL, get_settings().GENERAL_EMAIL),
            (IntentType.COMPLAINT, get_settings().COMPLAINT_EMAIL),
            (IntentType.INFORMATION, get_settings().INFORMATION_EMAIL),
            (IntentType.UNKNOWN, get_settings().SUPPORT_EMAIL)
        ]
        
        for intent_type, expected_department in test_cases:
            department = llm_service._get_department_for_intent(intent_type)
            assert department == expected_department
    
    @pytest.mark.asyncio
    async def test_concurrent_classifications(self, llm_service, mock_openai_client):
        """Test clasificaciones concurrentes."""
        
        # Arrange
        messages = [
            "¿Cuánto cuesta?",
            "Tengo un problema", 
            "Mi factura está mal",
            "Quiero hacer una queja"
        ]
        
        mock_responses = []
        expected_intents = [IntentType.SALES, IntentType.SUPPORT, IntentType.BILLING, IntentType.COMPLAINT]
        
        for i, intent in enumerate(expected_intents):
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.function_call.arguments = json.dumps({
                "intent": intent.value,
                "confidence": 0.8,
                "entities": {},
                "reasoning": f"Test classification {i}"
            })
            mock_response.usage.total_tokens = 100
            mock_responses.append(mock_response)
        
        mock_openai_client.chat.completions.create.side_effect = mock_responses
        
        # Act
        tasks = [llm_service.classify_intent(msg) for msg in messages]
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 4
        for i, result in enumerate(results):
            assert result.intent == expected_intents[i]
            assert result.confidence >= 0.8