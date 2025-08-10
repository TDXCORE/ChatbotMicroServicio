"""
Tests de integración para el flujo completo del chatbot.

Testa la integración entre todos los componentes principales.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from fastapi.testclient import TestClient
from src.models.intents import IntentType, MessageIntent
from src.models.messages import WhatsAppMessage, MessageType, MessageStatus


class TestChatbotIntegration:
    """Tests de integración end-to-end."""
    
    @pytest.fixture
    def app_with_mocked_services(self):
        """App con servicios mockeados para testing."""
        
        # Import aquí para evitar problemas de inicialización
        from main import app
        
        # Mock todos los servicios globales
        with patch('main.whatsapp_service') as mock_whatsapp, \
             patch('main.llm_service') as mock_llm, \
             patch('main.context_service') as mock_context, \
             patch('main.intent_classifier') as mock_classifier:
            
            # Configurar mocks
            mock_whatsapp.is_connected = True
            mock_whatsapp.get_status.return_value = {
                "connection_status": {"is_connected": True},
                "statistics": {"messages_sent": 0, "messages_received": 0}
            }
            mock_whatsapp.send_message = AsyncMock(return_value=True)
            
            mock_llm.get_stats.return_value = {"total_calls": 0}
            mock_llm.health_check.return_value = {"status": "healthy"}
            mock_llm.generate_response = AsyncMock(return_value="Test response")
            
            mock_context.health_check.return_value = {"status": "healthy"}
            mock_context.add_message_to_history = AsyncMock(return_value=True)
            mock_context.get_stats.return_value = {"contexts_loaded": 0}
            
            # Mock clasificación de intenciones
            mock_intent = MessageIntent(
                intent=IntentType.SALES,
                confidence=0.9,
                entities={},
                routing_department="ventas@empresa.com",
                reasoning="Test classification"
            )
            
            mock_classification_response = AsyncMock()
            mock_classification_response.message_intent = mock_intent
            mock_classification_response.context_updated = AsyncMock()
            mock_classification_response.processing_time_ms = 100
            mock_classification_response.fallback_used = False
            
            mock_classifier.classify_intent = AsyncMock(return_value=mock_classification_response)
            mock_classifier.health_check.return_value = {"status": "healthy"}
            mock_classifier.get_stats.return_value = {"classifications_performed": 0}
            
            yield app
    
    @pytest.fixture
    def client(self, app_with_mocked_services):
        """Test client para FastAPI."""
        return TestClient(app_with_mocked_services)
    
    def test_health_endpoint(self, client):
        """Test endpoint de health check."""
        
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "uptime_seconds" in data
        assert "services" in data
        assert "stats" in data
        
        # Verificar servicios
        services = data["services"]
        expected_services = ["whatsapp", "context_redis", "llm_openai", "intent_classifier"]
        
        for service in expected_services:
            assert service in services
    
    def test_root_endpoint(self, client):
        """Test endpoint raíz."""
        
        # Act
        response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "WhatsApp Chatbot con Detección de Intenciones"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
    
    def test_stats_endpoint(self, client):
        """Test endpoint de estadísticas."""
        
        # Act
        response = client.get("/stats")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "app_stats" in data
        assert "uptime_seconds" in data
        
        # Verificar stats de app
        app_stats = data["app_stats"]
        expected_keys = [
            "messages_received", "messages_processed", "messages_failed",
            "intents_classified", "uptime_start", "last_activity"
        ]
        
        for key in expected_keys:
            assert key in app_stats
    
    @pytest.mark.asyncio
    async def test_whatsapp_webhook_processing(self, client):
        """Test procesamiento de webhook de WhatsApp."""
        
        # Arrange
        webhook_data = {
            "id": "test_msg_001",
            "from_number": "1234567890@c.us",
            "body": "Hola, ¿cuánto cuesta el producto?",
            "timestamp": datetime.now().isoformat(),
            "message_type": "text"
        }
        
        # Act
        response = client.post("/webhook/whatsapp", json=webhook_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "accepted"
        assert data["message_id"] == "test_msg_001"
        assert data["processing"] == "background"
    
    def test_webhook_empty_message_ignored(self, client):
        """Test que mensajes vacíos son ignorados."""
        
        # Arrange
        webhook_data = {
            "id": "empty_msg_001",
            "from_number": "1234567890@c.us",
            "body": "   ",  # Solo espacios
            "timestamp": datetime.now().isoformat(),
            "message_type": "text"
        }
        
        # Act
        response = client.post("/webhook/whatsapp", json=webhook_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ignored"
        assert data["reason"] == "empty_message"
    
    def test_manual_classification_endpoint(self, client):
        """Test endpoint de clasificación manual."""
        
        # Arrange
        classification_request = {
            "message": "¿Cuánto cuesta el producto premium?",
            "user_id": "test_user_001",
            "timestamp": datetime.now().isoformat()
        }
        
        # Act
        response = client.post("/classify", json=classification_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["intent"] == "ventas"
        assert data["confidence"] >= 0.5
        assert "department" in data
        assert "reasoning" in data
        assert "processing_time_ms" in data
        assert "fallback_used" in data
    
    def test_webhook_invalid_data_error(self, client):
        """Test manejo de errores en webhook con datos inválidos."""
        
        # Arrange
        invalid_data = {
            "id": "invalid_msg",
            # Falta from_number requerido
            "body": "Test message"
        }
        
        # Act
        response = client.post("/webhook/whatsapp", json=invalid_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_end_to_end_sales_flow(self, app_with_mocked_services):
        """Test flujo completo: mensaje de ventas → clasificación → respuesta."""
        
        # Este test simula el flujo completo que no puede ser fácilmente testeado
        # con TestClient debido a background tasks
        
        from main import process_incoming_message, whatsapp_service, intent_classifier, llm_service, context_service
        
        # Arrange
        sales_message = WhatsAppMessage(
            id="sales_test_001",
            from_number="1234567890@c.us",
            to_number="chatbot@c.us",
            body="Hola, estoy interesado en comprar el producto premium. ¿Cuánto cuesta?",
            message_type=MessageType.TEXT,
            timestamp=datetime.now(),
            status=MessageStatus.RECEIVED
        )
        
        # Act
        await process_incoming_message(sales_message)
        
        # Assert - verificar que se llamaron los servicios correctos
        context_service.add_message_to_history.assert_called_once()
        intent_classifier.classify_intent.assert_called_once()
        llm_service.generate_response.assert_called_once()
        whatsapp_service.send_message.assert_called_once()
        
        # Verificar argumentos de la clasificación
        classify_call = intent_classifier.classify_intent.call_args[0][0]
        assert classify_call.message == sales_message.body
        assert classify_call.user_id == sales_message.from_number
        
        # Verificar que se envió respuesta
        send_call = whatsapp_service.send_message.call_args
        assert send_call[1]["to_number"] == sales_message.from_number
        assert isinstance(send_call[1]["message"], str)
    
    @pytest.mark.asyncio
    async def test_end_to_end_support_flow(self, app_with_mocked_services):
        """Test flujo completo para mensaje de soporte."""
        
        from main import process_incoming_message, intent_classifier
        
        # Arrange - configurar mock para soporte
        support_intent = MessageIntent(
            intent=IntentType.SUPPORT,
            confidence=0.95,
            entities={"issue": "technical_problem"},
            routing_department="soporte@empresa.com",
            reasoning="Usuario reporta problema técnico"
        )
        
        mock_response = AsyncMock()
        mock_response.message_intent = support_intent
        mock_response.context_updated = AsyncMock()
        mock_response.processing_time_ms = 150
        mock_response.fallback_used = False
        
        intent_classifier.classify_intent.return_value = mock_response
        
        support_message = WhatsAppMessage(
            id="support_test_001",
            from_number="9876543210@c.us",
            to_number="chatbot@c.us",
            body="Tengo un problema serio con la aplicación, no puedo acceder a mi cuenta",
            message_type=MessageType.TEXT,
            timestamp=datetime.now(),
            status=MessageStatus.RECEIVED
        )
        
        # Act
        await process_incoming_message(support_message)
        
        # Assert
        intent_classifier.classify_intent.assert_called_once()
        
        # Verificar que la clasificación fue correcta
        classify_call = intent_classifier.classify_intent.call_args[0][0]
        assert "problema" in classify_call.message.lower()
    
    def test_concurrent_webhook_requests(self, client):
        """Test manejo de múltiples webhooks concurrentes."""
        
        import concurrent.futures
        
        # Arrange
        webhook_requests = []
        for i in range(10):
            webhook_requests.append({
                "id": f"concurrent_msg_{i:03d}",
                "from_number": f"123456789{i}@c.us",
                "body": f"Test message {i}",
                "timestamp": datetime.now().isoformat(),
                "message_type": "text"
            })
        
        # Act
        def send_webhook(data):
            return client.post("/webhook/whatsapp", json=data)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_webhook, data) for data in webhook_requests]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Assert
        assert len(responses) == 10
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"
    
    def test_error_handling_in_webhook(self, client):
        """Test manejo de errores en webhook."""
        
        # Arrange
        webhook_data = {
            "id": "error_test_001",
            "from_number": "1234567890@c.us", 
            "body": "Test message",
            "timestamp": datetime.now().isoformat(),
            "message_type": "text"
        }
        
        # Mock para generar error interno
        with patch('main.process_incoming_message', side_effect=Exception("Internal error")):
            
            # Act
            response = client.post("/webhook/whatsapp", json=webhook_data)
            
            # Assert
            # El webhook debe aceptar el mensaje aunque el procesamiento falle
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"


class TestServiceIntegration:
    """Tests de integración entre servicios específicos."""
    
    @pytest.mark.asyncio
    async def test_llm_context_service_integration(self, mock_llm_service, mock_context_service):
        """Test integración entre LLM Service y Context Service."""
        
        # Arrange
        from src.agents.intent_classifier import IntentClassifierAgent
        
        agent = IntentClassifierAgent(
            llm_service=mock_llm_service,
            context_service=mock_context_service
        )
        
        # Simular contexto existente
        existing_context = AsyncMock()
        existing_context.user_id = "test_user"
        existing_context.message_history = ["Usuario: Hola", "Asistente: ¿En qué puedo ayudarte?"]
        existing_context.current_intent = None
        
        mock_context_service.get_context.return_value = existing_context
        
        # Mock respuesta de clasificación
        intent_result = MessageIntent(
            intent=IntentType.SALES,
            confidence=0.8,
            entities={},
            routing_department="ventas@empresa.com",
            reasoning="Test integration"
        )
        
        mock_llm_service.classify_intent.return_value = intent_result
        
        # Arrange request
        from src.models.intents import IntentClassificationRequest
        
        request = IntentClassificationRequest(
            message="¿Cuánto cuesta?",
            user_id="test_user"
        )
        
        # Act
        response = await agent.classify_intent(request)
        
        # Assert - verificar que los servicios interactuaron correctamente
        mock_context_service.get_context.assert_called_once_with("test_user")
        mock_llm_service.classify_intent.assert_called_once()
        mock_context_service.save_context.assert_called_once()
        
        # Verificar respuesta
        assert response.message_intent.intent == IntentType.SALES
        assert not response.fallback_used