"""
Configuración global para tests pytest.

Define fixtures y configuración común para todos los tests.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

from src.models.intents import IntentType, MessageIntent, ConversationContext
from src.models.messages import WhatsAppMessage, MessageType, MessageStatus
from src.utils.config import Settings, set_settings_for_testing


@pytest.fixture(scope="session")
def test_settings():
    """Configuración de testing."""
    return Settings(
        # OpenAI Mock
        OPENAI_API_KEY="sk-test-key-for-testing",
        OPENAI_MODEL="gpt-4-test",
        
        # Redis Mock
        REDIS_URL="redis://localhost:6379/1",  # Test DB
        REDIS_SESSION_TTL=300,  # 5 minutos para tests
        
        # WhatsApp Mock
        WHATSAPP_SESSION_PATH="./test_session",
        WHATSAPP_HEADLESS=True,
        WHATSAPP_MESSAGE_DELAY=0.1,  # Más rápido para tests
        
        # Rate limiting más permisivo
        RATE_LIMIT_PER_USER=100,
        RATE_LIMIT_WINDOW=60,
        
        # Environment
        ENVIRONMENT="testing",
        LOG_LEVEL="DEBUG",
        DEBUG=True,
        MOCK_EXTERNAL_SERVICES=True,
        
        # Confidence más bajo para tests
        INTENT_CONFIDENCE_THRESHOLD=0.5
    )


@pytest.fixture(autouse=True)
def setup_test_settings(test_settings):
    """Auto-setup settings de testing para todos los tests."""
    set_settings_for_testing(test_settings)


@pytest.fixture
def sample_whatsapp_message():
    """Mensaje WhatsApp de ejemplo para tests."""
    return WhatsAppMessage(
        id="test_msg_001",
        from_number="1234567890@c.us",
        to_number="chatbot@c.us",
        body="Hola, necesito ayuda con mi cuenta",
        message_type=MessageType.TEXT,
        timestamp=datetime.now(),
        status=MessageStatus.RECEIVED
    )


@pytest.fixture
def sample_sales_message():
    """Mensaje de ventas para tests de clasificación."""
    return WhatsAppMessage(
        id="sales_msg_001",
        from_number="1234567890@c.us",
        to_number="chatbot@c.us",
        body="¿Cuánto cuesta el producto premium? Estoy interesado en comprar",
        message_type=MessageType.TEXT,
        timestamp=datetime.now(),
        status=MessageStatus.RECEIVED
    )


@pytest.fixture
def sample_support_message():
    """Mensaje de soporte para tests."""
    return WhatsAppMessage(
        id="support_msg_001", 
        from_number="1234567890@c.us",
        to_number="chatbot@c.us",
        body="Tengo un problema con la aplicación, no funciona correctamente",
        message_type=MessageType.TEXT,
        timestamp=datetime.now(),
        status=MessageStatus.RECEIVED
    )


@pytest.fixture
def sample_conversation_context():
    """Contexto conversacional de ejemplo."""
    return ConversationContext(
        user_id="1234567890@c.us",
        conversation_id="test_conv_001",
        message_history=[
            "Usuario: Hola",
            "Asistente: ¡Hola! ¿En qué puedo ayudarte?",
            "Usuario: Tengo una pregunta sobre precios"
        ],
        current_intent=MessageIntent(
            intent=IntentType.SALES,
            confidence=0.8,
            entities={},
            routing_department="ventas@empresa.com",
            reasoning="Pregunta sobre precios"
        ),
        is_active=True,
        language="es"
    )


@pytest.fixture
def mock_openai_response():
    """Mock response de OpenAI para tests."""
    return {
        "choices": [{
            "message": {
                "function_call": {
                    "name": "classify_intent",
                    "arguments": json.dumps({
                        "intent": "ventas",
                        "confidence": 0.9,
                        "entities": {"product": "premium"},
                        "reasoning": "Usuario pregunta sobre precios de productos"
                    })
                }
            }
        }],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200
        }
    }


@pytest.fixture
def mock_redis_client():
    """Mock cliente Redis para tests."""
    client = AsyncMock()
    
    # Mock operaciones básicas
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.setex = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=0)
    client.ttl = AsyncMock(return_value=300)
    client.scan_iter = AsyncMock(return_value=iter([]))
    client.expire = AsyncMock(return_value=True)
    client.close = AsyncMock()
    
    return client


@pytest.fixture
def mock_whatsapp_subprocess():
    """Mock para subprocess de WhatsApp Node.js."""
    process_mock = MagicMock()
    
    # Mock del proceso
    process_mock.poll.return_value = None  # Proceso corriendo
    process_mock.stdin.write = MagicMock()
    process_mock.stdin.flush = MagicMock()
    process_mock.stdout.readline = MagicMock(return_value="")
    process_mock.terminate = MagicMock()
    process_mock.kill = MagicMock()
    process_mock.wait = MagicMock(return_value=0)
    
    return process_mock


class AsyncIterator:
    """Helper para crear async iterators en tests."""
    
    def __init__(self, seq):
        self.iter = iter(seq)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration


# Fixtures para servicios mock

@pytest.fixture
async def mock_llm_service():
    """Mock LLM Service para tests."""
    service = AsyncMock()
    
    # Mock clasificación exitosa
    service.classify_intent = AsyncMock(return_value=MessageIntent(
        intent=IntentType.SALES,
        confidence=0.9,
        entities={"product": "test"},
        routing_department="ventas@empresa.com",
        reasoning="Test classification"
    ))
    
    service.generate_response = AsyncMock(return_value="Respuesta de test generada")
    service.get_stats = MagicMock(return_value={
        "total_calls": 10,
        "successful_calls": 9,
        "failed_calls": 1
    })
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return service


@pytest.fixture
async def mock_context_service():
    """Mock Context Service para tests."""
    service = AsyncMock()
    
    service.initialize = AsyncMock(return_value=True)
    service.get_context = AsyncMock(return_value=None)
    service.save_context = AsyncMock(return_value=True)
    service.add_message_to_history = AsyncMock(return_value=True)
    service.update_intent = AsyncMock(return_value=True)
    service.get_conversation_summary = AsyncMock(return_value="Test conversation summary")
    service.clear_session = AsyncMock(return_value=True)
    service.get_stats = MagicMock(return_value={
        "contexts_loaded": 5,
        "contexts_saved": 5,
        "compressions_performed": 1
    })
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    service.close = AsyncMock()
    
    return service


@pytest.fixture
async def mock_whatsapp_service():
    """Mock WhatsApp Service para tests."""
    service = AsyncMock()
    
    service.start = AsyncMock(return_value=True)
    service.stop = AsyncMock()
    service.send_message = AsyncMock(return_value=True)
    service.get_next_message = AsyncMock(return_value=None)
    service.register_event_handler = MagicMock()
    service.is_connected = True
    service.get_status = AsyncMock(return_value={
        "connection_status": {"is_connected": True},
        "statistics": {"messages_sent": 5, "messages_received": 10}
    })
    
    return service


# Helpers para tests

def assert_intent_classification(result, expected_intent: IntentType, min_confidence: float = 0.5):
    """Helper para verificar clasificaciones de intención."""
    assert isinstance(result, MessageIntent)
    assert result.intent == expected_intent
    assert result.confidence >= min_confidence
    assert result.routing_department is not None
    assert result.reasoning is not None


def create_test_message(body: str, message_type: MessageType = MessageType.TEXT) -> WhatsAppMessage:
    """Helper para crear mensajes de test."""
    return WhatsAppMessage(
        id=f"test_msg_{hash(body)}",
        from_number="1234567890@c.us",
        to_number="chatbot@c.us",
        body=body,
        message_type=message_type,
        timestamp=datetime.now(),
        status=MessageStatus.RECEIVED
    )


# Pytest configuration

def pytest_configure(config):
    """Configuración global de pytest."""
    # Añadir markers personalizados
    config.addinivalue_line("markers", "integration: marca tests de integración")
    config.addinivalue_line("markers", "slow: marca tests lentos")
    config.addinivalue_line("markers", "external: marca tests que requieren servicios externos")


@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para tests async."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mocks para bibliotecas externas

@pytest.fixture(autouse=True) 
def mock_external_calls():
    """Mock automático para llamadas externas en tests."""
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('openai.AsyncOpenAI') as mock_openai, \
         patch('redis.asyncio.Redis') as mock_redis:
        
        # Configurar mocks básicos
        mock_popen.return_value = MagicMock()
        mock_openai.return_value = AsyncMock()
        mock_redis.return_value = AsyncMock()
        
        yield {
            'popen': mock_popen,
            'openai': mock_openai, 
            'redis': mock_redis
        }