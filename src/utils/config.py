"""
Configuración centralizada del sistema usando Pydantic Settings.

Maneja variables de entorno, validación y configuración por defecto
para todos los servicios del chatbot WhatsApp.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Configuración centralizada del sistema.
    
    Carga automáticamente desde variables de entorno y .env file.
    """
    
    # ================================
    # OpenAI Configuration
    # ================================
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4-1106-preview", description="Modelo OpenAI a usar")
    OPENAI_MAX_TOKENS: int = Field(default=1000, description="Tokens máximos por request")
    OPENAI_TEMPERATURE: float = Field(default=0.1, description="Temperature para OpenAI")
    
    # ================================
    # WhatsApp Configuration  
    # ================================
    WHATSAPP_SESSION_PATH: str = Field(default="./session", description="Path para sesión WhatsApp")
    WHATSAPP_HEADLESS: bool = Field(default=True, description="Modo headless para WhatsApp")
    WHATSAPP_MESSAGE_DELAY: float = Field(default=1.5, description="Delay entre mensajes")
    WHATSAPP_BROWSER_ARGS: str = Field(
        default="--no-sandbox,--disable-setuid-sandbox,--disable-dev-shm-usage",
        description="Argumentos del browser"
    )
    
    # ================================
    # Redis Configuration
    # ================================
    REDIS_URL: str = Field(default="redis://localhost:6379", description="URL de Redis")
    REDIS_SESSION_TTL: int = Field(default=7200, description="TTL para sesiones en segundos")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, description="Conexiones máximas a Redis")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True, description="Retry en timeout")
    
    # ================================
    # Rate Limiting
    # ================================
    RATE_LIMIT_PER_USER: int = Field(default=10, description="Límite por usuario")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Ventana de tiempo en segundos")
    
    # ================================
    # FastAPI Configuration
    # ================================
    PORT: int = Field(default=8000, description="Puerto del servidor")
    HOST: str = Field(default="0.0.0.0", description="Host del servidor")
    CORS_ORIGINS: str = Field(default="*", description="Orígenes CORS permitidos")
    SECRET_KEY: str = Field(default="change-this-in-production", description="Clave secreta")
    
    # ================================
    # Environment Configuration
    # ================================
    ENVIRONMENT: str = Field(default="development", description="Entorno: development/staging/production")
    LOG_LEVEL: str = Field(default="INFO", description="Nivel de logging")
    BASE_URL: str = Field(default="http://localhost:8000", description="URL base del servicio")
    DEBUG: bool = Field(default=False, description="Modo debug")
    
    # ================================
    # Database Configuration
    # ================================
    DATABASE_URL: str = Field(default="sqlite:///./whatsapp_chatbot.db", description="URL base de datos")
    
    # ================================
    # Render Configuration
    # ================================
    RENDER_EXTERNAL_URL: Optional[str] = Field(default=None, description="URL externa en Render")
    MAX_WORKERS: int = Field(default=4, description="Workers máximos")
    
    # ================================
    # Intent Classification
    # ================================
    INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Threshold de confianza")
    USE_FALLBACK_CLASSIFICATION: bool = Field(default=True, description="Usar fallback classification")
    
    # ================================
    # Department Emails
    # ================================
    SALES_EMAIL: str = Field(default="ventas@empresa.com", description="Email ventas")
    SUPPORT_EMAIL: str = Field(default="soporte@empresa.com", description="Email soporte")
    BILLING_EMAIL: str = Field(default="facturacion@empresa.com", description="Email facturación")
    GENERAL_EMAIL: str = Field(default="info@empresa.com", description="Email general")
    COMPLAINT_EMAIL: str = Field(default="gerencia@empresa.com", description="Email reclamos")
    INFORMATION_EMAIL: str = Field(default="info@empresa.com", description="Email información")
    
    # ================================
    # Health Checks & Monitoring
    # ================================
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="Intervalo health checks")
    HEALTH_CHECK_TIMEOUT: int = Field(default=10, description="Timeout health checks")
    
    # ================================
    # Memory Management
    # ================================
    MAX_CONTEXT_TOKENS: int = Field(default=6000, description="Tokens máximos de contexto")
    MAX_MEMORY_MESSAGES: int = Field(default=20, description="Mensajes máximos en memoria")
    
    # ================================
    # Error Handling
    # ================================
    MAX_API_RETRIES: int = Field(default=3, description="Reintentos máximos API")
    INITIAL_RETRY_DELAY: float = Field(default=1.0, description="Delay inicial retry")
    RETRY_MULTIPLIER: float = Field(default=2.0, description="Multiplicador retry")
    MAX_WHATSAPP_RECONNECTION_ATTEMPTS: int = Field(default=5, description="Reconexiones WhatsApp")
    
    # ================================
    # Security
    # ================================
    WEBHOOK_SECRET: Optional[str] = Field(default=None, description="Secret webhook")
    ALLOWED_IPS: str = Field(default="127.0.0.1,localhost", description="IPs permitidas")
    
    # ================================
    # Testing
    # ================================
    MOCK_EXTERNAL_SERVICES: bool = Field(default=False, description="Mock servicios externos")
    TEST_DATA_PATH: str = Field(default="./tests/data/", description="Path datos test")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @validator('WHATSAPP_SESSION_PATH')
    def validate_session_path(cls, v):
        """Valida y crea directorio de sesión si no existe."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())
    
    @validator('CORS_ORIGINS')
    def validate_cors_origins(cls, v):
        """Convierte CORS origins de string a lista."""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]
    
    @validator('ALLOWED_IPS')
    def validate_allowed_ips(cls, v):
        """Convierte IPs permitidas a lista."""
        return [ip.strip() for ip in v.split(",")]
    
    @validator('WHATSAPP_BROWSER_ARGS')
    def validate_browser_args(cls, v):
        """Convierte argumentos del browser a lista."""
        return [arg.strip() for arg in v.split(",")]
    
    @validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v):
        """Valida formato básico de OpenAI API key."""
        if not v.startswith('sk-'):
            raise ValueError('OpenAI API key debe empezar con sk-')
        if len(v) < 20:
            raise ValueError('OpenAI API key parece inválida (muy corta)')
        return v
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Valida que el environment sea válido."""
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment debe ser uno de: {valid_envs}')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Valida nivel de logging."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level debe ser uno de: {valid_levels}')
        return v.upper()
    
    # ================================
    # Computed Properties
    # ================================
    
    @property
    def is_production(self) -> bool:
        """Verifica si está en producción."""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Verifica si está en desarrollo."""
        return self.ENVIRONMENT == "development"
    
    @property
    def department_emails(self) -> Dict[str, str]:
        """Retorna diccionario de emails de departamentos."""
        return {
            "sales": self.SALES_EMAIL,
            "support": self.SUPPORT_EMAIL,
            "billing": self.BILLING_EMAIL,
            "general": self.GENERAL_EMAIL,
            "complaint": self.COMPLAINT_EMAIL,
            "information": self.INFORMATION_EMAIL
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Configuración para conexión Redis."""
        return {
            "url": self.REDIS_URL,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "retry_on_timeout": self.REDIS_RETRY_ON_TIMEOUT,
            "socket_connect_timeout": 5,
            "socket_timeout": 5,
            "health_check_interval": 30
        }
    
    @property
    def openai_config(self) -> Dict[str, Any]:
        """Configuración para OpenAI client."""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.OPENAI_MODEL,
            "max_tokens": self.OPENAI_MAX_TOKENS,
            "temperature": self.OPENAI_TEMPERATURE,
            "timeout": 30.0,
            "max_retries": self.MAX_API_RETRIES
        }
    
    @property
    def whatsapp_config(self) -> Dict[str, Any]:
        """Configuración para WhatsApp service."""
        return {
            "session_path": self.WHATSAPP_SESSION_PATH,
            "headless": self.WHATSAPP_HEADLESS,
            "message_delay": self.WHATSAPP_MESSAGE_DELAY,
            "browser_args": self.WHATSAPP_BROWSER_ARGS,
            "max_reconnection_attempts": self.MAX_WHATSAPP_RECONNECTION_ATTEMPTS
        }
    
    def get_database_url(self) -> str:
        """Obtiene URL de base de datos, ajustando para Render si es necesario."""
        if self.is_production and self.RENDER_EXTERNAL_URL:
            # En Render, usar URL externa para webhooks
            return self.DATABASE_URL.replace("localhost", "0.0.0.0")
        return self.DATABASE_URL
    
    def get_base_url(self) -> str:
        """Obtiene URL base, usando Render URL si está disponible."""
        if self.is_production and self.RENDER_EXTERNAL_URL:
            return self.RENDER_EXTERNAL_URL
        return self.BASE_URL


# Instancia global de configuración
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Obtiene instancia singleton de configuración.
    
    Returns:
        Settings: Configuración del sistema
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Recarga configuración desde archivos de entorno.
    
    Útil para testing o cambios dinámicos de configuración.
    
    Returns:
        Settings: Nueva configuración
    """
    global _settings
    _settings = Settings()
    return _settings


# Para testing - permite inyectar configuración mock
def set_settings_for_testing(test_settings: Settings):
    """
    Establece configuración para testing.
    
    Args:
        test_settings: Configuración de prueba
    """
    global _settings
    _settings = test_settings