"""
WhatsApp Service - Integración con WhatsApp Web.js

Servicio principal para comunicación con WhatsApp usando whatsapp-web.js
como subprocess Node.js. Maneja sesiones persistentes, rate limiting,
y reconexión automática.

Basado en: examples/whatsapp_integration/whatsapp_wrapper.py
"""

import asyncio
import json
import subprocess
import time
import logging
import os
import signal
from typing import Dict, Optional, Callable, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager

from ..models.messages import (
    WhatsAppMessage, OutgoingMessage, MessageType, MessageStatus,
    MessageQueue
)
from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ConnectionStatus:
    """Estado de conexión de WhatsApp."""
    is_connected: bool = False
    last_connected: Optional[datetime] = None
    reconnection_attempts: int = 0
    session_valid: bool = False
    qr_code_required: bool = False
    last_error: Optional[str] = None


class RateLimiter:
    """
    Rate limiter para mensajes WhatsApp.
    
    WhatsApp es muy estricto con rate limiting:
    - Más de 1 mensaje por 1.5 segundos puede causar ban temporal
    - Violaciones consistentes llevan a ban permanente
    """
    
    def __init__(self, min_interval: float = None):
        self.min_interval = min_interval or settings.WHATSAPP_MESSAGE_DELAY
        self.last_send_time = 0
        self.send_count = 0
        self.reset_time = time.time()
        
    async def wait_if_needed(self):
        """Espera si es necesario para respetar rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_send_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.info(f"Rate limiting: esperando {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.last_send_time = time.time()
        self.send_count += 1
        
        # Reset contador cada hora
        if current_time - self.reset_time > 3600:
            self.send_count = 0
            self.reset_time = current_time
            
        logger.debug(f"Mensajes enviados esta hora: {self.send_count}")


class WhatsAppService:
    """
    Servicio principal para integración WhatsApp Web.js.
    
    Features:
    - Session persistence automática (sobrevive reinicios)
    - Automatic reconnection con exponential backoff
    - Rate limiting integrado
    - Queue management para mensajes
    - Event-driven architecture
    - Health monitoring
    """
    
    def __init__(
        self,
        session_path: str = None,
        headless: bool = None,
        max_reconnection_attempts: int = None
    ):
        self.session_path = Path(session_path or settings.WHATSAPP_SESSION_PATH)
        self.headless = headless if headless is not None else settings.WHATSAPP_HEADLESS
        self.max_reconnection_attempts = max_reconnection_attempts or settings.MAX_WHATSAPP_RECONNECTION_ATTEMPTS
        
        # Estado de conexión
        self.status = ConnectionStatus()
        self.process: Optional[subprocess.Popen] = None
        self.rate_limiter = RateLimiter()
        
        # Queue de mensajes
        self.message_queue = MessageQueue(queue_id=f"whatsapp_{int(time.time())}")
        
        # Event handlers
        self.event_handlers: Dict[str, Callable] = {}
        
        # Monitoring
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_uptime": 0,
            "last_heartbeat": datetime.now()
        }
        
        # Ensure session directory exists
        self.session_path.mkdir(exist_ok=True, parents=True)
        
        # Crear script Node.js si no existe
        self._create_whatsapp_bot_script()
    
    def _create_whatsapp_bot_script(self):
        """
        Crea el script Node.js para WhatsApp Web.js integration.
        
        Este script corre como subprocess y comunica via JSON messages.
        CRÍTICO: Session persistence se maneja aquí.
        """
        
        bot_script = '''
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

// CRITICAL: LocalAuth con path persistente
const client = new Client({
    authStrategy: new LocalAuth({
        clientId: "whatsapp-chatbot-v1",
        dataPath: process.argv[2] || "./session"
    }),
    headless: process.argv[3] === "true",
    puppeteer: {
        headless: process.argv[3] === "true",
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox', 
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--single-process',
            '--disable-gpu',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor'
        ]
    }
});

// Event: QR Code para autenticación
client.on('qr', (qr) => {
    console.log('QR_CODE_REQUIRED');
    qrcode.generate(qr, {small: true});
    
    process.stdout.write(JSON.stringify({
        event: 'qr_code',
        data: { qr: qr, timestamp: new Date().toISOString() }
    }) + '\\n');
});

// Event: Cliente listo
client.on('ready', () => {
    console.log('✅ WhatsApp client ready!');
    
    process.stdout.write(JSON.stringify({
        event: 'ready',
        data: { 
            timestamp: new Date().toISOString(),
            phone_number: client.info?.wid?.user || 'unknown'
        }
    }) + '\\n');
});

// Event: Autenticación exitosa
client.on('authenticated', () => {
    console.log('✅ WhatsApp authenticated');
    
    process.stdout.write(JSON.stringify({
        event: 'authenticated',
        data: { timestamp: new Date().toISOString() }
    }) + '\\n');
});

// Event: Fallo de autenticación
client.on('auth_failure', (msg) => {
    console.error('❌ Authentication failed:', msg);
    
    process.stdout.write(JSON.stringify({
        event: 'auth_failure',
        data: { 
            message: msg, 
            timestamp: new Date().toISOString() 
        }
    }) + '\\n');
});

// Event: Desconexión
client.on('disconnected', (reason) => {
    console.log('⚠️ WhatsApp disconnected:', reason);
    
    process.stdout.write(JSON.stringify({
        event: 'disconnected',
        data: { 
            reason: reason, 
            timestamp: new Date().toISOString() 
        }
    }) + '\\n');
});

// Event: Mensaje recibido
client.on('message_create', async (message) => {
    // Solo procesar mensajes entrantes (no nuestros)
    if (message.fromMe) return;
    
    try {
        const contact = await message.getContact();
        const chat = await message.getChat();
        
        process.stdout.write(JSON.stringify({
            event: 'message_received',
            data: {
                id: message.id._serialized,
                from: message.from,
                to: message.to,
                body: message.body,
                type: message.type,
                timestamp: new Date(message.timestamp * 1000).toISOString(),
                isFromMe: message.fromMe,
                contact_name: contact.name || contact.pushname || 'Unknown',
                chat_name: chat.name || 'Individual',
                has_media: message.hasMedia,
                media_type: message.hasMedia ? message.type : null
            }
        }) + '\\n');
        
    } catch (error) {
        console.error('Error processing message:', error);
        
        process.stdout.write(JSON.stringify({
            event: 'message_error',
            data: {
                message_id: message.id._serialized,
                error: error.message,
                timestamp: new Date().toISOString()
            }
        }) + '\\n');
    }
});

// Manejo de comandos desde Python
process.stdin.on('data', async (data) => {
    try {
        const command = JSON.parse(data.toString().trim());
        
        if (command.action === 'send_message') {
            const { to, message, message_type } = command.data;
            
            try {
                let result;
                
                if (message_type === 'text' || !message_type) {
                    result = await client.sendMessage(to, message);
                } else if (message_type === 'media' && command.data.media_path) {
                    const media = MessageMedia.fromFilePath(command.data.media_path);
                    result = await client.sendMessage(to, media, { caption: message });
                }
                
                process.stdout.write(JSON.stringify({
                    event: 'message_sent',
                    data: {
                        to: to,
                        message: message,
                        message_id: result?.id?._serialized || 'unknown',
                        success: true,
                        timestamp: new Date().toISOString()
                    }
                }) + '\\n');
                
            } catch (error) {
                process.stdout.write(JSON.stringify({
                    event: 'message_send_error', 
                    data: {
                        to: to,
                        message: message,
                        error: error.message,
                        timestamp: new Date().toISOString()
                    }
                }) + '\\n');
            }
            
        } else if (command.action === 'get_status') {
            const state = await client.getState();
            
            process.stdout.write(JSON.stringify({
                event: 'status_response',
                data: {
                    state: state,
                    is_connected: client.info !== null,
                    phone_number: client.info?.wid?.user || null,
                    timestamp: new Date().toISOString()
                }
            }) + '\\n');
            
        } else if (command.action === 'heartbeat') {
            process.stdout.write(JSON.stringify({
                event: 'heartbeat_response',
                data: {
                    status: 'alive',
                    timestamp: new Date().toISOString()
                }
            }) + '\\n');
        }
        
    } catch (error) {
        console.error('Error processing command:', error);
        
        process.stdout.write(JSON.stringify({
            event: 'command_error',
            data: {
                error: error.message,
                timestamp: new Date().toISOString()
            }
        }) + '\\n');
    }
});

// Graceful shutdown
const shutdown = async () => {
    console.log('🛑 Shutting down WhatsApp client...');
    try {
        await client.destroy();
        process.exit(0);
    } catch (error) {
        console.error('Error during shutdown:', error);
        process.exit(1);
    }
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Inicializar cliente
console.log('🚀 Starting WhatsApp client...');
client.initialize();
'''
        
        # Guardar script Node.js
        script_path = self.session_path / "whatsapp-bot.js"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(bot_script)
        
        logger.info(f"✅ WhatsApp bot script creado en {script_path}")
        return script_path
    
    async def start(self, force_start: bool = False) -> bool:
        """
        Inicia el cliente WhatsApp subprocess.
        
        Args:
            force_start: Si True, fuerza el inicio incluso en producción
        
        Returns:
            True si inició exitosamente, False si falló
        """
        from ..utils.config import get_settings
        settings = get_settings()
        
        logger.info("🚀 Iniciando WhatsApp service...")
        
        # Skip WhatsApp startup in production where Node.js is not available
        # UNLESS force_start is True (para QR generation)
        if settings.is_production and not force_start:
            logger.warning("⚠️ WhatsApp service disabled in production environment")
            logger.warning("API endpoints will work but WhatsApp integration is disabled")
            self.status.is_connected = False  # Mark as not connected but don't fail
            return True  # Return True to allow app to start
        
        try:
            script_path = self.session_path / "whatsapp-bot.js"
            
            # Verificar que Node.js esté instalado
            try:
                subprocess.run(['node', '--version'], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                logger.error("❌ Node.js no está instalado")
                return False
            
            # Verificar dependencias Node.js
            package_json_path = self.session_path / "package.json"
            if not package_json_path.exists():
                await self._install_node_dependencies()
            
            # Iniciar subprocess Node.js
            self.process = subprocess.Popen([
                'node',
                str(script_path),
                str(self.session_path),  # Session path
                str(self.headless).lower()  # Headless mode
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.session_path)
            )
            
            # Iniciar monitoring de output
            asyncio.create_task(self._monitor_output())
            
            # Iniciar heartbeat monitoring
            asyncio.create_task(self._heartbeat_monitor())
            
            # Iniciar queue processor
            asyncio.create_task(self._process_message_queue())
            
            logger.info("✅ WhatsApp service iniciado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando WhatsApp service: {e}")
            return False
    
    async def _install_node_dependencies(self):
        """Instala dependencias Node.js necesarias."""
        # Skip Node.js installation in production environments where Node.js is not available
        from ..utils.config import get_settings
        settings = get_settings()
        
        if settings.is_production:
            logger.warning("⚠️ Skipping Node.js dependencies installation in production")
            logger.warning("WhatsApp functionality will be limited without Node.js environment")
            return
            
        logger.info("📦 Instalando dependencias Node.js...")
        
        package_json = {
            "name": "whatsapp-chatbot",
            "version": "1.0.0",
            "dependencies": {
                "whatsapp-web.js": "^1.21.0",
                "qrcode-terminal": "^0.12.0"
            }
        }
        
        # Crear package.json
        package_json_path = self.session_path / "package.json"
        with open(package_json_path, 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Instalar dependencias
        try:
            process = await asyncio.create_subprocess_exec(
                'npm', 'install',
                cwd=str(self.session_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Dependencias Node.js instaladas")
            else:
                logger.error(f"❌ Error instalando dependencias: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando npm install: {e}")
    
    async def _monitor_output(self):
        """
        Monitorea output del subprocess y maneja eventos.
        
        Corre continuamente para procesar mensajes desde Node.js.
        """
        if not self.process:
            return
        
        logger.info("👁️ Iniciando monitoring de WhatsApp output...")
        
        try:
            # Usar asyncio subprocess para non-blocking I/O
            while self.process.poll() is None:
                try:
                    # Leer línea con timeout para evitar bloqueo indefinido
                    line = await asyncio.wait_for(
                        asyncio.to_thread(self.process.stdout.readline),
                        timeout=1.0
                    )
                    
                    if not line:
                        # Si no hay línea, esperar un poco antes de continuar
                        await asyncio.sleep(0.1)
                        continue
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Logging básico para debug - MEJORADO
                    logger.debug(f"WhatsApp output: {line}")
                    
                    if not line.startswith('{'):
                        # Log non-JSON output para debugging
                        if 'error' in line.lower() or 'failed' in line.lower():
                            logger.warning(f"WhatsApp warning/error: {line}")
                        else:
                            logger.debug(f"WhatsApp info: {line}")
                        continue
                    
                    # Parsear eventos JSON
                    try:
                        event_data = json.loads(line)
                        await self._handle_event(event_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {e}, line: {line}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Timeout es normal, continuar monitoring
                    continue
                except Exception as e:
                    logger.error(f"❌ Error monitoring output: {e}")
                    # No romper el loop por errores menores
                    await asyncio.sleep(0.5)
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error crítico en monitoring: {e}")
        finally:
            logger.warning("⚠️ WhatsApp output monitoring terminado")
            
            # Verificar si el proceso terminó con error
            if self.process and self.process.poll() is not None:
                return_code = self.process.returncode
                if return_code != 0:
                    logger.error(f"❌ Proceso WhatsApp terminó con código: {return_code}")
                    
                    # Leer stderr para obtener más información
                    try:
                        stderr_output = self.process.stderr.read()
                        if stderr_output:
                            logger.error(f"❌ WhatsApp stderr: {stderr_output}")
                    except:
                        pass
    
    async def _handle_event(self, event_data: Dict[str, Any]):
        """Maneja eventos desde WhatsApp client."""
        event_type = event_data.get('event')
        data = event_data.get('data', {})
        
        logger.debug(f"📥 WhatsApp event: {event_type}")
        
        if event_type == 'ready':
            self.status.is_connected = True
            self.status.last_connected = datetime.now()
            self.status.session_valid = True
            self.status.reconnection_attempts = 0
            self.status.qr_code_required = False
            self.stats["last_heartbeat"] = datetime.now()
            logger.info("✅ WhatsApp client listo para usar!")
            
        elif event_type == 'qr_code':
            self.status.qr_code_required = True
            logger.info("📱 QR Code requerido - escanea con tu WhatsApp")
            
        elif event_type == 'authenticated':
            logger.info("🔐 WhatsApp autenticado exitosamente")
            
        elif event_type == 'auth_failure':
            self.status.last_error = data.get('message', 'Auth failed')
            logger.error(f"❌ Autenticación falló: {self.status.last_error}")
            
        elif event_type == 'disconnected':
            self.status.is_connected = False
            reason = data.get('reason', 'Unknown')
            self.status.last_error = f"Disconnected: {reason}"
            logger.warning(f"⚠️ WhatsApp desconectado: {reason}")
            
            # Programar reconexión
            asyncio.create_task(self._schedule_reconnection())
            
        elif event_type == 'message_received':
            await self._handle_incoming_message(data)
            
        elif event_type == 'message_sent':
            logger.info(f"✅ Mensaje enviado a {data.get('to')}")
            self.stats["messages_sent"] += 1
            
        elif event_type == 'message_send_error':
            logger.error(f"❌ Error enviando a {data.get('to')}: {data.get('error')}")
            
        elif event_type == 'heartbeat_response':
            self.stats["last_heartbeat"] = datetime.now()
            
        # Llamar event handlers personalizados
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"❌ Error en event handler {event_type}: {e}")
    
    async def _handle_incoming_message(self, data: Dict[str, Any]):
        """Procesa mensaje entrante y lo añade a la queue."""
        try:
            message = WhatsAppMessage(
                id=data['id'],
                from_number=data['from'],
                to_number=data['to'],
                body=data['body'],
                message_type=MessageType(data.get('type', 'text')),
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                is_from_me=data.get('isFromMe', False),
                status=MessageStatus.RECEIVED,
                metadata={
                    'contact_name': data.get('contact_name'),
                    'chat_name': data.get('chat_name'),
                    'has_media': data.get('has_media', False),
                    'media_type': data.get('media_type')
                }
            )
            
            # Añadir a queue para procesamiento
            self.message_queue.add_incoming(message)
            self.stats["messages_received"] += 1
            
            logger.info(f"📨 Mensaje recibido de {message.from_number}: {message.body[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error procesando mensaje entrante: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor de heartbeat para verificar que el proceso esté vivo."""
        while True:
            try:
                await asyncio.sleep(30)  # Check cada 30 segundos
                
                if self.status.is_connected and self.process:
                    await self._send_command({
                        "action": "heartbeat",
                        "data": {"timestamp": datetime.now().isoformat()}
                    })
                    
                    # Verificar si el último heartbeat es muy viejo
                    if datetime.now() - self.stats["last_heartbeat"] > timedelta(minutes=2):
                        logger.warning("⚠️ Heartbeat perdido - posible problema de conexión")
                        
            except Exception as e:
                logger.error(f"❌ Error en heartbeat monitor: {e}")
    
    async def _process_message_queue(self):
        """Procesa cola de mensajes salientes."""
        while True:
            try:
                message = self.message_queue.get_next_outgoing()
                if message:
                    await self._send_message_internal(message)
                else:
                    await asyncio.sleep(0.1)  # Breve pausa si no hay mensajes
                    
            except Exception as e:
                logger.error(f"❌ Error procesando queue: {e}")
                await asyncio.sleep(1)  # Pausa más larga en caso de error
    
    async def send_message(self, to_number: str, message: str, message_type: MessageType = MessageType.TEXT) -> bool:
        """
        Envía mensaje via WhatsApp con rate limiting automático.
        
        Args:
            to_number: Número destino WhatsApp
            message: Texto del mensaje
            message_type: Tipo de mensaje
            
        Returns:
            True si fue encolado exitosamente
        """
        if not self.status.is_connected:
            logger.error("❌ No se puede enviar mensaje: WhatsApp no conectado")
            return False
        
        try:
            outgoing_message = OutgoingMessage(
                to_number=to_number,
                body=message,
                message_type=message_type
            )
            
            self.message_queue.add_outgoing(outgoing_message)
            logger.info(f"📤 Mensaje encolado para {to_number}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error encolando mensaje: {e}")
            return False
    
    async def _send_message_internal(self, message: OutgoingMessage):
        """Envía mensaje internamente respetando rate limits."""
        try:
            # Aplicar rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Formatear número si es necesario
            to_number = message.to_number
            if not to_number.endswith('@c.us') and not to_number.endswith('@g.us'):
                to_number = f"{to_number}@c.us"
            
            # Enviar comando a Node.js
            command = {
                "action": "send_message",
                "data": {
                    "to": to_number,
                    "message": message.body,
                    "message_type": message.message_type.value
                }
            }
            
            await self._send_command(command)
            self.message_queue.mark_message_sent()
            
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje interno: {e}")
            # Incrementar retry attempts
            message.retry_attempts += 1
            if message.retry_attempts < message.max_retries:
                # Re-encolar para retry
                await asyncio.sleep(5)  # Wait antes de retry
                self.message_queue.add_outgoing(message)
    
    async def _send_command(self, command: Dict[str, Any]):
        """Envía comando al proceso Node.js."""
        if not self.process or not self.process.stdin:
            raise Exception("Proceso WhatsApp no disponible")
        
        try:
            command_json = json.dumps(command) + '\n'
            self.process.stdin.write(command_json)
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"❌ Error enviando comando: {e}")
            raise
    
    async def _schedule_reconnection(self):
        """Programa reconexión con exponential backoff."""
        if self.status.reconnection_attempts >= self.max_reconnection_attempts:
            logger.error("❌ Máximo número de intentos de reconexión alcanzado")
            return
        
        # Exponential backoff: 5s, 10s, 20s, 40s, 80s
        backoff_delay = min(5 * (2 ** self.status.reconnection_attempts), 300)
        
        logger.info(f"🔄 Programando reconexión en {backoff_delay}s (intento {self.status.reconnection_attempts + 1})")
        
        await asyncio.sleep(backoff_delay)
        
        self.status.reconnection_attempts += 1
        await self.restart()
    
    async def get_next_message(self) -> Optional[WhatsAppMessage]:
        """
        Obtiene próximo mensaje de la queue.
        
        Returns:
            Mensaje o None si no hay mensajes
        """
        try:
            if self.message_queue.incoming_messages:
                return self.message_queue.incoming_messages.pop(0)
            return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo mensaje: {e}")
            return None
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Registra handler personalizado para eventos."""
        self.event_handlers[event_type] = handler
        logger.info(f"📝 Handler registrado para evento: {event_type}")
    
    async def start_for_qr_generation(self) -> bool:
        """
        Inicia WhatsApp service específicamente para generar QR codes.
        
        Este método funciona incluso en producción y está optimizado
        para generar QR codes reales de WhatsApp Web.js.
        
        Returns:
            True si se inició exitosamente para QR generation
        """
        logger.info("📱 Iniciando WhatsApp service para QR generation...")
        
        try:
            # Verificar si Node.js está disponible
            try:
                result = subprocess.run(['node', '--version'], 
                                      check=True, capture_output=True, text=True)
                node_version = result.stdout.strip()
                logger.info(f"✅ Node.js detectado: {node_version}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("❌ Node.js no está disponible - no se puede generar QR real")
                return False
            
            # Verificar/instalar dependencias Node.js
            package_json_path = self.session_path / "package.json"
            if not package_json_path.exists():
                logger.info("📦 Instalando dependencias Node.js para QR generation...")
                await self._install_node_dependencies_for_qr()
            
            # Forzar inicio del servicio para QR generation
            return await self.start(force_start=True)
            
        except Exception as e:
            logger.error(f"❌ Error iniciando WhatsApp para QR generation: {e}")
            return False
    
    async def _install_node_dependencies_for_qr(self):
        """Instala dependencias Node.js específicamente para QR generation."""
        logger.info("📦 Instalando dependencias Node.js para QR generation...")
        
        package_json = {
            "name": "whatsapp-chatbot-qr",
            "version": "1.0.0",
            "description": "WhatsApp QR Code Generator",
            "dependencies": {
                "whatsapp-web.js": "^1.21.0",
                "qrcode-terminal": "^0.12.0"
            },
            "engines": {
                "node": ">=16.0.0"
            }
        }
        
        # Crear package.json
        package_json_path = self.session_path / "package.json"
        with open(package_json_path, 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Instalar dependencias con timeout extendido
        try:
            logger.info("🔄 Ejecutando npm install...")
            process = await asyncio.create_subprocess_exec(
                'npm', 'install', '--production', '--no-audit', '--no-fund',
                cwd=str(self.session_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Timeout de 5 minutos para npm install
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
                
                if process.returncode == 0:
                    logger.info("✅ Dependencias Node.js instaladas para QR generation")
                    return True
                else:
                    logger.error(f"❌ Error instalando dependencias: {stderr.decode()}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.error("❌ Timeout instalando dependencias Node.js")
                process.kill()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando npm install: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado detallado del servicio."""
        return {
            "connection_status": {
                "is_connected": self.status.is_connected,
                "last_connected": self.status.last_connected,
                "reconnection_attempts": self.status.reconnection_attempts,
                "session_valid": self.status.session_valid,
                "qr_code_required": self.status.qr_code_required,
                "last_error": self.status.last_error
            },
            "statistics": self.stats,
            "queue_status": {
                "incoming_count": len(self.message_queue.incoming_messages),
                "outgoing_count": len(self.message_queue.outgoing_messages),
                "processing_active": self.message_queue.processing_active
            },
            "rate_limiting": {
                "messages_sent_this_hour": self.rate_limiter.send_count,
                "last_send_time": self.rate_limiter.last_send_time
            }
        }
    
    async def restart(self):
        """Reinicia el cliente WhatsApp."""
        logger.info("🔄 Reiniciando WhatsApp client...")
        
        await self.stop()
        await asyncio.sleep(2)  # Breve pausa
        await self.start()
    
    async def stop(self):
        """Detiene el servicio WhatsApp gracefully."""
        logger.info("🛑 Deteniendo WhatsApp service...")
        
        if self.process:
            try:
                # Enviar SIGTERM para shutdown graceful
                self.process.terminate()
                
                # Esperar hasta 10 segundos para shutdown graceful
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    # Force kill si es necesario
                    self.process.kill()
                    
            except Exception as e:
                logger.error(f"❌ Error deteniendo proceso WhatsApp: {e}")
            finally:
                self.process = None
        
        self.status.is_connected = False
        logger.info("✅ WhatsApp service detenido")
    
    @property
    def is_connected(self) -> bool:
        """Verifica si WhatsApp está conectado."""
        return self.status.is_connected
    
    @asynccontextmanager
    async def lifespan_context(self):
        """Context manager para lifecycle del servicio."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
