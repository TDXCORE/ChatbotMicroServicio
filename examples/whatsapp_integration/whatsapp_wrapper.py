"""
WhatsApp Web.js Async Wrapper - Production Pattern

Este ejemplo muestra cómo integrar WhatsApp Web.js (Node.js) con Python asyncio
para un chatbot de producción.

Key patterns:
- Persistent session management (CRÍTICO para production)
- Async subprocess communication con Node.js
- Rate limiting para evitar WhatsApp bans
- Reconnection logic con exponential backoff
- Message queue management
"""

import asyncio
import json
import subprocess
import time
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WhatsAppMessage:
    """WhatsApp message structure."""
    id: str
    from_number: str
    to_number: str
    body: str
    timestamp: datetime
    message_type: str = "text"


@dataclass 
class ConnectionStatus:
    """WhatsApp connection status tracking."""
    is_connected: bool = False
    last_connected: Optional[datetime] = None
    reconnection_attempts: int = 0
    session_valid: bool = False


class RateLimiter:
    """
    Rate limiter para WhatsApp messages.
    
    WhatsApp is VERY aggressive with rate limiting:
    - More than 1 message per 1.5 seconds can trigger temporary ban
    - Consistent violations lead to permanent ban
    """
    
    def __init__(self, min_interval: float = 1.8):
        self.min_interval = min_interval
        self.last_send_time = 0
        
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_send_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.last_send_time = time.time()


class WhatsAppService:
    """
    Production-ready WhatsApp Web.js integration service.
    
    Features:
    - Persistent session storage (survives restarts)
    - Automatic reconnection con exponential backoff
    - Rate limiting integration
    - Message queue management
    - Event-driven architecture
    """
    
    def __init__(
        self, 
        session_path: str = "./session",
        headless: bool = True,
        max_reconnection_attempts: int = 5
    ):
        self.session_path = Path(session_path)
        self.headless = headless
        self.max_reconnection_attempts = max_reconnection_attempts
        
        # Connection management
        self.status = ConnectionStatus()
        self.process: Optional[subprocess.Popen] = None
        self.rate_limiter = RateLimiter()
        
        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[str, Callable] = {}
        
        # Ensure session directory exists
        self.session_path.mkdir(exist_ok=True)
        
        # Create Node.js WhatsApp bot script
        self._create_whatsapp_bot_script()
    
    def _create_whatsapp_bot_script(self):
        """
        Create Node.js script para WhatsApp Web.js integration.
        
        This script runs as a subprocess y communicates via JSON messages.
        CRÍTICO: Session persistence is handled here.
        """
        
        bot_script = '''
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

// CRITICAL: LocalAuth with persistent session path
const client = new Client({
    authStrategy: new LocalAuth({
        clientId: "whatsapp-chatbot",
        dataPath: process.argv[2] || "./session"  // Session path from Python
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
            '--disable-gpu'
        ]
    }
});

// Event handlers
client.on('qr', (qr) => {
    console.log('QR_CODE_GENERATED');
    qrcode.generate(qr, {small: true});
    
    // Send QR to Python process
    process.stdout.write(JSON.stringify({
        event: 'qr_code',
        data: { qr: qr }
    }) + '\\n');
});

client.on('ready', () => {
    console.log('WhatsApp client is ready!');
    
    process.stdout.write(JSON.stringify({
        event: 'ready',
        data: { timestamp: new Date().toISOString() }
    }) + '\\n');
});

client.on('authenticated', () => {
    console.log('WhatsApp client authenticated');
    
    process.stdout.write(JSON.stringify({
        event: 'authenticated', 
        data: { timestamp: new Date().toISOString() }
    }) + '\\n');
});

client.on('auth_failure', (msg) => {
    console.error('Authentication failed:', msg);
    
    process.stdout.write(JSON.stringify({
        event: 'auth_failure',
        data: { message: msg, timestamp: new Date().toISOString() }
    }) + '\\n');
});

client.on('disconnected', (reason) => {
    console.log('WhatsApp client disconnected:', reason);
    
    process.stdout.write(JSON.stringify({
        event: 'disconnected',
        data: { reason: reason, timestamp: new Date().toISOString() }
    }) + '\\n');
});

// Message handling
client.on('message_create', async (message) => {
    // Send message to Python process
    process.stdout.write(JSON.stringify({
        event: 'message_received',
        data: {
            id: message.id._serialized,
            from: message.from,
            to: message.to,
            body: message.body,
            timestamp: new Date(message.timestamp * 1000).toISOString(),
            isFromMe: message.fromMe,
            type: message.type
        }
    }) + '\\n');
});

// Handle commands from Python
process.stdin.on('data', async (data) => {
    try {
        const command = JSON.parse(data.toString().trim());
        
        if (command.action === 'send_message') {
            const { to, message } = command.data;
            
            try {
                await client.sendMessage(to, message);
                
                process.stdout.write(JSON.stringify({
                    event: 'message_sent',
                    data: { 
                        to: to, 
                        message: message, 
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
        }
        
    } catch (error) {
        console.error('Error processing command:', error);
    }
});

// Graceful shutdown handling
process.on('SIGINT', async () => {
    console.log('Shutting down WhatsApp client...');
    await client.destroy();
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('Terminating WhatsApp client...');
    await client.destroy();
    process.exit(0);
});

// Initialize client
client.initialize();
'''
        
        # Save Node.js script
        script_path = self.session_path / "whatsapp-bot.js"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(bot_script)
        
        return script_path
    
    async def start(self):
        """
        Start WhatsApp client subprocess.
        
        Handles initial authentication flow and establishes connection.
        """
        logger.info("Starting WhatsApp service...")
        
        try:
            # Start Node.js WhatsApp client
            script_path = self.session_path / "whatsapp-bot.js"
            
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
            bufsize=1
            )
            
            # Start output monitoring task
            asyncio.create_task(self._monitor_output())
            
            logger.info("WhatsApp service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start WhatsApp service: {e}")
            raise
    
    async def _monitor_output(self):
        """
        Monitor subprocess output y handle events.
        
        This runs continuously to process messages from Node.js process.
        """
        if not self.process:
            return
        
        while self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Skip non-JSON log lines
                if not line.startswith('{'):
                    logger.info(f"WhatsApp: {line}")
                    continue
                
                # Parse JSON event
                try:
                    event_data = json.loads(line)
                    await self._handle_event(event_data)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON output: {line}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error monitoring output: {e}")
                break
    
    async def _handle_event(self, event_data: Dict[str, Any]):
        """Handle events from WhatsApp client."""
        event_type = event_data.get('event')
        data = event_data.get('data', {})
        
        if event_type == 'ready':
            self.status.is_connected = True
            self.status.last_connected = datetime.now()
            self.status.session_valid = True
            self.status.reconnection_attempts = 0
            logger.info("WhatsApp client ready!")
            
        elif event_type == 'disconnected':
            self.status.is_connected = False
            reason = data.get('reason', 'Unknown')
            logger.warning(f"WhatsApp disconnected: {reason}")
            
            # Schedule reconnection
            await self._schedule_reconnection()
            
        elif event_type == 'message_received':
            # Add message to processing queue
            message = WhatsAppMessage(
                id=data['id'],
                from_number=data['from'],
                to_number=data['to'], 
                body=data['body'],
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                message_type=data.get('type', 'text')
            )
            
            # Only process messages not from us
            if not data.get('isFromMe', False):
                await self.message_queue.put(message)
                logger.info(f"Message queued from {message.from_number}")
        
        # Call registered event handlers
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def _schedule_reconnection(self):
        """
        Schedule reconnection con exponential backoff.
        
        CRÍTICO: Avoid overwhelming WhatsApp servers con rapid reconnection attempts.
        """
        if self.status.reconnection_attempts >= self.max_reconnection_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        # Exponential backoff: 5s, 10s, 20s, 40s, 80s
        backoff_delay = min(5 * (2 ** self.status.reconnection_attempts), 300)
        
        logger.info(f"Scheduling reconnection in {backoff_delay}s (attempt {self.status.reconnection_attempts + 1})")
        
        await asyncio.sleep(backoff_delay)
        
        self.status.reconnection_attempts += 1
        await self.restart()
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """
        Send message via WhatsApp con rate limiting.
        
        Args:
            to_number: WhatsApp number (include country code)
            message: Message text to send
            
        Returns:
            bool: Success status
        """
        if not self.status.is_connected:
            logger.error("Cannot send message: WhatsApp not connected")
            return False
        
        # Apply rate limiting - CRÍTICO para avoid bans
        await self.rate_limiter.wait_if_needed()
        
        try:
            # Format number (ensure includes @c.us suffix for individual chats)
            if not to_number.endswith('@c.us') and not to_number.endswith('@g.us'):
                to_number = f"{to_number}@c.us"
            
            # Send command to Node.js process
            command = {
                "action": "send_message",
                "data": {
                    "to": to_number,
                    "message": message
                }
            }
            
            if self.process and self.process.stdin:
                self.process.stdin.write(json.dumps(command) + '\n')
                self.process.stdin.flush()
                
                logger.info(f"Message sent to {to_number}")
                return True
            else:
                logger.error("WhatsApp process not available")
                return False
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def get_next_message(self) -> Optional[WhatsAppMessage]:
        """Get next message from queue."""
        try:
            # Wait up to 1 second for message
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register custom event handler."""
        self.event_handlers[event_type] = handler
    
    async def restart(self):
        """Restart WhatsApp client."""
        logger.info("Restarting WhatsApp client...")
        
        await self.stop()
        await asyncio.sleep(2)  # Brief pause
        await self.start()
    
    async def stop(self):
        """Stop WhatsApp client gracefully."""
        logger.info("Stopping WhatsApp service...")
        
        if self.process:
            try:
                # Send SIGTERM for graceful shutdown
                self.process.terminate()
                
                # Wait up to 10 seconds for graceful shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait), 
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    # Force kill if needed
                    self.process.kill()
                    
            except Exception as e:
                logger.error(f"Error stopping WhatsApp process: {e}")
            finally:
                self.process = None
        
        self.status.is_connected = False
        logger.info("WhatsApp service stopped")
    
    @property
    def is_connected(self) -> bool:
        """Check if WhatsApp is connected."""
        return self.status.is_connected
    
    @property 
    def connection_info(self) -> Dict[str, Any]:
        """Get detailed connection information."""
        return {
            "is_connected": self.status.is_connected,
            "last_connected": self.status.last_connected,
            "reconnection_attempts": self.status.reconnection_attempts,
            "session_valid": self.status.session_valid,
            "message_queue_size": self.message_queue.qsize()
        }


# Example usage
async def example_usage():
    """
    Example de cómo usar WhatsAppService en un chatbot.
    """
    
    # Initialize service
    whatsapp = WhatsAppService(
        session_path="./session",
        headless=True  # Set to False para development/debugging
    )
    
    # Register custom event handlers
    async def on_ready(data):
        print("✅ WhatsApp client ready for business!")
    
    async def on_disconnected(data):
        print(f"❌ WhatsApp disconnected: {data.get('reason')}")
    
    whatsapp.register_event_handler('ready', on_ready)
    whatsapp.register_event_handler('disconnected', on_disconnected)
    
    try:
        # Start WhatsApp client
        await whatsapp.start()
        
        # Message processing loop
        while True:
            # Check connection status
            if not whatsapp.is_connected:
                print("⏳ Waiting for WhatsApp connection...")
                await asyncio.sleep(5)
                continue
            
            # Process incoming messages
            message = await whatsapp.get_next_message()
            if message:
                print(f"📨 New message from {message.from_number}: {message.body}")
                
                # Send auto-reply (with rate limiting)
                await whatsapp.send_message(
                    message.from_number,
                    "¡Gracias por tu mensaje! Te responderemos pronto."
                )
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy loop
            
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
    finally:
        await whatsapp.stop()


if __name__ == "__main__":
    # Para testing local
    asyncio.run(example_usage())