"""
LangChain Memory Management Chain - WhatsApp Chatbot Pattern

Este ejemplo muestra cómo manejar memoria conversacional para chatbots WhatsApp
usando LangChain memory components con Redis persistence.

Key patterns:
- Redis-backed conversation memory
- Context window management (evitar token limits)  
- Conversation history compression
- Multi-user session management
- Memory cleanup y TTL handling
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import redis.asyncio as redis
from pydantic import BaseModel


class ConversationSummary(BaseModel):
    """Structured summary of conversation history."""
    user_id: str
    session_id: str
    summary: str
    key_points: List[str]
    last_intent: Optional[str] = None
    created_at: datetime
    message_count: int


class WhatsAppMemoryChain:
    """
    Production-ready conversation memory management para WhatsApp chatbots.
    
    Features:
    - Redis-backed persistent memory across restarts
    - Automatic context window management
    - Conversation summarization cuando se acerca token limit
    - Multi-user session isolation
    - Configurable TTL para cleanup automático
    """
    
    def __init__(
        self,
        redis_url: str,
        openai_api_key: str,
        max_token_limit: int = 6000,  # Leave room for prompt + response
        memory_window: int = 20,  # Keep last 20 exchanges
        session_ttl: int = 7200,  # 2 hours default TTL
        model_name: str = "gpt-4-1106-preview"
    ):
        self.redis_url = redis_url
        self.max_token_limit = max_token_limit
        self.memory_window = memory_window
        self.session_ttl = session_ttl
        
        # Initialize OpenAI for summarization
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1
        )
        
        # Redis connection
        self.redis_client = None
        
        # Memory storage for active sessions
        self._active_memories: Dict[str, ConversationBufferWindowMemory] = {}
        
        # Summarization prompt template
        self._setup_summarization_prompt()
    
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        try:
            await self.redis_client.ping()
            print("✅ Redis connection established")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def _setup_summarization_prompt(self):
        """Setup prompt template para conversation summarization."""
        self.summarization_prompt = PromptTemplate(
            input_variables=["conversation_history", "user_id"],
            template="""
Eres un asistente especializado en resumir conversaciones de un chatbot de atención al cliente en WhatsApp.

Tu tarea es crear un resumen conciso y útil de la conversación para mantener contexto sin exceder límites de tokens.

Conversación del usuario {user_id}:
{conversation_history}

Crea un resumen que incluya:
1. Tema principal de la conversación
2. Intención detectada del usuario (ventas, soporte, facturación, etc.)
3. Información clave mencionada (productos, números de cuenta, problemas específicos)
4. Estado actual de la consulta
5. Próximos pasos si los hay

Mantén el resumen breve pero informativo (máximo 200 palabras).

RESUMEN:
"""
        )
    
    def _get_session_key(self, user_id: str, session_id: str = "default") -> str:
        """Generate unique session key para Redis."""
        return f"whatsapp:memory:{user_id}:{session_id}"
    
    def _get_summary_key(self, user_id: str, session_id: str = "default") -> str:
        """Generate summary key para Redis."""
        return f"whatsapp:summary:{user_id}:{session_id}"
    
    async def get_memory(
        self, 
        user_id: str, 
        session_id: str = "default"
    ) -> ConversationBufferWindowMemory:
        """
        Get or create conversation memory for user session.
        
        Args:
            user_id: WhatsApp user identifier
            session_id: Session identifier (default para single ongoing conversation)
            
        Returns:
            ConversationBufferWindowMemory with persistent Redis backing
        """
        
        memory_key = f"{user_id}:{session_id}"
        
        # Return existing memory if already loaded
        if memory_key in self._active_memories:
            return self._active_memories[memory_key]
        
        # Create new memory with Redis backing
        session_key = self._get_session_key(user_id, session_id)
        
        # Initialize Redis-backed chat history
        chat_history = RedisChatMessageHistory(
            session_id=session_key,
            url=self.redis_url,
            key_prefix="whatsapp_msg:",
            ttl=self.session_ttl
        )
        
        # Create memory with window limit
        memory = ConversationBufferWindowMemory(
            k=self.memory_window,
            chat_memory=chat_history,
            return_messages=True,
            memory_key="chat_history",
            input_key="human_input",
            output_key="ai_response"
        )
        
        # Load existing summary if available
        await self._load_conversation_summary(user_id, session_id, memory)
        
        # Cache active memory
        self._active_memories[memory_key] = memory
        
        return memory
    
    async def _load_conversation_summary(
        self, 
        user_id: str, 
        session_id: str,
        memory: ConversationBufferWindowMemory
    ):
        """Load and prepend conversation summary si existe."""
        
        summary_key = self._get_summary_key(user_id, session_id)
        
        try:
            summary_data = await self.redis_client.get(summary_key)
            
            if summary_data:
                summary = ConversationSummary.model_validate_json(summary_data)
                
                # Add summary as system context if recent
                if summary.created_at > datetime.now() - timedelta(hours=4):
                    summary_message = AIMessage(
                        content=f"[RESUMEN CONVERSACIÓN PREVIA]: {summary.summary}"
                    )
                    
                    # Insert summary at beginning of memory
                    memory.chat_memory.messages.insert(0, summary_message)
                    
                    print(f"📋 Loaded conversation summary for {user_id}")
                    
        except Exception as e:
            print(f"⚠️ Error loading summary for {user_id}: {e}")
    
    async def add_message(
        self, 
        user_id: str, 
        human_message: str,
        ai_response: str,
        session_id: str = "default"
    ):
        """
        Add message exchange to conversation memory.
        
        Automatically handles token limits y summarization cuando necesario.
        """
        
        memory = await self.get_memory(user_id, session_id)
        
        # Add messages to memory
        memory.save_context(
            {"human_input": human_message},
            {"ai_response": ai_response}
        )
        
        # Check if we need to compress due to token limit
        await self._check_and_compress_memory(user_id, session_id, memory)
        
        print(f"💾 Saved conversation for {user_id} (messages: {len(memory.chat_memory.messages)})")
    
    async def _check_and_compress_memory(
        self,
        user_id: str,
        session_id: str, 
        memory: ConversationBufferWindowMemory
    ):
        """
        Check token count y compress memory si necessary.
        
        This prevents hitting OpenAI token limits in long conversations.
        """
        
        # Estimate token count (rough approximation)
        total_tokens = self._estimate_token_count(memory.chat_memory.messages)
        
        if total_tokens > self.max_token_limit:
            print(f"🗜️ Compressing memory for {user_id} (estimated tokens: {total_tokens})")
            await self._compress_conversation_memory(user_id, session_id, memory)
    
    def _estimate_token_count(self, messages: List[BaseMessage]) -> int:
        """Rough estimation of token count para messages."""
        total_chars = sum(len(msg.content) for msg in messages)
        # Rough approximation: 1 token ≈ 4 characters para English/Spanish
        return total_chars // 3  # More conservative estimate
    
    async def _compress_conversation_memory(
        self,
        user_id: str,
        session_id: str,
        memory: ConversationBufferWindowMemory
    ):
        """
        Compress conversation by summarizing older messages.
        
        Keeps recent messages intact pero summarizes older context.
        """
        
        messages = memory.chat_memory.messages
        
        if len(messages) < 10:  # Don't compress very short conversations
            return
        
        # Keep last 8 messages, summarize the rest
        messages_to_summarize = messages[:-8]
        messages_to_keep = messages[-8:]
        
        # Create conversation text for summarization
        conversation_text = "\n".join([
            f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
            for msg in messages_to_summarize
        ])
        
        try:
            # Generate summary using LLM
            summary_prompt = self.summarization_prompt.format(
                conversation_history=conversation_text,
                user_id=user_id
            )
            
            summary_response = await self.llm.ainvoke(summary_prompt)
            summary_text = summary_response.content
            
            # Create summary object
            summary = ConversationSummary(
                user_id=user_id,
                session_id=session_id,
                summary=summary_text,
                key_points=self._extract_key_points(summary_text),
                created_at=datetime.now(),
                message_count=len(messages_to_summarize)
            )
            
            # Save summary to Redis
            summary_key = self._get_summary_key(user_id, session_id)
            await self.redis_client.setex(
                summary_key,
                self.session_ttl,
                summary.model_dump_json()
            )
            
            # Replace old messages with summary
            memory.chat_memory.messages = [
                AIMessage(content=f"[RESUMEN]: {summary_text}")
            ] + messages_to_keep
            
            print(f"✅ Compressed {len(messages_to_summarize)} messages into summary for {user_id}")
            
        except Exception as e:
            print(f"❌ Error compressing memory for {user_id}: {e}")
            # Fallback: just keep recent messages
            memory.chat_memory.messages = messages_to_keep
    
    def _extract_key_points(self, summary_text: str) -> List[str]:
        """Extract key points from summary text."""
        # Simple extraction - could be enhanced with NLP
        lines = summary_text.split('\n')
        key_points = [
            line.strip('- ').strip() 
            for line in lines 
            if line.strip().startswith('-') or line.strip().startswith('•')
        ]
        return key_points[:5]  # Limit to 5 key points
    
    async def get_conversation_context(
        self, 
        user_id: str, 
        session_id: str = "default"
    ) -> str:
        """
        Get formatted conversation context para prompts.
        
        Returns recent conversation history as formatted string.
        """
        
        memory = await self.get_memory(user_id, session_id)
        
        if not memory.chat_memory.messages:
            return "No hay historial de conversación previo."
        
        # Format messages para context
        context_lines = []
        for msg in memory.chat_memory.messages[-10:]:  # Last 10 messages
            role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
            context_lines.append(f"{role}: {msg.content}")
        
        return "\n".join(context_lines)
    
    async def clear_session(self, user_id: str, session_id: str = "default"):
        """Clear conversation memory for user session."""
        
        memory_key = f"{user_id}:{session_id}"
        
        # Remove from active memories
        if memory_key in self._active_memories:
            del self._active_memories[memory_key]
        
        # Clear Redis data
        session_key = self._get_session_key(user_id, session_id)
        summary_key = self._get_summary_key(user_id, session_id)
        
        try:
            await self.redis_client.delete(f"whatsapp_msg:{session_key}")
            await self.redis_client.delete(summary_key)
            print(f"🗑️ Cleared session for {user_id}")
        except Exception as e:
            print(f"❌ Error clearing session for {user_id}: {e}")
    
    async def cleanup_expired_sessions(self):
        """
        Background task para cleanup expired sessions.
        
        Should be run periodically (e.g., daily cron job).
        """
        
        try:
            # Find all whatsapp memory keys
            keys = await self.redis_client.keys("whatsapp:*")
            
            expired_count = 0
            for key in keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No TTL set, or expired
                    await self.redis_client.delete(key)
                    expired_count += 1
            
            print(f"🧹 Cleaned up {expired_count} expired sessions")
            
        except Exception as e:
            print(f"❌ Error during session cleanup: {e}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# Example usage
async def example_usage():
    """
    Example de cómo usar WhatsAppMemoryChain en un chatbot.
    """
    
    import os
    
    # Initialize memory chain
    memory_chain = WhatsAppMemoryChain(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_token_limit=6000,
        memory_window=20
    )
    
    await memory_chain.initialize()
    
    try:
        user_id = "whatsapp_1234567890"
        
        # Simulate conversation
        await memory_chain.add_message(
            user_id=user_id,
            human_message="Hola, tengo una pregunta sobre mis facturas",
            ai_response="¡Hola! Estoy aquí para ayudarte con tus consultas de facturación. ¿Cuál es tu pregunta específica?"
        )
        
        await memory_chain.add_message(
            user_id=user_id,
            human_message="No recibí mi factura del mes pasado",
            ai_response="Entiendo tu preocupación. Te voy a ayudar a verificar el estado de tu factura. ¿Podrías proporcionarme tu número de cuenta?"
        )
        
        # Get conversation context
        context = await memory_chain.get_conversation_context(user_id)
        print("📋 Conversation Context:")
        print(context)
        
        # Simulate many more messages to trigger compression
        for i in range(15):
            await memory_chain.add_message(
                user_id=user_id,
                human_message=f"Mensaje de prueba número {i}",
                ai_response=f"Respuesta a mensaje {i}"
            )
        
        # Check if compression happened
        memory = await memory_chain.get_memory(user_id)
        print(f"📊 Messages in memory: {len(memory.chat_memory.messages)}")
        
    finally:
        await memory_chain.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())