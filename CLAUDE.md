### 🔄 Project Awareness & Context
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's chatbot architecture, LangChain patterns, and WhatsApp integration constraints.
- **Check `TASK.md`** before starting a new task. If the task isn't listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md` for the WhatsApp chatbot system.
- **Use Python virtual environment** whenever executing Python commands, including for unit tests with pytest.

### 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For the WhatsApp chatbot system this looks like:
    - `src/agents/` - LangChain agents (intent_classifier.py, conversation_agent.py, routing_agent.py)
    - `src/chains/` - LangChain chains (intent_chain.py, response_chain.py, context_chain.py)
    - `src/services/` - Integration services (whatsapp_service.py, llm_service.py, context_service.py)
    - `src/models/` - Pydantic models (intents.py, messages.py, routing.py)
    - `src/utils/` - Utilities (validators.py, formatters.py, logger.py)
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python-dotenv and load_dotenv()** for environment variables.

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (agents, chains, services, API routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least for each component:
    - 1 test for expected use (successful intent classification, message processing)
    - 1 edge case (malformed WhatsApp messages, network timeouts)
    - 1 failure case (OpenAI API errors, WhatsApp disconnection)
  - **Mock external services**: WhatsApp Web.js, OpenAI API, Redis connections
  - **Test async functions properly** using @pytest.mark.asyncio

### ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a "Discovered During Work" section.

### 📎 Style & Conventions
- **Use Python** as the primary language for the chatbot backend.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation** especially for WhatsApp messages, intents, and routing models.
- **Use `FastAPI` for the API endpoints** (webhooks, health checks) and **Redis** for session storage.
- **Use `LangChain/LangGraph`** for all AI agent implementations and conversation workflows.
- **Use `OpenAI`** for LLM calls with structured output via function calling.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or WhatsApp/OpenAI setup steps are modified.
- **Comment non-obvious code** especially LangChain agent configurations, WhatsApp session management, and OpenAI prompt engineering.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.
- **Document all environment variables** in .env.example with clear descriptions.
- **Include troubleshooting sections** for common WhatsApp Web.js and OpenAI API issues.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use verified packages: langchain, langgraph, openai, fastapi, pydantic, redis.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
- **For WhatsApp integration**: Always implement proper session persistence and reconnection logic.
- **For OpenAI calls**: Always include rate limiting, error handling, and structured output validation.
- **For LangChain agents**: Always configure max_iterations and implement fallback logic.

### 🚨 Critical WhatsApp Chatbot Constraints
- **Session Persistence**: WhatsApp Web.js MUST have persistent session storage or bot disconnects constantly
- **Rate Limiting**: MUST implement per-user rate limiting to avoid WhatsApp bans
- **Error Recovery**: MUST handle WhatsApp disconnections gracefully with automatic reconnection
- **OpenAI Limits**: MUST implement exponential backoff for OpenAI API rate limits
- **Render Constraints**: MUST use background tasks for processing > 30 seconds
- **Memory Management**: MUST compress long conversation history to avoid context window limits
- **Health Checks**: MUST include WhatsApp connection status in /health endpoint