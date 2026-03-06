# Frequently Asked Questions About Chalie AI

Find answers to common questions about Chalie. See also [Quick Start](01-QUICK-START.md) and [Architecture](04-ARCHITECTURE.md).

## General

**What is Chalie?**  
Chalie is an open-source, local-first autonomous AI agent framework designed for developers. It enables building intelligent agents that can interact with tools, manage memory, and execute complex workflows autonomously.

**Is Chalie free to use?**  
Yes, Chalie is completely free and open-source under the MIT license. There are no subscription fees or usage limits when running it locally on your own infrastructure.

**Can Chalie run offline?**  
Chalie can operate fully offline once models are downloaded, though initial setup requires internet access for dependencies. All agent operations, memory storage, and tool execution happen locally without external connectivity requirements.

**What are the system requirements?**  
Chalie requires Node.js 18+, approximately 4GB RAM minimum (8GB recommended), and sufficient disk space for model downloads. A modern CPU is required; GPU acceleration is optional but improves inference speed significantly.

**Is Chalie production-ready?**  
Chalie is suitable for development, testing, and staging environments with active community support. Production deployments should implement additional monitoring, backup strategies, and security hardening based on your specific use case requirements.

## Installation

**How do I install Chalie?**  
Install Chalie globally using `npm i -g chalie` or clone the repository and run `npm install && npm start`. The quick start guide provides detailed installation steps for different operating systems and deployment scenarios.

**Do I need Docker to run Chalie?**  
Docker is optional but recommended for isolated deployments and production environments. You can run Chalie directly with Node.js if you prefer native execution without containerization overhead.

**Does Chalie work on Windows?**  
Yes, Chalie supports Windows 10/11 fully through WSL2 or native Node.js installation. Some Docker-based features may require additional configuration depending on your specific Windows setup and hardware.

**Can I run Chalie on Raspberry Pi?**  
Chalie runs on Raspberry Pi 4 (4GB+) with ARM64 support, though performance depends on model size and complexity. Consider using smaller models or cloud inference for resource-constrained edge devices like single-board computers.

## LLM Providers

**Which LLM providers are supported?**  
Chalie supports OpenAI, Anthropic Claude, Google Gemini, Ollama (local), LM Studio, and any OpenAI-compatible API endpoint. Provider configuration is flexible through environment variables or the web interface settings panel.

**Can I use multiple providers simultaneously?**  
Yes, Chalie allows configuring multiple LLM providers with automatic fallback capabilities. You can set primary and backup providers to ensure reliability during service outages or rate limit scenarios.

**What are the best models for agent tasks?**  
For complex reasoning tasks, GPT-4o, Claude 3.5 Sonnet, or local Llama 3 70B work well. Simpler tasks can use smaller models like Llama 3 8B via Ollama to reduce costs and improve response times significantly.

**How much does it cost to run Chalie?**  
Costs vary by provider: OpenAI charges per token, while local models have zero inference costs after hardware investment. Typical agent operations range from $0.01-$0.50 per hour depending on model choice and usage intensity.

## Memory & Privacy

**How does Chalie's memory system work?**  
Chalie uses a hierarchical memory architecture with short-term context, long-term vector storage, and persistent knowledge graphs. The system automatically manages relevance scoring and retrieval for optimal context utilization during conversations.

**Where is my data stored?**  
All data stores locally in your project directory under `~/.chalie/` by default with SQLite databases and JSON files. No data leaves your machine unless you explicitly configure cloud sync or external API integrations.

**Does Chalie collect telemetry or analytics?**  
Chalie collects zero telemetry by design—no usage tracking, no analytics, no phone home calls. The framework operates entirely locally respecting user privacy and data sovereignty principles throughout all operations.

**Can I delete my conversation history?**  
Yes, you can delete individual conversations through the web interface or clear all data via `chalie reset` command. Memory persistence is fully under your control with granular deletion options for specific agents or time periods.

## Capabilities

**How autonomous are Chalie agents?**  
Chalie agents operate with high autonomy levels, capable of multi-step planning, tool invocation, and self-correction without human intervention. Autonomy depth depends on prompt engineering and available tool configurations in your deployment.

**What is the router system?**  
The router intelligently distributes tasks across specialized worker processes based on capability matching and load balancing. This architecture enables parallel processing while maintaining coherent conversation state across distributed operations.

**Can I create custom tools for Chalie?**  
Yes, Chalie's extensible tooling framework supports creating custom Node.js modules with defined schemas and capabilities. The Tools System guide provides templates and examples for building domain-specific agent functions efficiently.

**Does Chalie support voice interactions?**  
Chalie integrates with speech-to-text and text-to-speech services through configurable tools like ElevenLabs or local Whisper models. Voice capabilities require additional setup but enable fully conversational multimodal agent experiences.

## Development

**How do I run tests for Chalie?**  
Run the test suite using `npm test` which executes Jest unit tests, integration tests, and E2E scenarios covering core functionality. Tests validate tool execution, memory operations, provider integrations, and workflow orchestration comprehensively.

**Can I extend Chalie with custom tools?**  
Absolutely—Chalie's modular architecture supports creating custom tools as Node.js packages following the defined interface patterns. Custom tools integrate seamlessly with the agent's planning system and can access local resources securely.

**How do I contribute to Chalie development?**  
Contribute by forking the repository, implementing features or fixes, submitting PRs with tests, and documenting changes in CHANGELOG.md. The project welcomes contributions across code, documentation, tool creation, and community support channels.

---
## Related Documentation
- [Quick Start](01-QUICK-START.md)
- [Architecture](04-ARCHITECTURE.md)
- [Tools System](09-TOOLS.md)
