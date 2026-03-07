# Tutorial: Your First 10 Minutes with Chalie

Welcome! In this tutorial, you'll go from zero to having your first conversation with Chalie in about 10 minutes. By the end, you'll understand how to install, configure, and start interacting with your personal intelligence layer.

---

## What You'll Learn

- ✅ Installing Chalie on your machine
- ✅ Setting up an LLM provider (we recommend Ollama for local, private use)
- ✅ Creating your account via the web interface
- ✅ Having your first conversation with Chalie

---

## Prerequisites

Before you begin, ensure you have:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.9+ | Required for running Chalie |
| **Git** | Any | For cloning the repository (optional) |
| **Docker** | Optional | Only needed if you want sandboxed tools |

### Check Your Python Version

```bash
python3 --version
# Should output: Python 3.9.x or higher
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

---

## Step 1: Install Chalie (2 minutes)

### Option A: Quick Installer (Recommended for Most Users)

The easiest way to get started is using the official installer script:

```bash
curl -fsSL https://chalie.ai/install | bash
```

This will:
1. Check that Python 3.9+ is available
2. Download and set up Chalie in `~/.chalie/`
3. Install the `chalie` CLI command
4. Take about 2 minutes on a typical connection

### Option B: From Source (For Developers)

If you want to run from source or contribute:

```bash
# Clone the repository
git clone https://github.com/chalie-ai/chalie.git
cd chalie

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

---

## Step 2: Start Chalie (1 minute)

### If You Used the Installer

Simply run:

```bash
chalie
```

Chalie will start and open automatically at **http://localhost:8081** in your browser.

### If You Installed from Source

Run:

```bash
python backend/run.py
```

Or use the launcher script:

```bash
./run.sh
```

The application will be available at **http://localhost:8081**.

> **Tip:** To run on a different port, add `--port=9000` to either command.

---

## Step 3: Set Up an LLM Provider (3 minutes)

Chalie needs an AI model to power conversations. You have two main options:

### Option A: Ollama (Local & Private — Recommended)

Ollama runs models entirely on your machine—no API costs, fully private.

#### 1. Install Ollama

Download from [ollama.ai](https://ollama.ai) and follow the installation instructions for your OS.

#### 2. Pull a Model

Open a terminal and run:

```bash
ollama pull qwen:8b
```

This downloads a capable, efficient model (~5GB). Other options include `mistral`, `llama3`, or `gemma`.

#### 3. Configure in Chalie

1. Open **http://localhost:8081/on-boarding/** in your browser
2. Click **"Configure Provider"** or **"Add Provider"**
3. Select **Ollama** as the platform
4. Fill in the details:
   - **Name**: `ollama-local` (or any name you prefer)
   - **Model**: `qwen:8b`
   - **Host**: `http://localhost:11434`
5. Click **Save**

### Option B: Cloud Provider (OpenAI, Anthropic, or Gemini)

If you prefer cloud models, you'll need an API key:

| Provider | Get Key At | Example Model |
|----------|------------|---------------|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | `gpt-4o` |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | `claude-haiku-3.5` |
| Google Gemini | [ai.google.dev](https://ai.google.dev) | `gemini-2.0-flash` |

In the onboarding page, select your provider and paste your API key when prompted.

---

## Step 4: Create Your Account (1 minute)

After configuring a provider, you'll be redirected to create an account:

1. On **http://localhost:8081/on-boarding/**
2. Enter a password of your choice
3. Click **Create Account**

You're now logged in and ready to chat!

---

## Step 5: Have Your First Conversation (3 minutes)

You should now see the Chalie chat interface. Let's start talking!

### Try These Prompts

Type any of these into the message box and press Enter:

```
Hi, I'm new here. Tell me about yourself.
```

Or ask Chalie to remember something:

```
Remember that my favorite color is blue.
```

Or create a simple task:

```
Add "buy coffee" to my shopping list.
```

### What Happens Behind the Scenes?

When you send a message, Chalie:

1. **Retrieves relevant memories** from your conversation history and stored facts
2. **Decides how to respond** using its cognitive mode router (RESPOND, CLARIFY, ACKNOWLEDGE, ACT, or IGNORE)
3. **Generates a response** using your configured LLM provider
4. **Learns from the interaction**, storing new memories if relevant

---

## Quick Reference: Chalie CLI Commands

| Command | Description |
|---------|-------------|
| `chalie` | Start Chalie (default port 8081) |
| `chalie --port=9000` | Start on a custom port |
| `chalie stop` | Stop the running process |
| `chalie restart` | Restart Chalie |
| `chalie status` | Check if Chalie is running |
| `chalie logs` | Follow the log output |
| `chalie update` | Update to the latest version |

---

## Troubleshooting

### "Port 8081 is already in use"

Run on a different port:

```bash
chalie --port=9000
```

Then open **http://localhost:9000**.

### Ollama Connection Failed

Make sure Ollama is running:

```bash
ollama serve
```

Or restart it if already running.

### "Provider not configured"

Go back to the onboarding page at **http://localhost:8081/on-boarding/** and verify your provider settings are saved.

---

## What's Next?

Congratulations! You've completed your first 10 minutes with Chalie. Here are some next steps:

- 📚 Read [docs/03-WEB-INTERFACE.md](./03-WEB-INTERFACE.md) to explore the full UI
- 🔧 Learn about tools in [docs/09-TOOLS.md](./09-TOOLS.md)
- 🏗️ Build your own tool with [docs/18-TUTORIAL-BUILD-A-TOOL.md](./18-TUTORIAL-BUILD-A-TOOL.md)
- 🧠 Understand the cognitive architecture in [docs/07-COGNITIVE-ARCHITECTURE.md](./07-COGNITIVE-ARCHITECTURE.md)

---

## Uninstalling Chalie (If Needed)

To completely remove Chalie:

```bash
chalie stop
rm -rf ~/.chalie ~/.local/bin/chalie
```

Remove the `export PATH="$HOME/.local/bin:$PATH"` line from your shell config (`~/.bashrc` or `~/.zshrc`) if it was added by the installer.

---

**Welcome to Chalie!** 🎉 You now have a personal intelligence layer that remembers, adapts, and acts on your behalf. Enjoy exploring!