# ADK Agent Studio & Workflows Hub

This project is a local workspace for developing, running, and analyzing multi-agent workflows built with Google's **Agent Development Kit (ADK)** and the **Gemini API**. It contains two primary components: a multi-agent backend travel planning system and a premium interactive Streamlit chat interface.

---

## 📂 Project Structure

```text
samples/
├── .gitignore                    # Local environment files, pycache, and logs ignore patterns
├── README.md                     # Project documentation
├── google-adk-workflows/         # Multi-Agent backend orchestration modules
│   ├── env.example               # Template environment settings
│   ├── .env                      # Local key configurations (ignored)
│   ├── subagent.py               # Core specialists (Flight, Hotel, Sightseeing, Summary)
│   ├── simple/                   # Sequential coordination coordinator
│   ├── dispatcher/               # Tool-based router coordinator
│   ├── parallel/                 # Concurrent executor coordinator
│   └── self_critic/              # QA Validator coordination loop
└── streamlit_client/             # Interactive Streamlit frontend client
    ├── app.py                    # Main Streamlit client source file
    ├── requirements.txt          # Client dependencies
    └── .env                      # Local client key configurations (ignored)
```

---

## 🏗️ Architecture Overview

The workspace separates **Agent Logic** from the **Client User Interface**:

```
┌────────────────────────────────────────┐
│         Streamlit Client (UI)          │
│  - App Layout & Custom CSS             │
│  - Agent Discovery & Cache Purging     │
│  - Threaded Queue Event Listener       │
└───────────────────┬────────────────────┘
                    │  Async run_agent_stream()
┌───────────────────▼────────────────────┐
│      Google ADK Engine (Backend)       │
│  - Runner & InMemory Services          │
│  - Orchestrator (Sequential/Parallel)  │
│  - Sub-Agents (Flight, Hotel, etc.)    │
└────────────────────────────────────────┘
```

1. **Agent Backend (`google-adk-workflows/`)**: Implements specialized agents (Flight, Hotel, Sightseeing, and Compilation agents) and defines workflows (Simple, Dispatcher, Parallel, and Self-Critic) that coordinate how they run sequentially or concurrently.
2. **Streamlit Client (`streamlit_client/`)**: Scans the workspace directory to discover agents on disk, dynamically configures runtime parameters (Gemini API Key and Model Name), purges Python's module cache to hot-reload changes instantly, and streams agent outputs live using a background thread and a thread-safe event queue.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.13+**
- A **Google Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Configure Environment Variables
You must set your credentials in a local environment file. 
Copy the `.env` template to both the client and workflows directories:

```bash
# Set up client environment variables
cp google-adk-workflows/env.example streamlit_client/.env

# Set up workflows environment variables
cp google-adk-workflows/env.example google-adk-workflows/.env
```

Open both `.env` files and add your actual API Key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
MODEL_NAME=gemini-3.5-flash
```

*(Alternatively, you can input your key directly into the Streamlit sidebar config panel during runtime).*

### 3. Run the Streamlit Interface
Start the application server using your configured Python virtual environment:

```bash
# Run from the project root directory
python -m streamlit run streamlit_client/app.py --browser.gatherUsageStats false
```

Once running, navigate to the local URL (usually **[http://localhost:8502](http://localhost:8502)**) in your web browser.

---

## ⚙️ Features

- **Workflow Selector**: Switch between different multi-agent coordination modes (Simple, Dispatcher, Parallel, Self-Critic) on the fly.
- **Model Selector**: Switch target runtimes between **Gemini 3.5 Flash** (`gemini-3.5-flash`) and **Gemini 3.1 Flash Lite** (`gemini-3.1-flash-lite`).
- **Cache Reloading**: Edit agent files locally and click **Reload Agent Source** to clear memory caches and load your new agent parameters immediately.
- **Sub-Agent Live Logging**: View real-time status updates of intermediate agent runs (e.g. `FlightAgent` executing, `HotelAgent` booking) inside collapsed status drawers before the final compiled answer arrives.
