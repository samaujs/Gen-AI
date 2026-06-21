import asyncio
import os
import sys
import queue
import threading
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env in the streamlit_client directory
load_dotenv(Path(__file__).parent / ".env")

# Add the parent directory of this app to sys.path so we can locate the virtualenv site-packages if needed,
# though we run streamlit with the virtualenv's python directly.
samples_dir = Path(__file__).resolve().parent.parent

# Set up Streamlit Page config
st.set_page_config(
    page_title="ADK Agent Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style injection for premium look
st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .header-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.5em;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #b0bec5;
        font-weight: 400;
        font-size: 1.1em;
        margin: 8px 0 0 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0c1015 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom status/card containers */
    .info-card {
        background-color: #171c24;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
    }
    
    .info-card-title {
        font-weight: 600;
        font-size: 1.05em;
        color: #eceff1;
        margin-bottom: 8px;
    }
    
    .info-card-content {
        color: #90a4ae;
        font-size: 0.9em;
        line-height: 1.4;
    }
    
    /* Dynamic pill badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .badge-python { background-color: #1e3a8a; color: #93c5fd; }
    .badge-yaml { background-color: #115e59; color: #99f6e4; }
    .badge-subagent { background-color: #312e81; color: #c7d2fe; }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to clear python import caches so environment changes apply immediately
def clear_agent_cache(agent_name: str):
    to_delete = []
    for module_name in list(sys.modules.keys()):
        if module_name == agent_name or module_name.startswith(f"{agent_name}."):
            to_delete.append(module_name)
    # Also pop common subagent module if it's imported
    if "subagent" in sys.modules:
        to_delete.append("subagent")
    
    for key in to_delete:
        sys.modules.pop(key, None)

# Helper to scan for ADK Agents in GenAI/samples
@st.cache_data(show_spinner="Scanning local directory for ADK agents...")
def discover_agents(root_dir: Path):
    discovered = []
    # Avoid scanning deep inside standard environment/virtualenv/git/client folders
    ignore_dirs = {".git", "venv", ".venv", "streamlit_client", "__pycache__"}
    
    # We walk the directory
    for p in root_dir.rglob("*"):
        if any(part in ignore_dirs or part.startswith(".") for part in p.parts):
            continue
        
        # Check if this folder has agent.py or root_agent.yaml
        if p.is_dir():
            if (p / "agent.py").exists() or (p / "root_agent.yaml").exists():
                # Get path relative to the root directory
                rel_path = p.relative_to(root_dir)
                discovered.append({
                    "name": p.name,
                    "agents_dir": p.parent,
                    "rel_path": str(rel_path),
                    "full_path": p
                })
    
    # Sort alphabetically by relative path
    discovered.sort(key=lambda x: x["rel_path"])
    return discovered

# Load the actual ADK imports
try:
    from google.adk.cli.utils.agent_loader import AgentLoader
    from google.adk.runners import Runner
    from google.adk.apps.app import App
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
    from google.genai import types
except ImportError as e:
    st.error(f"Failed to import Google ADK components. Please make sure google-adk is installed. Error: {e}")
    st.stop()

# Helper function to get custom icons for various subagents
def get_avatar_for_author(author: str) -> str:
    author_lower = author.lower()
    if "flight" in author_lower:
        return "✈️"
    elif "hotel" in author_lower:
        return "🏨"
    elif "sightseeing" in author_lower or "tourism" in author_lower:
        return "🗺️"
    elif "summary" in author_lower or "itinerary" in author_lower:
        return "📋"
    elif "critic" in author_lower or "reviewer" in author_lower or "validator" in author_lower:
        return "🔍"
    elif "coordinator" in author_lower or "planner" in author_lower or "workflow" in author_lower:
        return "🤖"
    return "🤖"

# Async generator execution wrapper running in a separate thread
def run_agent_stream(agents_parent_dir, agent_name, prompt):
    q = queue.Queue()
    
    async def runner_task():
        try:
            # Set up the loader
            loader = AgentLoader(agents_dir=str(agents_parent_dir))
            agent_or_app = loader.load_agent(agent_name)
            
            session_app_name = (
                agent_or_app.name if isinstance(agent_or_app, App) else agent_name
            )
            
            app = (
                agent_or_app
                if isinstance(agent_or_app, App)
                else App(name=session_app_name, root_agent=agent_or_app)
            )
            
            # Use in-memory services for fast and simple client execution
            session_service = InMemorySessionService()
            artifact_service = InMemoryArtifactService()
            credential_service = InMemoryCredentialService()
            
            runner = Runner(
                app=app,
                artifact_service=artifact_service,
                session_service=session_service,
                credential_service=credential_service,
            )
            
            session = await session_service.create_session(
                app_name=session_app_name, user_id="streamlit_user"
            )
            
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
            ):
                q.put(("event", event))
                
            await runner.close()
            q.put(("done", None))
        except Exception as ex:
            q.put(("error", ex))

    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner_task())
        loop.close()

    thread = threading.Thread(target=run_loop)
    thread.start()
    
    while True:
        try:
            msg_type, val = q.get(timeout=180)  # 3 minute timeout for long multi-agent chains
            if msg_type == "event":
                yield val
            elif msg_type == "done":
                break
            elif msg_type == "error":
                raise val
        except queue.Empty:
            raise TimeoutError("The agent run timed out (no events received for 3 minutes).")

@st.dialog("Agent Communication & Trajectory Trace", width="large")
def show_trajectory_details(trajectory):
    st.markdown(f"### Trajectory for: *\"{trajectory['prompt']}\"*")
    st.markdown(f"**Orchestrator**: `{trajectory['agent']}`")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    for event in trajectory["trace"]:
        author = event["author"]
        avatar = get_avatar_for_author(author)
        
        with st.container(border=True):
            col1, col2 = st.columns([1, 15])
            with col1:
                st.markdown(f"<div style='font-size: 1.5em; margin-top: 3px;'>{avatar}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{author}**")
            
            for part in event["parts"]:
                if part["type"] == "text":
                    st.markdown(part["value"])
                elif part["type"] == "function_call":
                    st.markdown(f"⚙️ **Tool Call**: `{part['name']}`")
                    st.json(part["args"])
                elif part["type"] == "function_response":
                    st.markdown(f"📥 **Tool Response**: `{part['name']}`")
                    st.json(part["response"])
            
            if event.get("usage"):
                st.markdown(
                    f"<div style='font-size: 0.8em; color: #90a4ae; margin-top: 10px; text-align: right;'>"
                    f"Token Usage — Prompt: {event['usage']['prompt_tokens']} | Candidates: {event['usage']['candidates_tokens']} | Total: {event['usage']['total_tokens']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ----------------- MAIN UI -----------------

# Discover Agents
agents_list = discover_agents(samples_dir)

# Title Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Premium Tour Planning Agent</h1>
        <p class="header-subtitle">ADK Strike Force with Google Agent Development Kit (ADK) that runs locally.</p>
    </div>
""", unsafe_allow_html=True)

# Check if any agents were found
if not agents_list:
    st.warning(f"No ADK agents discovered in the `{samples_dir}` directory. Make sure you have folders with `agent.py` or `root_agent.yaml` files.")
    st.stop()

# Sidebar Configuration
st.sidebar.markdown("<h2 style='color: white; font-family: Outfit; margin-bottom: 20px;'>Configuration</h2>", unsafe_allow_html=True)

# Agent Selector
agent_names = [a["rel_path"] for a in agents_list]
selected_rel_path = st.sidebar.selectbox("Select Agent Workflow", options=agent_names)
selected_agent = next(a for a in agents_list if a["rel_path"] == selected_rel_path)

# Set up environment variables overrides
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: white; font-family: Outfit; font-size: 1.1em;'>Model & Credentials</h3>", unsafe_allow_html=True)

# Google API Key Override
api_key_input = st.sidebar.text_input(
    "Google API Key",
    type="password",
    value=os.environ.get("GOOGLE_API_KEY", ""),
    help="Enter your Gemini API key to override environment settings."
)

if api_key_input:
    os.environ["GOOGLE_API_KEY"] = api_key_input

# Model Name Override
model_options = {
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.1 Pro - Preview": "gemini-3.1-pro-preview",
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite"
}
current_model = os.environ.get("MODEL_NAME", "gemini-3.1-flash-lite")

# Determine default index
try:
    default_index = list(model_options.values()).index(current_model)
except ValueError:
    default_index = 0

selected_display_name = st.sidebar.selectbox(
    "Model Name",
    options=list(model_options.keys()),
    index=default_index
)
selected_model = model_options[selected_display_name]

# Apply model override
os.environ["MODEL_NAME"] = selected_model

# Clear Cache / Reload button
if st.sidebar.button("🔄 Reload Agent Source"):
    clear_agent_cache(selected_agent["name"])
    st.toast(f"Cleared cache for agent '{selected_agent['name']}'!", icon="🔄")

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# Display Agent Details
st.sidebar.markdown("<h3 style='color: white; font-family: Outfit; font-size: 1.1em;'>Active Agent Metadata</h3>", unsafe_allow_html=True)

# Load the selected agent for metadata inspection
try:
    # Set parent path to sys.path so it can import subagent modules
    agent_parent = str(selected_agent["agents_dir"])
    if agent_parent not in sys.path:
        sys.path.insert(0, agent_parent)
        
    loader = AgentLoader(agents_dir=agent_parent)
    loaded_obj = loader.load_agent(selected_agent["name"])
    
    agent_instance = loaded_obj.root_agent if isinstance(loaded_obj, App) else loaded_obj
    
    # Render Agent Details Card
    lang = "python" if (selected_agent["full_path"] / "agent.py").exists() else "yaml"
    lang_badge = f"<span class='badge badge-{lang}'>{lang.upper()}</span>"
    
    st.sidebar.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">{agent_instance.name}</div>
            <div class="info-card-content">
                <p><strong>Path:</strong> <code>{selected_rel_path}</code></p>
                <p><strong>Description:</strong> {agent_instance.description or 'No description provided.'}</p>
                <p><strong>Language:</strong> {lang_badge}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Subagents listing if present
    if hasattr(agent_instance, "sub_agents") and agent_instance.sub_agents:
        st.sidebar.markdown("<h4 style='color: white; font-size: 0.95em;'>Coordinated Sub-agents</h4>", unsafe_allow_html=True)
        for sub in agent_instance.sub_agents:
            sub_desc = sub.description or "No description."
            st.sidebar.markdown(f"""
                <div class="info-card" style="margin-bottom: 8px; padding: 10px;">
                    <div class="info-card-title" style="font-size:0.9em; margin-bottom: 2px;">🤖 {sub.name}</div>
                    <div class="info-card-content" style="font-size:0.8em;">{sub_desc}</div>
                </div>
            """, unsafe_allow_html=True)
except Exception as load_err:
    st.sidebar.error(f"Could not load metadata: {load_err}")

# Agent Observability & Trajectories
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: white; font-family: Outfit; font-size: 1.1em;'>🕵️ Agent Observability</h3>", unsafe_allow_html=True)

if "trajectories" not in st.session_state or not st.session_state.trajectories:
    st.sidebar.info("No active trajectories recorded yet. Run a prompt to view the trace.")
else:
    run_options = []
    for idx, traj in enumerate(st.session_state.trajectories):
        prompt_preview = traj["prompt"][:22] + "..." if len(traj["prompt"]) > 22 else traj["prompt"]
        run_options.append(f"Query {idx + 1}: {prompt_preview} ({traj['agent']})")
    
    selected_run_idx = st.sidebar.selectbox(
        "Select Trajectory",
        options=range(len(run_options)),
        format_func=lambda x: run_options[x]
    )
    
    if st.sidebar.button("🔍 View Communication Trace", use_container_width=True):
        show_trajectory_details(st.session_state.trajectories[selected_run_idx])

# Reset Chat Button
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.trajectories = []
    st.rerun()

# ----------------- CHAT ROOM -----------------

# Initialize Chat History and Trajectories
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "trajectories" not in st.session_state:
    st.session_state.trajectories = []

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.write(msg["content"])

# Check API Key before proceeding
if not os.environ.get("GOOGLE_API_KEY"):
    st.info("⚠️ Please enter your **Google API Key** in the sidebar to run the ADK agents.", icon="🔑")
    st.stop()

# User Input Box
if user_prompt := st.chat_input("Ask the agent something..."):
    # Clear cache before running to ensure updated env vars are evaluated
    clear_agent_cache(selected_agent["name"])
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt,
        "avatar": "👤"
    })
    
    with st.chat_message("user", avatar="👤"):
        st.write(user_prompt)
        
    # Execution
    with st.spinner("Invoking agent workflow..."):
        # Real-time multi-agent logging container
        status_box = st.status("🚀 Agent executing...", expanded=True)
        
        # Dictionary to accumulate response chunks by agent author
        agent_responses = {}
        # Accumulate trace events for observability
        run_trace = []
        
        try:
            # Run the agent stream
            event_stream = run_agent_stream(
                agents_parent_dir=selected_agent["agents_dir"],
                agent_name=selected_agent["name"],
                prompt=user_prompt
            )
            
            for event in event_stream:
                author = event.author or "Agent"
                
                # Extract parts for trajectory observability
                trace_parts = []
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        part_info = {}
                        if getattr(part, "text", None):
                            part_info["type"] = "text"
                            part_info["value"] = part.text
                        elif getattr(part, "function_call", None):
                            part_info["type"] = "function_call"
                            part_info["name"] = part.function_call.name
                            part_info["args"] = part.function_call.args
                        elif getattr(part, "function_response", None):
                            part_info["type"] = "function_response"
                            part_info["name"] = part.function_response.name
                            part_info["response"] = part.function_response.response
                        
                        if part_info:
                            trace_parts.append(part_info)
                
                usage = None
                if getattr(event, "usage_metadata", None):
                    usage = {
                        "prompt_tokens": event.usage_metadata.prompt_token_count,
                        "candidates_tokens": event.usage_metadata.candidates_token_count,
                        "total_tokens": event.usage_metadata.total_token_count
                    }
                
                run_trace.append({
                    "author": author,
                    "timestamp": getattr(event, "timestamp", 0.0),
                    "parts": trace_parts,
                    "usage": usage
                })
                
                # Fetch text content
                text = ""
                if event.content and event.content.parts:
                    text = "".join(part.text for part in event.content.parts if part.text)
                
                if text:
                    # Log event in status box
                    status_box.write(f"**[{author}]**: {text}")
                    
                    # Accumulate for final chat message
                    if author not in agent_responses:
                        agent_responses[author] = ""
                    agent_responses[author] += text
                    
        except Exception as run_err:
            status_box.update(label="❌ Execution Failed", state="error")
            st.error(f"Execution failed: {run_err}")
        else:
            status_box.update(label="✅ Execution Completed", state="complete")
            
            # Save trajectory
            st.session_state.trajectories.append({
                "prompt": user_prompt,
                "agent": selected_agent["name"],
                "trace": run_trace
            })
            
            # Post final consolidated results to the chat
            for author, response_text in agent_responses.items():
                avatar_icon = get_avatar_for_author(author)
                
                # Format response nicely
                formatted_response = f"### {author}\n{response_text}"
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": formatted_response,
                    "avatar": avatar_icon
                })
                
                with st.chat_message("assistant", avatar=avatar_icon):
                    st.write(formatted_response)
                    
            # Force refresh so streamlit re-renders with new history
            st.rerun()
