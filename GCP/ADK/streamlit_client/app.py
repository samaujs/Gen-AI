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

# Helper function to clear python import caches so environment changes and parent ownerships apply cleanly
def clear_all_agent_caches():
    to_delete = ["subagent", "simple", "dispatcher", "parallel", "self_critic"]
    for module_name in list(sys.modules.keys()):
        for name in to_delete:
            if module_name == name or module_name.startswith(f"{name}."):
                sys.modules.pop(module_name, None)

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
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.genai import types
except ImportError as e:
    st.error(f"Failed to import Google ADK components. Please make sure google-adk is installed. Error: {e}")
    st.stop()

from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
import re

def is_app_instance(obj):
    return obj is not None and getattr(obj.__class__, "__name__", "") == "App"

class SmartMemoryService(InMemoryMemoryService):
    """An improved in-memory memory service that filters stop words,
    sorts memories newest-first, and prioritizes trip summaries.
    """
    
    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        # Retrieve raw results from the superclass
        response = await super().search_memory(app_name=app_name, user_id=user_id, query=query)
        
        # Define stop words to prevent false positive keyword matches
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'details'
        }
        
        # Filter memories using meaningful words if available
        query_words = set(word.lower() for word in re.findall(r'[A-Za-z]+', query))
        meaningful_words = query_words - stop_words
        
        if response.memories and meaningful_words:
            filtered_memories = []
            for mem in response.memories:
                text = " ".join([p.text for p in mem.content.parts if p.text]).lower()
                event_words = set(re.findall(r'[A-Za-z]+', text))
                if any(w in event_words for w in meaningful_words):
                    filtered_memories.append(mem)
            response.memories = filtered_memories
            
        # Sort by timestamp descending (newest first)
        if response.memories:
            response.memories.sort(key=lambda m: m.timestamp or "", reverse=True)
            
        # Prioritize final summaries and remove duplicates
        prioritized = []
        others = []
        seen_texts = set()
        
        for mem in response.memories:
            if not mem.content or not mem.content.parts:
                continue
            text = " ".join([p.text for p in mem.content.parts if p.text]).strip()
            if not text:
                continue
                
            norm_text = text[:300]
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            
            author_lower = (mem.author or "").lower()
            text_lower = text.lower()
            if "summary" in author_lower or "itinerary" in author_lower or "travel itinerary" in text_lower:
                prioritized.append(mem)
            else:
                others.append(mem)
                
        response.memories = (prioritized + others)[:5]
        return response

def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(v) for v in obj]
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return make_serializable(obj.model_dump())
    if hasattr(obj, "dict") and callable(obj.dict):
        return make_serializable(obj.dict())
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if hasattr(obj, "__dict__"):
        return {str(k): make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj)

def save_chat_and_traces_to_json(session_id, chat_history, trajectories):
    import json
    from pathlib import Path
    
    log_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "chat_history.json"
    
    # Read existing history if present
    data = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
            
    # Safely serialize trajectories to standard dicts
    safe_trajectories = []
    for traj in trajectories:
        safe_trace = []
        for event in traj.get("trace", []):
            safe_parts = []
            for part in event.get("parts", []):
                part_type = part.get("type", "unknown")
                if part_type == "text":
                    safe_parts.append({"type": "text", "value": part.get("value", "")})
                elif part_type == "function_call":
                    safe_parts.append({
                        "type": "function_call",
                        "name": part.get("name", ""),
                        "args": part.get("args", {})
                    })
                elif part_type == "function_response":
                    safe_parts.append({
                        "type": "function_response",
                        "name": part.get("name", ""),
                        "response": part.get("response", "")
                    })
            safe_trace.append({
                "author": event.get("author", "Agent"),
                "timestamp": event.get("timestamp", 0.0),
                "parts": safe_parts,
                "usage": event.get("usage")
            })
        safe_trajectories.append({
            "prompt": traj.get("prompt", ""),
            "agent": traj.get("agent", ""),
            "trace": safe_trace
        })
        
    data[session_id] = {
        "conversations": chat_history,
        "observability_traces": safe_trajectories
    }
    
    # Write back to JSON file using recursive serializer helper
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(make_serializable(data), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat history to JSON: {e}")

def append_to_app_log(session_id, user_prompt, agent_responses):
    import datetime
    from pathlib import Path
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [Session ID: {session_id}]\n"
    log_line += f"USER: {user_prompt}\n"
    for author, response in agent_responses.items():
        log_line += f"{author.upper()}: {response.strip()}\n"
    log_line += "-" * 80 + "\n\n"
    
    log_paths = [
        Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/logs/app.log")
    ]
    
    for path in log_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"Error appending to app.log at {path}: {e}")

def parse_app_log():
    import re
    from pathlib import Path
    
    log_path = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/logs/app.log")
    if not log_path.exists():
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()
        except Exception as e:
            print(f"Error creating app.log: {e}")
        return {}
        
    sessions = {}
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {}
        
    # Split by the divider to get blocks
    blocks = content.split("-" * 80)
    
    header_pattern = re.compile(r"\[.*?\]\s+\[Session ID:\s+([^\]]+)\]")
    
    AGENT_NAME_MAPPING = {
        "TRIPSUMMARYAGENT": "TripSummaryAgent",
        "FLIGHTAGENT": "FlightAgent",
        "HOTELAGENT": "HotelAgent",
        "SIGHTSEEINGAGENT": "SightseeingAgent",
        "WEATHERAGENT": "WeatherAgent",
        "CRITICAGENT": "CriticAgent",
        "COORDINATORAGENT": "CoordinatorAgent",
        "PLANNERAGENT": "PlannerAgent",
        "WORKFLOWAGENT": "WorkflowAgent"
    }
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = block.splitlines()
        if not lines:
            continue
            
        # The first line contains the session ID header
        header_line = lines[0].strip()
        match = header_pattern.match(header_line)
        if not match:
            continue
            
        session_id = match.group(1)
        if session_id not in sessions:
            sessions[session_id] = []
            
        current_msg = None
        for line in lines[1:]:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Check if line starts a new message
            parts = line.split(":", 1)
            if len(parts) == 2 and (parts[0].strip() == "USER" or (parts[0].strip().isupper() and "AGENT" in parts[0].strip())):
                if current_msg:
                    sessions[session_id].append(current_msg)
                
                role_name = parts[0].strip()
                message_content = parts[1].strip()
                
                if role_name == "USER":
                    current_msg = {
                        "role": "user",
                        "author": "user",
                        "content": message_content,
                        "avatar": "👤"
                    }
                else:
                    mapped_author = AGENT_NAME_MAPPING.get(role_name, role_name.title())
                    avatar = get_avatar_for_author(mapped_author)
                    current_msg = {
                        "role": "assistant",
                        "author": mapped_author,
                        "content": f"### {mapped_author}\n{message_content}",
                        "avatar": avatar
                    }
            else:
                if current_msg:
                    current_msg["content"] += "\n" + line
                    
        if current_msg:
            sessions[session_id].append(current_msg)
            
    return sessions

async def load_past_session_to_adk(session_id, chat_history):
    import streamlit as st
    from google.adk.events.event import Event
    from google.genai import types
    import uuid
    
    session_service = st.session_state.adk_session_service
    memory_service = st.session_state.adk_memory_service
    
    # Resolve exact application name matching agent metadata
    session_app_name = st.session_state.get("current_session_app_name")
    if not session_app_name:
        agent_name = st.session_state.get("current_agent_name", "parallel")
        try:
            from google.adk.types import App
            from google.adk.platform.agent_loader import AgentLoader
            from pathlib import Path
            workflows_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/google-adk-workflows")
            loader = AgentLoader(agents_dir=str(workflows_dir))
            agent_or_app = loader.load_agent(agent_name)
            session_app_name = agent_or_app.name if is_app_instance(agent_or_app) else agent_name
        except Exception:
            session_app_name = agent_name

    # Clear existing session in session_service if it exists, to avoid AlreadyExistsError
    try:
        await session_service.delete_session(
            app_name=session_app_name,
            user_id="streamlit_user",
            session_id=session_id
        )
    except Exception:
        pass
        
    # Re-create empty session
    session = await session_service.create_session(
        app_name=session_app_name,
        user_id="streamlit_user",
        session_id=session_id
    )
    
    # Construct and append events
    for msg in chat_history:
        author = msg.get("author", "user")
        role = "user" if author == "user" else "model"
        
        text_content = msg["content"]
        if role == "model" and text_content.startswith("### "):
            parts = text_content.split("\n", 1)
            if len(parts) == 2:
                text_content = parts[1]
                
        event = Event(
            author=author,
            content=types.Content(role=role, parts=[types.Part(text=text_content)]),
            invocation_id=str(uuid.uuid4())
        )
        await session_service.append_event(session=session, event=event)
        
    # Retrieve final compiled session with all historical events
    final_session = await session_service.get_session(
        app_name=session_app_name,
        user_id="streamlit_user",
        session_id=session_id
    )
    
    # Load historical session into Long-Term Memory
    print(f"DEBUG load_past_session_to_adk: session app_name={final_session.app_name}, user_id={final_session.user_id}, id={final_session.id}, events_count={len(final_session.events)}")
    await memory_service.add_session_to_memory(final_session)
    print(f"DEBUG load_past_session_to_adk: memory keys = {list(memory_service._session_events.keys())}")
    for k, v in memory_service._session_events.items():
        print(f"DEBUG load_past_session_to_adk: key {k} has sessions: {list(v.keys())}")

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
            # Clear all agent caches to avoid parent conflicts
            clear_all_agent_caches()
            # Set up the loader
            loader = AgentLoader(agents_dir=str(agents_parent_dir))
            agent_or_app = loader.load_agent(agent_name)
            
            session_app_name = (
                agent_or_app.name if is_app_instance(agent_or_app) else agent_name
            )
            
            app = (
                agent_or_app
                if is_app_instance(agent_or_app)
                else App(name=session_app_name, root_agent=agent_or_app)
            )
            
            import streamlit as st
            
            # Retrieve or instantiate session-state persisted services
            if "adk_session_service" not in st.session_state:
                st.session_state.adk_session_service = InMemorySessionService()
            if "adk_artifact_service" not in st.session_state:
                st.session_state.adk_artifact_service = InMemoryArtifactService()
            if "adk_credential_service" not in st.session_state:
                st.session_state.adk_credential_service = InMemoryCredentialService()
            if "adk_memory_service" not in st.session_state:
                st.session_state.adk_memory_service = SmartMemoryService()
                
            session_service = st.session_state.adk_session_service
            artifact_service = st.session_state.adk_artifact_service
            credential_service = st.session_state.adk_credential_service
            memory_service = st.session_state.adk_memory_service
            
            # Reset short-term session if agent workflow changes
            if st.session_state.get("current_agent_name") != agent_name:
                st.session_state.current_agent_name = agent_name
                if "adk_session_id" in st.session_state:
                    del st.session_state["adk_session_id"]
            
            # Reuse session_id for multi-turn conversations
            if "adk_session_id" not in st.session_state:
                session = await session_service.create_session(
                    app_name=session_app_name, user_id="streamlit_user"
                )
                st.session_state.adk_session_id = session.id
            else:
                try:
                    session = await session_service.get_session(
                        app_name=session_app_name,
                        user_id="streamlit_user",
                        session_id=st.session_state.adk_session_id
                    )
                    if session is None:
                        session = await session_service.create_session(
                            app_name=session_app_name,
                            user_id="streamlit_user",
                            session_id=st.session_state.adk_session_id
                        )
                except Exception:
                    session = await session_service.create_session(
                        app_name=session_app_name, user_id="streamlit_user"
                    )
                    st.session_state.adk_session_id = session.id
                    
            runner = Runner(
                app=app,
                artifact_service=artifact_service,
                session_service=session_service,
                credential_service=credential_service,
                memory_service=memory_service,
            )
            
            import datetime
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            context_prefix = f"[System Context: Current local time is {current_time_str}. Use this to resolve relative dates or refer to past memories.]\n"
            content = types.Content(role='user', parts=[types.Part(text=context_prefix + prompt)])
            
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=content,
            ):
                q.put(("event", event))
                
            # Retrieve the latest session with all completed events, then ingest into memory service
            session = await session_service.get_session(
                app_name=session_app_name,
                user_id="streamlit_user",
                session_id=session.id
            )
            await memory_service.add_session_to_memory(session)
            
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
    
    run_trace = []
    agent_responses = {}
    
    while True:
        try:
            msg_type, val = q.get(timeout=180)  # 3 minute timeout for long multi-agent chains
            if msg_type == "event":
                event = val
                author = event.author or "Agent"
                
                # Extract parts for observability
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
                    if author not in agent_responses:
                        agent_responses[author] = ""
                    agent_responses[author] += text
                    
                yield val
            elif msg_type == "done":
                # Save trajectory to session state
                if "trajectories" in st.session_state:
                    st.session_state.trajectories.append({
                        "prompt": prompt,
                        "agent": agent_name,
                        "trace": run_trace
                    })
                    
                # Post final consolidated results to the chat history
                for author, response_text in agent_responses.items():
                    avatar_icon = get_avatar_for_author(author)
                    formatted_response = f"### {author}\n{response_text}"
                    if "chat_history" in st.session_state:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": formatted_response,
                            "avatar": avatar_icon
                        })
                        
                # Save session chat history and traces to JSON, and conversations to app.log
                session_id = st.session_state.get("adk_session_id")
                if session_id:
                    save_chat_and_traces_to_json(session_id, st.session_state.chat_history, st.session_state.trajectories)
                    append_to_app_log(session_id, prompt, agent_responses)
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
try:
    default_agent_index = agent_names.index("google-adk-workflows/parallel")
except ValueError:
    default_agent_index = 0

selected_rel_path = st.sidebar.selectbox(
    "Select Agent Workflow",
    options=agent_names,
    index=default_agent_index
)
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
    clear_all_agent_caches()
    st.toast("Cleared all agent caches successfully!", icon="🔄")

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# Display Agent Details
st.sidebar.markdown("<h3 style='color: white; font-family: Outfit; font-size: 1.1em;'>Active Agent Metadata</h3>", unsafe_allow_html=True)

# Load the selected agent for metadata inspection
try:
    # Clear all agent caches to avoid parent conflicts
    clear_all_agent_caches()
    # Set parent path to sys.path so it can import subagent modules
    agent_parent = str(selected_agent["agents_dir"])
    if agent_parent not in sys.path:
        sys.path.insert(0, agent_parent)
        
    loader = AgentLoader(agents_dir=agent_parent)
    loaded_obj = loader.load_agent(selected_agent["name"])
    
    session_app_name = loaded_obj.name if is_app_instance(loaded_obj) else selected_agent["name"]
    st.session_state.current_session_app_name = session_app_name
    
    agent_instance = loaded_obj.root_agent if is_app_instance(loaded_obj) else loaded_obj
    
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
    import traceback
    traceback.print_exc()
    st.sidebar.error(f"Could not load metadata: {load_err}")

# Agent Observability & Trajectories
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# New Chat Button
if st.sidebar.button("🧹 New chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.trajectories = []
    st.session_state.adk_memory_service = SmartMemoryService()
    st.session_state.adk_session_service = InMemorySessionService()
    st.session_state.adk_artifact_service = InMemoryArtifactService()
    st.session_state.adk_credential_service = InMemoryCredentialService()
    import uuid
    st.session_state.adk_session_id = str(uuid.uuid4())
    st.rerun()

# Restore Past Session dropdown
try:
    past_sessions = parse_app_log()
    if past_sessions:
        st.sidebar.markdown("<h3 style='color: white; font-family: Outfit; font-size: 1.1em;'>🕰️ Restore Chat</h3>", unsafe_allow_html=True)
        
        session_ids = list(past_sessions.keys())
        
        def format_session_option(sid):
            user_msgs = [m for m in past_sessions[sid] if m["role"] == "user"]
            turns = len(user_msgs)
            first_prompt = "Empty"
            if user_msgs:
                first_prompt = user_msgs[0]["content"][:15] + "..." if len(user_msgs[0]["content"]) > 15 else user_msgs[0]["content"]
            return f"{first_prompt} ({turns} turns) - {sid[:8]}"
            
        selected_sid = st.sidebar.selectbox(
            "Select Past Conversation",
            options=session_ids,
            format_func=format_session_option,
            key="past_session_selector"
        )
        
        if st.sidebar.button("📂 Load Selected Chat", use_container_width=True):
            history = past_sessions[selected_sid]
            st.session_state.chat_history = history
            st.session_state.adk_session_id = selected_sid
            
            # Load it into the ADK Session Service and Long-Term memory
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(load_past_session_to_adk(selected_sid, history))
            finally:
                loop.close()
                
            st.toast("Chat session loaded successfully!", icon="📂")
            st.rerun()
except Exception as parse_err:
    st.sidebar.error(f"Error loading past sessions: {parse_err}")

st.sidebar.markdown("<h3 style='color: white; font-family: Outfit; font-size: 1.1em;'>🕵️ Agent Observability</h3>", unsafe_allow_html=True)

if "trajectories" not in st.session_state or not st.session_state.trajectories:
    st.sidebar.info("No active trajectories recorded yet. Run a prompt to view the trace.")
else:
    run_options = []
    for idx, traj in enumerate(st.session_state.trajectories):
        prompt_preview = traj["prompt"][:22] + "..." if len(traj["prompt"]) > 22 else traj["prompt"]
        run_options.append(f"Qn{idx + 1}: {prompt_preview} ({traj['agent']})")
    
    selected_run_idx = st.sidebar.selectbox(
        "Select Trajectory",
        options=range(len(run_options)),
        format_func=lambda x: run_options[x]
    )
    
    if st.sidebar.button("🔍 View Communication Trace", use_container_width=True):
        show_trajectory_details(st.session_state.trajectories[selected_run_idx])

# ----------------- CHAT ROOM -----------------

# Initialize Chat History, Trajectories, Session ID, and Memory Services
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "trajectories" not in st.session_state:
    st.session_state.trajectories = []
if "adk_session_service" not in st.session_state:
    st.session_state.adk_session_service = InMemorySessionService()
if "adk_artifact_service" not in st.session_state:
    st.session_state.adk_artifact_service = InMemoryArtifactService()
if "adk_credential_service" not in st.session_state:
    st.session_state.adk_credential_service = InMemoryCredentialService()
if "adk_memory_service" not in st.session_state:
    st.session_state.adk_memory_service = SmartMemoryService()
if "adk_session_id" not in st.session_state:
    import uuid
    st.session_state.adk_session_id = str(uuid.uuid4())

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
    clear_all_agent_caches()
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt,
        "avatar": "👤"
    })
    
    with st.chat_message("user", avatar="👤"):
        st.write(user_prompt)
        
    # Execution
    # Create containers outside the spinner so they render above it
    status_container = st.container()
    chat_container = st.container()
    spinner_container = st.empty()
    
    with spinner_container.container():
        with st.spinner("Agent is still working on the query..."):
            with status_container:
                # Real-time multi-agent logging container
                status_box = st.status("🚀 Agent executing...", expanded=True)
            
            # Dictionary to accumulate response chunks by agent author
            agent_responses = {}
            # Keep track of active chat placeholders for real-time streaming
            chat_placeholders = {}
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
                    
                    # Log tool calls/responses in status box
                    for part_info in trace_parts:
                        if part_info["type"] == "function_call":
                            status_box.write(f"⚙️ **{author}** calling tool `{part_info['name']}`")
                        elif part_info["type"] == "function_response":
                            status_box.write(f"📥 **{author}** tool returned response")
                    
                    # Fetch text content
                    text = ""
                    if event.content and event.content.parts:
                        text = "".join(part.text for part in event.content.parts if part.text)
                    
                    if text:
                        # Accumulate response
                        if author not in agent_responses:
                            agent_responses[author] = ""
                            # Create chat message placeholder dynamically
                            avatar_icon = get_avatar_for_author(author)
                            with chat_container:
                                with st.chat_message("assistant", avatar=avatar_icon):
                                    chat_placeholders[author] = st.empty()
                        
                        agent_responses[author] += text
                        
                        # Update streaming response
                        chat_placeholders[author].markdown(f"### {author}\n{agent_responses[author]}")
                        
            except Exception as run_err:
                spinner_container.empty()
                status_box.update(label="❌ Execution Failed", state="error")
                st.error(f"Execution failed: {run_err}")
            else:
                spinner_container.empty()
                status_box.update(label="✅ Execution Completed", state="complete")
                # Force refresh so streamlit re-renders with new history
                st.rerun()
