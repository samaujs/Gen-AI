import asyncio
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

# 1. Setup Streamlit Mock BEFORE imports
st_mock = MagicMock()
sidebar_mock = MagicMock()
st_mock.sidebar = sidebar_mock
st_mock.cache_data = lambda *args, **kwargs: lambda f: f

class DictMock:
    def __init__(self):
        super().__setattr__('_data', {})
    def __contains__(self, key):
        return key in self._data
    def __getitem__(self, key):
        return self._data[key]
    def __setitem__(self, key, value):
        self._data[key] = value
    def __delitem__(self, key):
        del self._data[key]
    def get(self, key, default=None):
        return self._data.get(key, default)
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"DictMock has no attribute {name}")
    def __setattr__(self, name, value):
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value

st_mock.session_state = DictMock()

def selectbox_mock(label, options, index=0, **kwargs):
    if options:
        if "Agent Workflow" in label:
            for opt in options:
                if "parallel" in opt:
                    return opt
        return options[index]
    return None

sidebar_mock.selectbox.side_effect = selectbox_mock
sidebar_mock.text_input.return_value = ""
sidebar_mock.button.return_value = False

sys.modules['streamlit'] = st_mock

# 2. Add paths
workflows_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/google-adk-workflows")
client_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/streamlit_client")
sys.path.insert(0, str(workflows_dir))
sys.path.insert(0, str(client_dir))

from dotenv import load_dotenv
load_dotenv(client_dir / ".env")

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
os.environ["MODEL_NAME"] = "gemini-3.1-flash-lite"

# Setup custom logs location for isolation
logs_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/logs")
logs_dir.mkdir(parents=True, exist_ok=True)
app_log_path = logs_dir / "app.log"

# Backup existing log if exists
log_backup = None
if app_log_path.exists():
    log_backup = app_log_path.read_text(encoding="utf-8")
    app_log_path.unlink()

# Write a mock session log containing a New Zealand trip itinerary from 01 - 15 Dec 2026
session_id = "restore-test-session-uuid-12345"
mock_log = f"""[2026-06-27 15:40:00] [Session ID: {session_id}]
USER: Provide a detailed travel itinerary plan to New Zealand from 01 - 15 Dec 2026 including direct flights from Singapore and 5 star hotel accommodations. This plan should include sightseeings places with introductions for each day.
TRIPSUMMARYAGENT: # Luxury Couple's Getaway: New Zealand Itinerary **Dates:** 01 December 2026 – 15 December 2026. This trip features direct flights from Singapore and lodging at 5-star hotel accommodations.
--------------------------------------------------------------------------------

"""
app_log_path.write_text(mock_log, encoding="utf-8")

from app import parse_app_log, load_past_session_to_adk, run_agent_stream, SmartMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
import app
print(f"DEBUG test_restore_chat: app path is {app.__file__}")
def run_query(agent_name, prompt):
    event_stream = run_agent_stream(
        agents_parent_dir=workflows_dir,
        agent_name=agent_name,
        prompt=prompt
    )
    final_text = ""
    for event in event_stream:
        text = ""
        if event.content and event.content.parts:
            text = "".join(part.text for part in event.content.parts if part.text)
        if text:
            final_text += text
    return final_text

async def main():
    agent_name = "parallel"
    
    # 1. Parse app.log
    print("\n--- 1. PARSING APP.LOG ---")
    sessions = parse_app_log()
    assert session_id in sessions, f"Could not find session ID {session_id} in parsed log sessions!"
    history = sessions[session_id]
    print(f"✓ Parsed history successfully! Total messages: {len(history)}")
    print(f"USER Message: '{history[0]['content']}'")
    print(f"AGENT Message: '{history[1]['content'][:80]}...'")
    
    # 2. Re-initialize and load past session to ADK
    print("\n--- 2. RESTORING SESSION AND LOADING INTO LONG-TERM MEMORY ---")
    st_mock.session_state.chat_history = []
    st_mock.session_state.trajectories = []
    st_mock.session_state.adk_session_service = InMemorySessionService()
    st_mock.session_state.adk_artifact_service = InMemoryArtifactService()
    st_mock.session_state.adk_credential_service = InMemoryCredentialService()
    st_mock.session_state.adk_memory_service = SmartMemoryService()
    
    # Restore
    await load_past_session_to_adk(session_id, history)
    print("✓ Session reloaded and ingested into Long-Term Memory successfully.")
    
    # 3. Clear short-term variables to force long-term memory retrieval
    st_mock.session_state.chat_history = []
    st_mock.session_state.trajectories = []
    st_mock.session_state.adk_session_id = str(uuid.uuid4()) # Generate new UUID so Query 2 has to use memory
    
    # 4. Ask about the period of the restored itinerary
    query = "What is the period of this itinerary plan?"
    print(f"\n--- 3. EXECUTING FOLLOW-UP QUERY (New Session) ---")
    print(f"Prompt: '{query}'")
    response = run_query(agent_name, query)
    print(f"Response: '{response.strip()}'")
    
    # Restore log backup
    if log_backup:
        app_log_path.write_text(log_backup, encoding="utf-8")
    else:
        app_log_path.unlink()
        
    if "01" in response and "15" in response and "Dec" in response:
        print("\nSUCCESS: Chat restoration and long-term memory load verified!")
        sys.exit(0)
    else:
        print("\nFAILURE: Agent was unable to recall the trip dates from the restored long-term memory.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
