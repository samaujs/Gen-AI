import asyncio
import os
import sys
import json
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
workflows_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/google-adk-workflows")
client_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/streamlit_client")
sys.path.insert(0, str(workflows_dir))
sys.path.insert(0, str(client_dir))

from dotenv import load_dotenv
load_dotenv(client_dir / ".env")

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
os.environ["MODEL_NAME"] = "gemini-3.1-flash-lite"

# Clean up log files from previous runs to ensure clean test assertions
json_path = Path("/Users/samaujs/Year_2026/GenAI/samples/logs/chat_history.json")
logs_app_path = Path("/Users/samaujs/Year_2026/GenAI/samples/logs/app.log")

for p in [json_path, logs_app_path]:
    if p.exists():
        p.unlink()

from app import run_agent_stream, SmartMemoryService

def run_query(agent_name, prompt):
    event_stream = run_agent_stream(
        agents_parent_dir=workflows_dir,
        agent_name=agent_name,
        prompt=prompt
    )
    for event in event_stream:
        pass  # drain the stream to finish execution

async def main():
    agent_name = "parallel"
    
    # --- SESSION 1 ---
    # Set up initial session ID and services
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
    
    st_mock.session_state.chat_history = []
    st_mock.session_state.trajectories = []
    st_mock.session_state.adk_session_service = InMemorySessionService()
    st_mock.session_state.adk_artifact_service = InMemoryArtifactService()
    st_mock.session_state.adk_credential_service = InMemoryCredentialService()
    st_mock.session_state.adk_memory_service = SmartMemoryService()
    
    session_id_1 = str(uuid.uuid4())
    st_mock.session_state.adk_session_id = session_id_1
    
    query1 = "Give me a quick 1-day trip info to Auckland containing a flight and a hotel."
    print(f"\n[Session 1] ID: {session_id_1}")
    print(f"[Session 1] Running prompt: '{query1}'")
    
    # Simulate user sending prompt
    st_mock.session_state.chat_history.append({
        "role": "user",
        "content": query1,
        "avatar": "👤"
    })
    run_query(agent_name, query1)
    
    # Assert JSON file exists and contains session_id_1 key
    assert json_path.exists(), "chat_history.json was not created!"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert session_id_1 in json_data, f"session_id_1 ({session_id_1}) not found in chat_history.json!"
    print("✓ Session 1 data correctly stored in chat_history.json.")
    
    # Assert app.log contains conversation turn
    assert logs_app_path.exists(), "logs/app.log was not created!"
    with open(logs_app_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert session_id_1 in log_content, "session_id_1 not found in app.log!"
    print("✓ Session 1 turn recorded in app.log.")
    
    # --- SIMULATE NEW CHAT RESET ---
    print("\n--- SIMULATING 'NEW CHAT' BUTTON CLICK ---")
    st_mock.session_state.chat_history = []
    st_mock.session_state.trajectories = []
    st_mock.session_state.adk_memory_service = SmartMemoryService()
    st_mock.session_state.adk_session_service = InMemorySessionService()
    st_mock.session_state.adk_artifact_service = InMemoryArtifactService()
    st_mock.session_state.adk_credential_service = InMemoryCredentialService()
    
    session_id_2 = str(uuid.uuid4())
    st_mock.session_state.adk_session_id = session_id_2
    
    # Assert reset state is empty
    assert len(st_mock.session_state.chat_history) == 0, "Chat history not empty!"
    assert len(st_mock.session_state.trajectories) == 0, "Trajectories not empty!"
    print("✓ Verified chat history and trajectories cleared.")
    
    # --- SESSION 2 ---
    query2 = "Give me a quick 1-day trip info to Queenstown containing a flight and a hotel."
    print(f"\n[Session 2] ID: {session_id_2}")
    print(f"[Session 2] Running prompt: '{query2}'")
    
    st_mock.session_state.chat_history.append({
        "role": "user",
        "content": query2,
        "avatar": "👤"
    })
    run_query(agent_name, query2)
    
    # Assert JSON file exists and contains both session keys
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert session_id_1 in json_data, "session_id_1 was lost after new chat!"
    assert session_id_2 in json_data, f"session_id_2 ({session_id_2}) not found in chat_history.json!"
    print("✓ Both Session 1 and Session 2 are safely indexed inside chat_history.json.")
    
    # Assert app.log contains second session turn
    with open(logs_app_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert session_id_2 in log_content, "session_id_2 not found in app.log!"
    print("✓ Session 2 turn recorded in app.log.")
    
    print("\nALL REQUIREMENT CHECKS COMPLETED SUCCESSFULLY!")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
