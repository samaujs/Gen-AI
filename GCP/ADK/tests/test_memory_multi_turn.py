import asyncio
import os
import sys
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
            
    def __delattr__(self, name):
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"DictMock has no attribute {name}")

st_mock.session_state = DictMock()

def selectbox_mock(label, options, index=0, **kwargs):
    if options:
        # Default index to parallel agent
        if "Agent Workflow" in label:
            for i, opt in enumerate(options):
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

from app import run_agent_stream

def run_query(agent_name, prompt):
    print("=" * 80)
    print(f"PROMPT: {prompt}")
    print("=" * 80)
    
    event_stream = run_agent_stream(
        agents_parent_dir=workflows_dir,
        agent_name=agent_name,
        prompt=prompt
    )
    
    final_text = ""
    for event in event_stream:
        author = event.author or "Agent"
        
        text = ""
        if event.content and event.content.parts:
            text = "".join(part.text for part in event.content.parts if part.text)
        
        if text:
            final_text += text
            # Print a snippet of compiling agent outputs
            if "Summary" in author or "Planner" in author:
                snippet = text.strip().replace('\n', ' ')[:80]
                print(f"[{author}]: {snippet}...")
            
        # Check for tool calls to print load_memory tool invocations
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    print(f"[{author}] (Tool Call): {part.function_call.name} with args: {part.function_call.args}")
                elif getattr(part, "function_response", None):
                    print(f"[{author}] (Tool Response): {part.function_response.name}")
                    
    print("=" * 80)
    return final_text

async def main():
    agent_name = "parallel"
    
    # Query 1: Create NZ trip
    query1 = (
        "Provide a detailed travel itinerary plan to New Zealand from 01 - 15 Dec 2026 including "
        "direct flights from Singapore and 5 star hotel accommodations. This plan should include "
        "sightseeings places with introductions for each day. The itinerary plan should be a highly "
        "memorable and relaxing trip for a couple."
    )
    print("\n--- RUNNING USER QUERY 1 (Saves itinerary into long-term memory) ---")
    response1 = run_query(agent_name, query1)
    
    # Verify that the memory was saved
    memory_service = st_mock.session_state.get("adk_memory_service")
    if memory_service:
        print(f"\nLong-Term Memory Sessions: {list(memory_service._session_events.get('ParallelWorkflow/streamlit_user', {}).keys())}")
    else:
        print("\nERROR: Long-term memory service not initialized!")
        sys.exit(1)
        
    # Reset short-term session to simulate a NEW session / conversation turn
    # This deletes the short-term session_id so Query 2 must rely on long-term memory retrieval
    if "adk_session_id" in st_mock.session_state:
        del st_mock.session_state["adk_session_id"]
        print("\nShort-term session cleared (simulating a new conversation thread).")
        
    # Query 2: Ask about the period of Query 1
    query2 = "What is the period of this itinerary plan?"
    print("\n--- RUNNING USER QUERY 2 (Should retrieve period from long-term memory) ---")
    response2 = run_query(agent_name, query2)
    
    print("\nFINAL RESPONSES SUMMARY:")
    print("-" * 50)
    print(f"Query 1 Target: New Zealand (01 - 15 Dec 2026)")
    print(f"Query 2 Response:\n{response2.strip()}")
    print("-" * 50)
    
    if "01" in response2 and "15" in response2 and "Dec" in response2:
        print("\nSUCCESS: Long-term memory verified! Response 2 correctly retrieved the 01-15 Dec 2026 period.")
        sys.exit(0)
    else:
        print("\nFAILURE: Long-term memory failed to retrieve the correct period.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
