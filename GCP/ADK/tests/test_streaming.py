import asyncio
import os
import sys
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock

# 1. Setup Streamlit Mock BEFORE imports
st_mock = MagicMock()
sidebar_mock = MagicMock()
st_mock.sidebar = sidebar_mock
st_mock.cache_data = lambda *args, **kwargs: lambda f: f

def selectbox_mock(label, options, index=0, **kwargs):
    if options:
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

# Configure environment variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
os.environ["MODEL_NAME"] = "gemini-3.1-flash-lite"

from app import run_agent_stream

def test_agent(agent_name, prompt):
    print("=" * 80)
    print(f"Testing streaming for agent: {agent_name}")
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    try:
        event_stream = run_agent_stream(
            agents_parent_dir=workflows_dir,
            agent_name=agent_name,
            prompt=prompt
        )
        
        events_count = 0
        text_count = 0
        for event in event_stream:
            events_count += 1
            author = event.author or "Agent"
            
            text = ""
            if event.content and event.content.parts:
                text = "".join(part.text for part in event.content.parts if part.text)
            
            # Print a snippet of the event
            if text:
                text_count += 1
                snippet = text.strip().replace('\n', ' ')[:60]
                print(f"[{author}] (Text chunk {text_count}): {snippet}...")
            
            # Check for tool calls
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "function_call", None):
                        print(f"[{author}] (Tool Call): {part.function_call.name}")
                    elif getattr(part, "function_response", None):
                        print(f"[{author}] (Tool Response): {part.function_response.name}")
                        
        print(f"\nSUCCESS: Agent {agent_name} completed with {events_count} total events and {text_count} text chunks.")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\nERROR: Agent {agent_name} failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return False

if __name__ == "__main__":
    prompt = "Give me a quick 1-day trip info to Auckland containing a flight and a hotel."
    
    # Test simple agent
    simple_ok = test_agent("simple", prompt)
    
    # Test parallel agent
    parallel_ok = test_agent("parallel", prompt)
    
    if simple_ok and parallel_ok:
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
