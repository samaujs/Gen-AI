import sys
from pathlib import Path

# Setup sys.path exactly as Streamlit client does
samples_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK")
sys.path.insert(0, str(samples_dir))

selected_agent = {
    "agents_dir": Path("/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/google-adk-workflows")
}

agent_parent = str(selected_agent["agents_dir"])
if agent_parent not in sys.path:
    sys.path.insert(0, agent_parent)

from google.adk.cli.utils.agent_loader import AgentLoader

# Import clear_all_agent_caches from app
sys.path.insert(0, str(samples_dir / "streamlit_client"))
from app import clear_all_agent_caches

def test_load(agent_name):
    print(f"\n--- Loading {agent_name} ---")
    clear_all_agent_caches()
    loader = AgentLoader(agents_dir=agent_parent)
    loaded_obj = loader.load_agent(agent_name)
    print(f"Successfully loaded {agent_name}:", loaded_obj)

# Test sequence representing user selecting different agents
try:
    test_load("dispatcher")
    test_load("parallel")
    test_load("self_critic")
    test_load("simple")
    test_load("dispatcher")
    test_load("simple")
    print("\nSUCCESS: All agents loaded cleanly in sequence without metadata conflicts!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
