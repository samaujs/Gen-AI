import sys
from pathlib import Path

# Setup sys.path exactly as Streamlit client does
samples_dir = Path("/Users/samaujs/Year_2026/GenAI/samples")
sys.path.insert(0, str(samples_dir))

selected_agent = {
    "name": "simple",
    "agents_dir": Path("/Users/samaujs/Year_2026/GenAI/samples/google-adk-workflows"),
    "rel_path": "google-adk-workflows/simple",
    "full_path": Path("/Users/samaujs/Year_2026/GenAI/samples/google-adk-workflows/simple")
}

agent_parent = str(selected_agent["agents_dir"])
if agent_parent not in sys.path:
    sys.path.insert(0, agent_parent)

from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.apps.app import App

try:
    loader = AgentLoader(agents_dir=agent_parent)
    loaded_obj = loader.load_agent(selected_agent["name"])
    print("LOADED OBJ:", loaded_obj)
except Exception as e:
    import traceback
    traceback.print_exc()
