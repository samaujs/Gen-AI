import sys
from pathlib import Path

workflows_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/google-adk-workflows")
sys.path.insert(0, str(workflows_dir))

from google.adk.cli.utils.agent_loader import AgentLoader

loader = AgentLoader(agents_dir=str(workflows_dir))
try:
    loaded_obj = loader.load_agent("simple")
    print("Loaded successfully:", loaded_obj)
except Exception as e:
    import traceback
    traceback.print_exc()
