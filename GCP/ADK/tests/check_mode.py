import sys
from pathlib import Path
sys.path.insert(0, "/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/google-adk-workflows")

from google.adk.agents import LlmAgent
import inspect

print("LlmAgent constructor signature:")
print(inspect.signature(LlmAgent.__init__))
