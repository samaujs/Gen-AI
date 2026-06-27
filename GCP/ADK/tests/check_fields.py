import sys
from pathlib import Path
sys.path.insert(0, "/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/google-adk-workflows")

from google.adk.agents import LlmAgent
for name, field in LlmAgent.model_fields.items():
    print(f"{name}: {field.annotation} (default: {field.default})")
