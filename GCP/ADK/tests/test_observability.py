import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# 1. Setup Streamlit Mock BEFORE imports
st_mock = MagicMock()
sidebar_mock = MagicMock()
st_mock.sidebar = sidebar_mock

# Mock st.cache_data decorator
st_mock.cache_data = lambda *args, **kwargs: lambda f: f

# Mock selectbox to return appropriate values for different labels
def selectbox_mock(label, options, **kwargs):
    if 'Agent Workflow' in label:
        return 'google-adk-workflows/self_critic'
    elif 'Model Name' in label:
        return 'Gemini 3.5 Flash'
    return options[0] if options else None

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
load_dotenv()
os.environ["MODEL_NAME"] = "gemini-3.1-flash-lite"

from app import run_agent_stream, get_avatar_for_author

def run_nz_query():
    # Prompt requested by user
    prompt = (
        "Provide a detailed travel itinerary plan to New Zealand from 01 - 15 Dec 2026 including "
        "direct flights from Singapore and 5 star hotel accommodations. This plan should include "
        "daily sightseeings that are relaxing. It should be a highly memorable trip.\n\n"
        "For example:\n"
        "Day 1 - From Singapore to Auckland\n"
        "Day 2 - Travel to Waiheke Island\n"
        "...\n"
        "Day 15 - From Auckland to Singapore"
    )
    
    print(f"Executing query:\n{prompt}\n")
    print("=" * 80)
    
    event_stream = run_agent_stream(
        agents_parent_dir=workflows_dir,
        agent_name="self_critic",
        prompt=prompt
    )
    
    run_trace = []
    agent_responses = {}
    
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
        
        # Log text content
        text = ""
        if event.content and event.content.parts:
            text = "".join(part.text for part in event.content.parts if part.text)
            
        if text:
            if author not in agent_responses:
                agent_responses[author] = ""
            agent_responses[author] += text
            
            # Print real-time updates
            print(f"\n--- Event from [{author}] ---")
            print(text.strip())

    print("\n" + "=" * 80)
    print("RUN COMPLETE. WRITING TRAJECTORY LOGS...")
    
    # Save the detailed trace results
    output_log_path = Path("/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/scratch/nz_observability_trace.txt")
    with open(output_log_path, "w", encoding="utf-8") as f:
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write("=" * 80 + "\n")
        f.write("CONSOLIDATED AGENT RESPONSES:\n\n")
        for author, response in agent_responses.items():
            f.write(f"[{author}]:\n{response.strip()}\n\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILED OBSERVABILITY & COMMUNICATIONS TRAJECTORY:\n\n")
        for idx, event in enumerate(run_trace):
            f.write(f"Event {idx + 1} - Author: {event['author']} (Timestamp: {event['timestamp']})\n")
            if event['usage']:
                f.write(f"  Token Usage: {event['usage']}\n")
            for part in event['parts']:
                if part['type'] == 'text':
                    f.write(f"  [Text]:\n{part['value'].strip()}\n")
                elif part['type'] == 'function_call':
                    f.write(f"  [Tool Call]: {part['name']} with args: {part['args']}\n")
                elif part['type'] == 'function_response':
                    f.write(f"  [Tool Response]: {part['name']} returning:\n{part['response']}\n")
            f.write("-" * 40 + "\n")
            
    print(f"Trajectory trace successfully saved to: {output_log_path}")

if __name__ == "__main__":
    run_nz_query()
