"""
Dispatcher Agent
Routes travel requests to appropriate specialized agents.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all agents from common subagent file
from subagent import flight_agent, hotel_agent, sightseeing_agent, trip_summary_agent

# Convert specialized agents into tools
flight_tool = agent_tool.AgentTool(agent=flight_agent)
hotel_tool = agent_tool.AgentTool(agent=hotel_agent)
sightseeing_tool = agent_tool.AgentTool(agent=sightseeing_agent)


root_agent = LlmAgent(
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash'),
    name="TripPlanner",
    instruction=f"""
   Acts as a comprehensive trip planner.
   - Use the FlightAgent to find and book flights
   - Use the HotelAgent to find and book accommodation
   - Use the SightSeeingAgent to find information on places to visit

   Based on the user request, sequentially invoke the sub-agents to gather all necessary trip details.:
   - Flight details (from FlightAgent)
   - Hotel booking confirmations (from HotelAgent)
   - Sightseeing information (from SightSeeingAgent)

   Ensure the final output is structured and clearly presents all trip details in an organized manner.
   You will generate customer preferences and complete the task without asking too many questions, making reasonable assumptions when necessary.
   """,
    tools=[flight_tool, hotel_tool, sightseeing_tool]
)


# Hi, Please suggest me places to visit in paris in july for honeymoon and book flight from Delhi and hotel for 5 nights from 15th July 2025 