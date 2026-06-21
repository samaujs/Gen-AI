"""
Simple Agent
Basic trip planner coordinator that manages sub-agents.
"""

from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Import all agents from common subagent file
from subagent import flight_agent, hotel_agent, sightseeing_agent, weather_agent

MODEL_GEMINI_3_1_FLASH_LITE = os.getenv('MODEL_NAME', 'gemini-3.1-flash-lite')

# Root agent acting as a Trip Planner coordinator
root_agent = LlmAgent(
    model=MODEL_GEMINI_3_1_FLASH_LITE,

    name="TripPlanner",
    instruction="""
    Acts as a comprehensive trip planner.
    - Use the FlightAgent to find and book flights
    - Use the HotelAgent to find and book accommodation
    - Use the SightseeingAgent to find information on places to visit
    - Use the WeatherAgent to check current weather conditions at the destination
    - Coordinate between all agents to provide complete trip planning
    - Ensure all user requirements are met across flight, hotel, sightseeing, and weather needs
    - Complete the task and provide a final compiled itinerary without asking the user clarifying questions. Make reasonable assumptions for missing details.
    """,
    sub_agents=[flight_agent, hotel_agent, sightseeing_agent, weather_agent] # The coordinator manages these sub-agents
) 
