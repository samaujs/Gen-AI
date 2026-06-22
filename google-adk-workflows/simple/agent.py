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
    instruction="""You are a comprehensive trip planner coordinator.
    Based on the user request, you must sequentially invoke the sub-agents one by one to gather all necessary trip details:
    1. First, call `transfer_to_agent` with `FlightAgent` to search and book flights.
    2. Once `FlightAgent` has responded with the flights, call `transfer_to_agent` with `HotelAgent` to book accommodation.
    3. Once `HotelAgent` has responded with the hotel details, call `transfer_to_agent` with `WeatherAgent` to check destination weather.
    4. Once `WeatherAgent` has responded with the weather details, call `transfer_to_agent` with `SightseeingAgent` to get sightseeing recommendations.
    5. Once all four sub-agents have responded, compile all the information (flights, hotel, weather, and sightseeing) into a single, beautifully structured 15-day travel itinerary with daily details.
    
    Ensure all user requirements are met. Do not ask the user clarifying questions; make reasonable assumptions for missing details.
    """,
    sub_agents=[flight_agent, hotel_agent, sightseeing_agent, weather_agent] # The coordinator manages these sub-agents
) 
