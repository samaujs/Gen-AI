"""
Common SubAgents File
Contains hotel_agent, sightseeing_agent, and trip_summary_agent
that can be used by dispatcher, parallel, and self_critic agents.
"""

from google.genai.types import Content, Part
from typing import AsyncGenerator
from google.adk.agents import  LlmAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_GEMINI_3_1_FLASH_LITE = os.getenv('MODEL_NAME', 'gemini-3.1-flash-lite')

# Flight Agent
flight_agent = LlmAgent(
    # Select LLM
    model=MODEL_GEMINI_3_1_FLASH_LITE,
    name="FlightAgent",
    description="Flight booking agent",
    instruction="""You are a flight booking agent.
    - You take any flight booking or confirmation request. Even if the request contains other details (such as hotel or sightseeing), ignore them and focus ONLY on flight details.
    - You check for available flights based on user preferences.
    - You must return ONLY a valid JSON with flight booking and confirmation details, including flight number, departure and arrival times, airline, price, and status. Do not output any conversational text or explanations.
    - If the `transfer_to_agent` tool is available, you must call it to transfer control back to the TripPlanner agent after outputting the JSON.
    - If the user does not provide specific details, make reasonable assumptions about the flight and booking details.
    """
)

# Hotel Agent
hotel_agent = LlmAgent(
    # Select LLM
    model=MODEL_GEMINI_3_1_FLASH_LITE,
    name="HotelAgent",
    description="Hotel booking agent",
    instruction="""You are a hotel booking agent.
    - You take any hotel booking or confirmation request. Even if the request contains other details (such as flight or sightseeing), ignore them and focus ONLY on hotel details.
    - Always return ONLY a valid JSON with hotel booking and confirmation details, including hotel name, check-in and check-out dates, room type, price, and status. Do not output any conversational text or explanations.
    - If the `transfer_to_agent` tool is available, you must call it to transfer control back to the TripPlanner agent after outputting the JSON.
    - If the user does not provide specific details, make reasonable assumptions about the hotel and booking details.
    """
)

# Sightseeing Agent
sightseeing_agent = LlmAgent(
    # Select LLM
    model=MODEL_GEMINI_3_1_FLASH_LITE,
    name="SightseeingAgent",
    description="Sightseeing information agent",
    instruction="""You are a sightseeing information agent.
    - You take any sightseeing request and suggest only the top 2 best places to visit, timings, and any other relevant details. Focus ONLY on sightseeing.
    - Always return ONLY a valid JSON with sightseeing information, including places to visit, timings, and any other relevant details. Do not output any conversational text or explanations.
    - If the `transfer_to_agent` tool is available, you must call it to transfer control back to the TripPlanner agent after outputting the JSON.
    - If the user does not provide specific details, make reasonable assumptions about the sightseeing options available.
    """
)

# Trip Summary Agent
trip_summary_agent = LlmAgent(
    # Select LLM 
    model=MODEL_GEMINI_3_1_FLASH_LITE,
    name="TripSummaryAgent",
    instruction="Summarize the trip details from the flight, hotel, sightseeing, and weather agents. Summarise JSON responses into a single summary document with all trip information like a travel itinerary. The summary should be well-structured and clearly present all trip details in an organized manner using text format only like a travel itinerary. Under the weather update section, include relevant weather emojis (e.g. ☀️, 🌧️, ⛅) to make it visually engaging.",
    output_key="trip_summary"
)

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StdioServerParameters

# Weather MCP Toolset pointing to the local python weather server
weather_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='/Users/samaujs/Year_2026/GenAI/VirtualEnv/gen_ai/bin/python',
            args=['/Users/samaujs/Year_2026/GenAI/samples/google-adk-workflows/weather_server.py']
        )
    )
)

# Weather Agent
weather_agent = LlmAgent(
    model=os.getenv('MODEL_NAME', 'gemini-3.1-flash-lite'),
    name="WeatherAgent",
    description="Weather checking agent",
    instruction="""You are a weather checking agent.
    - Always use the WeatherService tool to check current weather conditions at the destination location.
    - Always return a valid JSON with weather details, including temperature (with thermometer/temperature emojis, e.g. 🌡️), conditions (with weather condition emojis, e.g. ☀️, 🌧️, ⛅), and recommendations for packing or activities (with matching emojis, e.g. 🧥, 🕶️, 🚶) based on user request.
    - If the `transfer_to_agent` tool is available, you must call it to transfer control back to the TripPlanner agent after outputting the JSON.
    - If the user does not provide specific details, make reasonable assumptions about the location.
    """,
    tools=[weather_toolset]
)

 
