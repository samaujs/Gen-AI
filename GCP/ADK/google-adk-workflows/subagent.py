"""
Common SubAgents File
Contains hotel_agent, sightseeing_agent, and trip_summary_agent
that can be used by dispatcher, parallel, and self_critic agents.
"""

from google.genai.types import Content, Part
from typing import AsyncGenerator
from google.adk.agents import  LlmAgent
from google.adk.tools.load_memory_tool import LoadMemoryTool
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
    - Only call the `transfer_to_agent` tool if it is explicitly provided in your tools list. If it is not provided, do not attempt to call it.
    - If the user does not provide specific details, make reasonable assumptions about the flight and booking details.
    - If the user query references a previous trip, travel period, or past details (e.g. "first query", "last trip", "same period", "previous trip", "this itinerary plan", "this plan"), you MUST use the `load_memory` tool with a relevant query (e.g. "trip details") to retrieve past session history and extract the dates, destination, or preferences.
    """,
    tools=[LoadMemoryTool()]
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
    - Only call the `transfer_to_agent` tool if it is explicitly provided in your tools list. If it is not provided, do not attempt to call it.
    - If the user does not provide specific details, make reasonable assumptions about the hotel and booking details.
    - If the user query references a previous trip, travel period, or past details (e.g. "first query", "last trip", "same period", "previous trip", "this itinerary plan", "this plan"), you MUST use the `load_memory` tool with a relevant query (e.g. "trip details") to retrieve past session history and extract the dates, destination, or preferences.
    """,
    tools=[LoadMemoryTool()]
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
    - Only call the `transfer_to_agent` tool if it is explicitly provided in your tools list. If it is not provided, do not attempt to call it.
    - If the user does not provide specific details, make reasonable assumptions about the sightseeing options available.
    - If the user query references a previous trip, travel period, or past details (e.g. "first query", "last trip", "same period", "previous trip", "this itinerary plan", "this plan"), you MUST use the `load_memory` tool with a relevant query (e.g. "trip details") to retrieve past session history and extract the dates, destination, or preferences.
    """,
    tools=[LoadMemoryTool()]
)

# Trip Summary Agent
trip_summary_agent = LlmAgent(
    # Select LLM 
    model=MODEL_GEMINI_3_1_FLASH_LITE,
    name="TripSummaryAgent",
    instruction="Summarize the trip details from the flight, hotel, sightseeing, and weather agents. Summarise JSON responses into a single summary document with all trip information like a travel itinerary. The summary should be well-structured and clearly present all trip details in an organized manner using text format only like a travel itinerary. Under the weather update section, include relevant weather emojis (e.g. ☀️, 🌧️, ⛅) to make it visually engaging. If the user query references a previous trip, travel period, or past details (e.g. 'first query', 'last trip', 'same period', 'previous trip', 'this itinerary plan', 'this plan'), you MUST use the `load_memory` tool with a relevant query (e.g. 'trip details') to retrieve past session history and verify dates, destination, or preferences.",
    tools=[LoadMemoryTool()],
    output_key="trip_summary"
)

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StdioServerParameters

# Weather MCP Toolset pointing to the local python weather server
weather_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='/Users/samaujs/Year_2026/GenAI/VirtualEnv/gen_ai/bin/python',
            args=['/Users/samaujs/Year_2026/GenAI/samples/Gen-AI/GCP/ADK/google-adk-workflows/weather_server.py']
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
    - Only call the `transfer_to_agent` tool if it is explicitly provided in your tools list. If it is not provided, do not attempt to call it.
    - If the user does not provide specific details, make reasonable assumptions about the location.
    - If the user query references a previous trip, travel period, or past details (e.g. "first query", "last trip", "same period", "previous trip", "this itinerary plan", "this plan"), you MUST use the `load_memory` tool with a relevant query (e.g. "trip details") to retrieve past session history and extract the dates, destination, or preferences.
    """,
    tools=[weather_toolset, LoadMemoryTool()]
)
