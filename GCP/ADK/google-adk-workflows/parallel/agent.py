"""
Parallel Agent
Orchestrates parallel execution of travel agents for maximum efficiency.
"""

from google.adk.agents import  ParallelAgent, SequentialAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all agents from common subagent file
from subagent import flight_agent, hotel_agent, sightseeing_agent, trip_summary_agent

plan_parallel = ParallelAgent(
    name="ParallelTripPlanner",
    sub_agents=[flight_agent, hotel_agent],
    description="Fetch flight and hotel information parallely. Each sub-agent will return a JSON response with their respective details."
)


# Main parallel workflow
root_agent = SequentialAgent(
    name="ParallelWorkflow",
    description="Orchestrates parallel execution of travel planning tasks",
    sub_agents=[
        sightseeing_agent,  
        plan_parallel,
        trip_summary_agent     
    ]
) 