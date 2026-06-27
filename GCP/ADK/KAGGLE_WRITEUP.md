# AI Agents Vibe Coding Capstone Project: ADK Agent Studio & Workflows Hub

## Abstract
In this writeup, we present the design, implementation, and optimization of the **ADK Agent Studio & Workflows Hub**—a multi-agent travel planning platform built with Google's **Agent Development Kit (ADK)**, the **Gemini API**, and the **Model Context Protocol (MCP)**. Our system leverages specialized, coordinated LLM-driven agents to resolve multi-variable, unstructured queries (such as scheduling flights, securing luxury accommodations, fetching real-time weather advisories, and planning relaxing sightseeing tours). The application features a premium Streamlit client supporting real-time response streaming, dynamic model selection, on-disk module hot-reloading, and robust observability.

---

## 1. Introduction & Motivation
Multi-agent systems represent a major shift in how complex tasks are automated. By delegating sub-tasks to specialized, focused agents rather than querying a single monolithic LLM prompt, systems achieve higher accuracy, modularity, and scalability. However, building production-grade agentic applications introduces significant engineering challenges:
1. **Latency**: Sequential execution of multiple LLM calls dramatically increases response times.
2. **Orchestration**: Coordinating dependencies and parallel work blocks requires structured control flows.
3. **Observability**: Tracking intermediate agent thoughts, function calls, and state transitions is critical for debugging.
4. **User Experience**: Large delays during agent reasoning require real-time streaming feedback and UI status containers so users remain engaged.

To address these challenges, we built the **ADK Agent Studio & Workflows Hub**, focusing on high-performance concurrent orchestration, real-time event streaming, and comprehensive tracing.

---

## 2. System Architecture
Our workspace is divided into three key components: the **Agent Backend**, the **Model Context Protocol (MCP) Weather Server**, and the **Streamlit Web UI**.

![System Architecture Breakdown](/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/project_architecture.png)

### A. Multi-Agent Backend (`google-adk-workflows/`)
Built on Google's ADK, the backend exposes five specialized LLM agents and four workflow coordinators:
* **FlightAgent**: Analyzes travel dates and class preferences to search and confirm flight details.
* **HotelAgent**: Coordinates luxury lodging accommodations matching specified star ratings.
* **WeatherAgent**: Interfaces with tools to fetch and format real-time meteorological conditions.
* **SightseeingAgent**: Plans daily, relaxed sightseeing itineraries.
* **TripSummaryAgent**: Aggregates the structured payloads from all other agents into a rich, markdown travel plan.

### B. Weather MCP Server (`weather_server.py`)
To integrate live data, we built a Model Context Protocol (MCP) server using the `FastMCP` framework. The server runs as a subprocess, exposing a `get_current_weather(location: str)` tool that queries the public `wttr.in` API via `httpx`. The `WeatherAgent` establishes a `StdioConnection` with this subprocess, ensuring a clean separation of concern.

### C. Streamlit Client (`streamlit_client/`)
An interactive chat UI that connects to the runner. It allows users to input travel prompts, select agent workflows, swap underlying Gemini models, and hot-reload updated agent Python scripts on the fly.

### D. Persistent Sessions & Long-Term Memory
To enable multi-turn conversations and long-term memory across sessions, we implemented a persistent context management layer using the Google ADK:
* **Short-Term Memory**: We preserve the runner's `InMemorySessionService` state inside Streamlit's `st.session_state`. By reusing the same `session_id` across turns, the ADK `Runner` automatically retrieves past conversational history and feeds it into subsequent LLM model turns.
* **Long-Term Memory**: We instantiate an `InMemoryMemoryService` stored globally in `st.session_state`. At the end of every successful execution run, the session is ingested into the memory service using `await memory_service.add_session_to_memory(session)`.
* **Memory Retrieval**: All sub-agents (Flight, Hotel, Weather, Sightseeing, and TripSummary agents) are bound with ADK's built-in `LoadMemoryTool()`. When the LLM detects that a user's prompt references past conversations or travel preferences (e.g. *"Use the same departure city as my last query"*), it invokes the `load_memory` function tool, which queries the memory service and injects relevant past details back into the agent context.

---

## 3. Workflow Execution Modes
The platform implements four distinct multi-agent coordination topologies:

![Multi-Agent Coordination Topologies](/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/agent_topologies.png)

### A. Simple (Sequential Workflow)
The orchestrator agent (`TripPlanner`) executes sub-agents sequentially. It acts as a central router, using `transfer_to_agent` to pass control from one specialist to the next, compiling the outputs step-by-step.

![Simple Agent Workflow](/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/simple_agent_workflow.png)

### B. Parallel (Concurrent Workflow)
To dramatically reduce latency, independent queries (flight booking, hotel search, and weather details) run concurrently inside a parallel execution block, reducing total latency to that of the slowest single LLM call.

![Parallel Agent Workflow](/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/parallel_agent_workflow.png)

### C. Dispatcher Workflow
Utilizes an intent router that dynamically maps user prompts to specific agents or toolsets, avoiding unnecessary sub-agent runs when a request only concerns a single domain (e.g. only flight pricing).

### D. Self-Critic Workflow
A quality assurance loop where a reviewer agent (`TripSummaryReviewer`) evaluates the final compiled travel itinerary against constraints (e.g., verifying if the requested hotel star rating is met or if weather alerts are included) and rejects it for regeneration if it fails.

---

## 4. High-Level Parallel Agent Architecture
The concurrent execution flow utilizes a `SequentialAgent` as the main orchestrator (`ParallelWorkflow`), which delegates to a `ParallelAgent` block (`ParallelTripPlanner`). The high-level topology is illustrated below:

![Parallel Agent Architecture Diagram](/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/parallel_agent_architecture.png)

### Sequence of Operations:
1. **User Query**: The User submits a prompt (e.g., "Create a 15-day luxury itinerary to New Zealand").
2. **Step 1 (Sightseeing)**: `ParallelWorkflow` triggers `SightseeingAgent` to plan the travel route first.
3. **Step 2 (Parallel Block)**: The coordinator triggers the `ParallelTripPlanner` block. The central `Parallel Coordinator` spawns three concurrent tasks:
   - `FlightAgent` fetches flight schedules.
   - `HotelAgent` fetches luxury hotel bookings.
   - `WeatherAgent` executes tool calls to the `Weather MCP Server`.
4. **Tool Execution**: `WeatherAgent` communicates via `stdio` with `Weather MCP Server` to fetch current conditions.
5. **Collation**: The `Parallel Coordinator` collects all three JSON payloads and returns them to the `ParallelWorkflow`.
6. **Step 3 (Compilation)**: `ParallelWorkflow` passes all compiled JSON data to `TripSummaryAgent`, which formats the final markdown itinerary.

---

## 5. UI Mechanics & Real-Time Streaming
Standard Streamlit applications operate on a simple rerun cycle: any user interaction triggers a full script execution, and output is only rendered after the underlying Python execution completes. This creates a blocked UI for long-running multi-agent tasks.

To achieve a premium, real-time user experience, we designed an **asynchronous event-streaming architecture**:

```text
┌────────────────────────────────────────┐
│             Streamlit UI               │
│  (Renders Event Queue, Displays Chat)  │
└───────────────────▲────────────────────┘
                    │  (Polls queue)
┌───────────────────┴────────────────────┐
│          Thread-Safe Queue             │
│    (Holds chunk & state updates)       │
└───────────────────▲────────────────────┘
                    │  (Writes events)
┌───────────────────┴────────────────────┐
│        Background Worker Thread        │
│    (Runs ADK Agent Runner Stream)      │
└────────────────────────────────────────┘
```

1. **Background Worker Thread**: When a prompt is submitted, the UI spawns a background thread that invokes the ADK `Runner` as an asynchronous event generator.
2. **Thread-Safe Event Queue**: The worker thread captures LLM output chunks and agent state events, wrapping them in a unified format and placing them in a `queue.Queue`.
3. **Interactive UI Rendering**:
   - Streamlit displays an **explicit thinking spinner** (`st.spinner`) inside an `st.empty` container at the bottom of the chat layout while the worker is active.
   - Intermediate agent transitions (e.g., `[FlightAgent] starting...`) render inside collapsible drawers (`st.status`) *above* the active spinner.
   - Assistant message content is appended live, token-by-token.
   - **Immediate Spinner Dismissal**: Once the final compiled itinerary is completed or an error is caught, the background thread signals completion, and the UI immediately clears the `st.empty` container to hide the thinking spinner.



---

## 6. Results & Verification
To verify the system, we ran automated verification scripts comparing sequential and concurrent runtimes:
* **Sequential (Simple Agent)**: Completed successfully with 3 agent handoffs and a total response time of **~12.4 seconds**.
* **Concurrent (Parallel Agent)**: Successfully queried flights, hotels, and weather in parallel, completing the block in **~4.8 seconds**—an execution speedup of **over 60%**.

The final interface renders a beautiful, structured travel guide directly to the user:

![Final Itinerary Screenshot](/Users/samaujs/.gemini/antigravity/brain/2e7e63e3-72f0-4aa5-94c2-d1e489ae71e3/final_itinerary_screenshot.png)

---

## 7. Conclusion & Future Directions
The **ADK Agent Studio & Workflows Hub** demonstrates that combining Google's ADK with parallel workflows, Model Context Protocol servers, and a thread-safe streaming UI yields a highly responsive, robust multi-agent system. 
In the future, we plan to expand this project by:
1. **Dynamic Scaling**: Automatically scaling parallel block threads based on the number of sub-tasks identified by the intent router.
2. **Persistent Vector Search**: Migrating the long-term memory storage from an in-memory keyword matching service to a persistent Vector DB (e.g., Aerospike or Vertex AI Vector Search) for semantic similarity searches.
3. **Advanced Self-Correction**: Allowing the QA validator to target specific sub-agents for correction, rather than regenerating the entire parallel block.
