# Multi-Agent Trip Planning System

A sophisticated multi-agent workflow built with Google's Agent Development Kit (ADK) that demonstrates the power of specialized AI agents working together to solve complex travel planning tasks.

## 🏗️ Architecture

Instead of building one monolithic "super agent," this system employs a team of specialized AI agents, each expert in their domain:

### Core Agents

All basic agents are consolidated in `subagent.py`:

1. **FlightAgent** - Flight booking specialist
   - Handles flight searches and bookings
   - Returns structured JSON with flight details
   - Makes reasonable assumptions when details are missing

2. **HotelAgent** - Hotel booking specialist  
   - Manages hotel searches and reservations
   - Provides accommodation details in JSON format
   - Handles various room types and booking preferences

3. **SightseeingAgent** - Tourism specialist
   - Recommends top 2 attractions per destination
   - Provides timing and relevant details
   - Focuses on must-see locations

4. **TripSummaryAgent** - Summary compilation specialist
   - Compiles trip details into comprehensive itinerary
   - Creates structured travel summaries
   - Formats information for easy reading

### Orchestration Agents

Each orchestration agent has its own folder with a dedicated `agent.py`:

5. **SimpleAgent** (`simple/`) - Basic trip coordinator
   - Simple sub-agent coordination pattern
   - Direct management of flight, hotel, and sightseeing agents
   - Perfect for straightforward trip planning

6. **DispatcherAgent** (`dispatcher/`) - Intelligent request router
   - Analyzes requests and routes to appropriate specialists
   - Uses agent tools for flexible coordination
   - Handles simple to complex multi-step requests

7. **ParallelAgent** (`parallel/`) - Efficiency optimizer  
   - Runs flight and hotel agents in parallel for speed
   - Sequential execution: sightseeing → parallel(flight+hotel) → summary
   - Maximizes efficiency for independent flight and hotel tasks

8. **SelfCriticAgent** (`self_critic/`) - Quality assurance specialist
   - Same parallel execution as ParallelAgent (flight+hotel in parallel)
   - Adds quality control: trip summary reviewer and validator
   - Ensures output meets quality standards before delivery

### Workflow Patterns

- **Parallel Execution**: Flight and hotel bookings run concurrently for efficiency
- **Sequential Orchestration**: Dependent tasks execute in logical order  
- **Feedback Loops**: Built-in quality assurance and validation
- **State Management**: Agents communicate through shared session state

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google API Key for Gemini models

### Installation

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd adk_workflows
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**:
   ```bash
   # Copy environment template
   cp env.example .env
   
   # Edit .env and add your Google API key
   # Get your key from: https://aistudio.google.com/app/apikey
   ```

5. **Launch the web interface**:
   ```bash
   adk web
   ```
   This will open a web interface where you can choose and test any of the available agents.

## 📋 Configuration

### Environment Variables (.env)

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
MODEL_NAME=gemini-2.0-flash

# Optional
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## 🏃‍♂️ Usage

Once you've completed the installation steps, simply run:

```bash
adk web
```

This opens a web interface where you can:
- Select any of the 4 orchestration agents (Simple, Dispatcher, Parallel, Self-Critic)
- Test different types of trip planning requests
- See how each agent handles various scenarios

### Example Requests You Can Test

- **Simple**: "Find me a flight to Paris"
- **Complex**: "Book a flight to Paris and find a hotel near the Eiffel Tower"  
- **Comprehensive**: "Plan a 3-day trip to Tokyo with flights, accommodation, and sightseeing"

## 🎯 Agent Details

### FlightAgent
- **Purpose**: Specialized flight booking and information
- **Input**: Flight preferences, dates, destinations
- **Output**: JSON with flight details, prices, booking status
- **Features**: Intelligent assumptions for missing details

### HotelAgent  
- **Purpose**: Hotel booking and accommodation management
- **Input**: Location, dates, room preferences
- **Output**: JSON with hotel details, pricing, availability
- **Features**: Room type optimization, location-based suggestions

### SightseeingAgent
- **Purpose**: Tourism recommendations and itinerary planning
- **Input**: Destination, interests, duration
- **Output**: JSON with top 2 attractions, timings, details
- **Features**: Curated recommendations, practical timing info

### TripSummaryAgent
- **Purpose**: Quality assurance and trip compilation
- **Components**:
  - **TripSummaryAgent**: Compiles comprehensive itinerary
  - **TripSummaryReviewer**: Quality check and validation
  - **ValidateTripSummary**: Final approval and feedback
- **Output**: Validated, complete travel itinerary

## 🔄 Workflow Options

### Simple Workflow
```
TripPlanner (root_agent) → Coordinates FlightAgent + HotelAgent + SightseeingAgent
```

### Dispatcher Workflow  
```
DispatcherAgent → Analyzes request → Routes to appropriate tools → Compiles response
```

### Parallel Workflow
```
SightseeingAgent → FlightAgent + HotelAgent (parallel) → TripSummaryAgent
```

### Self-Critic Workflow
```
SightseeingAgent → FlightAgent + HotelAgent (parallel) → TripSummaryAgent → Reviewer → Validator
```

## 🧪 Development

### Project Structure

```
adk_workflows/
├── subagent.py            # All core agents (flight, hotel, sightseeing, trip_summary)
├── simple/
│   └── agent.py           # Basic trip coordinator
├── dispatcher/
│   └── agent.py           # Intelligent request router
├── parallel/
│   └── agent.py           # Parallel execution optimizer
├── self_critic/
│   └── agent.py           # Quality assurance workflow
├── requirements.txt       # Dependencies
├── env.example           # Environment template
├── README.md             # Documentation
└── SETUP_INSTRUCTIONS.md # Setup guide
```

### Adding New Agents

**Core Agents:**
1. Add your new core agent to `subagent.py`
2. Import and use in orchestration agents as needed

**Orchestration Agents:**
1. Create a new folder: `new_orchestrator/`
2. Add `agent.py` with your orchestration logic
3. Import core agents from `subagent.py`





## 🎯 Benefits of Multi-Agent Architecture

1. **Specialization**: Each agent excels in its specific domain
2. **Scalability**: Easy to add new agents or modify existing ones
3. **Maintainability**: Clear separation of concerns
4. **Efficiency**: Parallel execution for independent tasks
5. **Quality**: Built-in review and validation processes
6. **Flexibility**: Modular design allows easy customization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your specialized agent or enhancement
4. Update documentation
5. Submit a pull request

## 📜 License

[Add your license information here]

## 🆘 Support

For issues and questions:
- Open an issue in the repository
- Check the [Google ADK documentation](https://ai.google.dev/adk)
- Review the agent implementation examples

---

**Built with Google Agent Development Kit (ADK)** - Empowering intelligent multi-agent workflows with Gemini. 