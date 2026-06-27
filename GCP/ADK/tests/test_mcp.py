import asyncio
import os
import sys
from pathlib import Path

# Add samples/google-adk-workflows to path
workflows_dir = Path("/Users/samaujs/Year_2026/GenAI/samples/google-adk-workflows")
sys.path.insert(0, str(workflows_dir))

from dotenv import load_dotenv
load_dotenv()
os.environ["MODEL_NAME"] = "gemini-3.5-flash"

try:
    from google.adk.cli.utils.agent_loader import AgentLoader
    from google.adk.runners import Runner
    from google.adk.apps.app import App
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
    from google.genai import types
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

async def main():
    loader = AgentLoader(agents_dir=str(workflows_dir))
    # Load 'self_critic' agent
    agent_or_app = loader.load_agent("self_critic")
    
    app = agent_or_app if isinstance(agent_or_app, App) else App(name="self_critic", root_agent=agent_or_app)
    
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    credential_service = InMemoryCredentialService()
    
    runner = Runner(
        app=app,
        artifact_service=artifact_service,
        session_service=session_service,
        credential_service=credential_service,
    )
    
    session = await session_service.create_session(app_name="self_critic", user_id="cli_user")
    
    prompt = "Suggest places to visit in Tokyo for 3 days and check the current weather details there. Also book a hotel."
    content = types.Content(role='user', parts=[types.Part(text=prompt)])
    
    print(f"Running agent with prompt: {prompt}\n")
    try:
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            # Print intermediate steps and messages
            author = event.author or "System"
            text_content = ""
            if event.content and event.content.parts:
                text_content = "".join([p.text for p in event.content.parts if p.text])
            print(f"[{author}]: {text_content}")
    finally:
        await runner.close()

if __name__ == "__main__":
    asyncio.run(main())
