"""
Seed data for Agent Replay demo.
Generates realistic agent sessions for demonstration purposes.
"""
import asyncio
import uuid
from typing import List
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import AsyncSessionLocal, init_db
from app.models import AgentSession, SessionStep, Replay, ReplayStep
from app.schemas import SessionCreate, SessionStepCreate, ReplayCreate, ReplayConfig, Metadata
from app.engine.recorder import SessionRecorder
from app.engine.replay_engine import ReplayEngine


async def create_debug_session(db: AsyncSession) -> AgentSession:
    """Create a debugging agent session."""
    recorder = SessionRecorder(db)
    
    # Create session
    session = await recorder.create_session(SessionCreate(
        name="Debugging Python ImportError",
        agent_id="coding-agent-1",
        model="claude-3.5-sonnet",
        metadata={
            "task": "Debug ImportError in main.py",
            "environment": "development",
            "language": "python"
        }
    ))
    
    # Record steps
    steps = [
        SessionStepCreate(
            step_number=1,
            step_type="think",
            content="I need to debug this ImportError in main.py. Let me first examine the error message.",
            metadata=Metadata(tokens_used=45, latency_ms=320)
        ),
        SessionStepCreate(
            step_number=2,
            step_type="tool_call",
            content="Reading file: main.py",
            metadata=Metadata(
                tool_name="read_file",
                tool_args={"path": "main.py"},
                tokens_used=28,
                latency_ms=150
            )
        ),
        SessionStepCreate(
            step_number=3,
            step_type="tool_result",
            content="File main.py contents:\n```python\nimport non_existent_module\n\nprint('Hello World')\n```",
            metadata=Metadata(
                tool_name="read_file",
                tokens_used=120,
                latency_ms=80
            )
        ),
        SessionStepCreate(
            step_number=4,
            step_type="observation",
            content="I can see the issue: the code is trying to import 'non_existent_module' which doesn't exist.",
            metadata=Metadata(tokens_used=35, latency_ms=210)
        ),
        SessionStepCreate(
            step_number=5,
            step_type="think",
            content="Let me check what modules are available in the environment.",
            metadata=Metadata(tokens_used=30, latency_ms=180)
        ),
        SessionStepCreate(
            step_number=6,
            step_type="tool_call",
            content="Listing installed Python packages",
            metadata=Metadata(
                tool_name="run_command",
                tool_args={"command": "pip list"},
                tokens_used=25,
                latency_ms=160
            )
        ),
        SessionStepCreate(
            step_number=7,
            step_type="tool_result",
            content="Installed packages:\n- numpy 1.24.0\n- pandas 2.0.0\n- requests 2.31.0",
            metadata=Metadata(
                tool_name="run_command",
                tokens_used=90,
                latency_ms=120
            )
        ),
        SessionStepCreate(
            step_number=8,
            step_type="observation",
            content="The module 'non_existent_module' is not in the installed packages.",
            metadata=Metadata(tokens_used=28, latency_ms=190)
        ),
        SessionStepCreate(
            step_number=9,
            step_type="think",
            content="Maybe it's a typo. Let me check if there's a similar module or if it should be installed.",
            metadata=Metadata(tokens_used=38, latency_ms=240)
        ),
        SessionStepCreate(
            step_number=10,
            step_type="tool_call",
            content="Searching for similar modules",
            metadata=Metadata(
                tool_name="search",
                tool_args={"query": "python non_existent_module alternative"},
                tokens_used=32,
                latency_ms=280
            )
        ),
        SessionStepCreate(
            step_number=11,
            step_type="tool_result",
            content="Search results show no module named 'non_existent_module'. It might be a custom module that needs to be created.",
            metadata=Metadata(
                tool_name="search",
                tokens_used=85,
                latency_ms=450
            )
        ),
        SessionStepCreate(
            step_number=12,
            step_type="decision",
            content="I'll create the missing module. First, let me create non_existent_module.py with basic functionality.",
            metadata=Metadata(tokens_used=42, latency_ms=310)
        ),
        SessionStepCreate(
            step_number=13,
            step_type="tool_call",
            content="Creating file: non_existent_module.py",
            metadata=Metadata(
                tool_name="write_file",
                tool_args={
                    "path": "non_existent_module.py",
                    "content": "def hello():\n    return 'Hello from non_existent_module!'"
                },
                tokens_used=55,
                latency_ms=220
            )
        ),
        SessionStepCreate(
            step_number=14,
            step_type="tool_result",
            content="File created successfully.",
            metadata=Metadata(
                tool_name="write_file",
                tokens_used=25,
                latency_ms=180
            )
        ),
        SessionStepCreate(
            step_number=15,
            step_type="tool_call",
            content="Running main.py to test",
            metadata=Metadata(
                tool_name="run_command",
                tool_args={"command": "python main.py"},
                tokens_used=22,
                latency_ms=170
            )
        ),
        SessionStepCreate(
            step_number=16,
            step_type="tool_result",
            content="Output: Hello World\nScript ran successfully!",
            metadata=Metadata(
                tool_name="run_command",
                tokens_used=40,
                latency_ms=320
            )
        ),
        SessionStepCreate(
            step_number=17,
            step_type="decision",
            content="Success! The ImportError is fixed. The issue was missing module 'non_existent_module', which I created.",
            metadata=Metadata(tokens_used=48, latency_ms=290)
        ),
        SessionStepCreate(
            step_number=18,
            step_type="think",
            content="I should also update the documentation to mention this new module.",
            metadata=Metadata(tokens_used=32, latency_ms=210)
        ),
        SessionStepCreate(
            step_number=19,
            step_type="tool_call",
            content="Updating README.md",
            metadata=Metadata(
                tool_name="write_file",
                tool_args={
                    "path": "README.md",
                    "content": "Added non_existent_module for main.py compatibility."
                },
                tokens_used=38,
                latency_ms=190
            )
        ),
        SessionStepCreate(
            step_number=20,
            step_type="tool_result",
            content="README updated successfully.",
            metadata=Metadata(
                tool_name="write_file",
                tokens_used=25,
                latency_ms=160
            )
        ),
        SessionStepCreate(
            step_number=21,
            step_type="decision",
            content="Task completed. Debugging successful.",
            metadata=Metadata(tokens_used=22, latency_ms=180)
        )
    ]
    
    await recorder.record_batch_steps(session.id, steps)
    await recorder.complete_session(session.id)
    
    return session


async def create_research_session(db: AsyncSession) -> AgentSession:
    """Create a research agent session."""
    recorder = SessionRecorder(db)
    
    # Create session
    session = await recorder.create_session(SessionCreate(
        name="Research: Quantum Computing Applications",
        agent_id="research-agent-1",
        model="gpt-4-turbo",
        metadata={
            "topic": "Quantum Computing",
            "depth": "introductory",
            "sources": 3
        }
    ))
    
    # Record steps
    steps = [
        SessionStepCreate(
            step_number=1,
            step_type="think",
            content="I need to research quantum computing applications for a general audience.",
            metadata=Metadata(tokens_used=35, latency_ms=280)
        ),
        SessionStepCreate(
            step_number=2,
            step_type="tool_call",
            content="Searching: quantum computing applications 2024",
            metadata=Metadata(
                tool_name="web_search",
                tool_args={"query": "quantum computing practical applications 2024"},
                tokens_used=28,
                latency_ms=420
            )
        ),
        SessionStepCreate(
            step_number=3,
            step_type="tool_result",
            content="Search results:\n1. Drug discovery and material science\n2. Cryptography and cybersecurity\n3. Financial modeling and optimization\n4. Machine learning acceleration",
            metadata=Metadata(
                tool_name="web_search",
                tokens_used=95,
                latency_ms=580
            )
        ),
        SessionStepCreate(
            step_number=4,
            step_type="observation",
            content="Found four main application areas. Let me dive deeper into each.",
            metadata=Metadata(tokens_used=30, latency_ms=190)
        ),
        SessionStepCreate(
            step_number=5,
            step_type="tool_call",
            content="Getting details on drug discovery applications",
            metadata=Metadata(
                tool_name="web_search",
                tool_args={"query": "quantum computing drug discovery molecular simulation"},
                tokens_used=32,
                latency_ms=410
            )
        ),
        SessionStepCreate(
            step_number=6,
            step_type="tool_result",
            content="Quantum computers can simulate molecular interactions at quantum level, potentially accelerating drug discovery by years.",
            metadata=Metadata(
                tool_name="web_search",
                tokens_used=65,
                latency_ms=520
            )
        ),
        SessionStepCreate(
            step_number=7,
            step_type="tool_call",
            content="Getting details on cryptography applications",
            metadata=Metadata(
                tool_name="fetch_url",
                tool_args={"url": "https://example.com/quantum-cryptography"},
                tokens_used=25,
                latency_ms=380
            )
        ),
        SessionStepCreate(
            step_number=8,
            step_type="error",
            content="Error: Could not fetch URL (404 Not Found)",
            metadata=Metadata(
                tool_name="fetch_url",
                error_message="404 Not Found",
                tokens_used=40,
                latency_ms=320
            )
        ),
        SessionStepCreate(
            step_number=9,
            step_type="think",
            content="URL not found. Let me try a different source.",
            metadata=Metadata(tokens_used=28, latency_ms=210)
        ),
        SessionStepCreate(
            step_number=10,
            step_type="tool_call",
            content="Searching: quantum cryptography post-quantum algorithms",
            metadata=Metadata(
                tool_name="web_search",
                tool_args={"query": "post-quantum cryptography algorithms"},
                tokens_used=30,
                latency_ms=430
            )
        ),
        SessionStepCreate(
            step_number=11,
            step_type="tool_result",
            content="Quantum computers threaten current encryption but also enable quantum key distribution for ultra-secure communication.",
            metadata=Metadata(
                tool_name="web_search",
                tokens_used=58,
                latency_ms=510
            )
        ),
        SessionStepCreate(
            step_number=12,
            step_type="observation",
            content="Interesting duality: quantum computing breaks old encryption but enables new, more secure methods.",
            metadata=Metadata(tokens_used=38, latency_ms=240)
        ),
        SessionStepCreate(
            step_number=13,
            step_type="tool_call",
            content="Getting details on financial applications",
            metadata=Metadata(
                tool_name="fetch_url",
                tool_args={"url": "https://finance.example.com/quantum-optimization"},
                tokens_used=26,
                latency_ms=370
            )
        ),
        SessionStepCreate(
            step_number=14,
            step_type="tool_result",
            content="Quantum optimization can solve complex financial modeling problems like portfolio optimization much faster than classical computers.",
            metadata=Metadata(
                tool_name="fetch_url",
                tokens_used=62,
                latency_ms=410
            )
        ),
        SessionStepCreate(
            step_number=15,
            step_type="tool_call",
            content="Getting details on machine learning applications",
            metadata=Metadata(
                tool_name="web_search",
                tool_args={"query": "quantum machine learning QML applications"},
                tokens_used=28,
                latency_ms=440
            )
        ),
        SessionStepCreate(
            step_number=16,
            step_type="tool_result",
            content="Quantum ML could accelerate training of certain models and discover new algorithms, but is still in early research stages.",
            metadata=Metadata(
                tool_name="web_search",
                tokens_used=55,
                latency_ms=500
            )
        ),
        SessionStepCreate(
            step_number=17,
            step_type="think",
            content="Now I have information on all four areas. Let me synthesize this into a coherent summary.",
            metadata=Metadata(tokens_used=38, latency_ms=290)
        ),
        SessionStepCreate(
            step_number=18,
            step_type="decision",
            content="I'll structure the summary by application area, highlighting key benefits and current status.",
            metadata=Metadata(tokens_used=42, latency_ms=310)
        ),
        SessionStepCreate(
            step_number=19,
            step_type="tool_call",
            content="Writing summary document",
            metadata=Metadata(
                tool_name="write_document",
                tool_args={
                    "title": "Quantum Computing Applications Summary",
                    "sections": ["Drug Discovery", "Cryptography", "Finance", "Machine Learning"]
                },
                tokens_used=48,
                latency_ms=350
            )
        ),
        SessionStepCreate(
            step_number=20,
            step_type="tool_result",
            content="Document created successfully with 4 sections summarizing key applications.",
            metadata=Metadata(
                tool_name="write_document",
                tokens_used=45,
                latency_ms=420
            )
        ),
        SessionStepCreate(
            step_number=21,
            step_type="decision",
            content="Research complete. Summary document created with insights on four major application areas.",
            metadata=Metadata(tokens_used=42, latency_ms=280)
        )
    ]
    
    await recorder.record_batch_steps(session.id, steps)
    await recorder.complete_session(session.id)
    
    return session


async def create_multi_agent_session(db: AsyncSession) -> AgentSession:
    """Create a multi-agent collaboration session."""
    recorder = SessionRecorder(db)
    
    # Create session
    session = await recorder.create_session(SessionCreate(
        name="Multi-Agent: Build Weather Dashboard",
        agent_id="coordinator-agent",
        model="claude-3-opus",
        metadata={
            "project": "Weather Dashboard",
            "agents": ["researcher", "developer"],
            "tech_stack": ["React", "FastAPI", "OpenWeather API"]
        }
    ))
    
    # Record steps
    steps = [
        SessionStepCreate(
            step_number=1,
            step_type="think",
            content="I need to coordinate building a weather dashboard. I'll split work between a researcher and a developer.",
            metadata=Metadata(tokens_used=45, latency_ms=350)
        ),
        SessionStepCreate(
            step_number=2,
            step_type="decision",
            content="I'll start by having the researcher find weather APIs, while the developer sets up the project structure.",
            metadata=Metadata(tokens_used=42, latency_ms=310)
        ),
        SessionStepCreate(
            step_number=3,
            step_type="tool_call",
            content="[To Researcher] Find available weather APIs with free tiers",
            metadata=Metadata(
                tool_name="send_message",
                tool_args={
                    "to": "researcher-agent",
                    "message": "Research weather APIs with good free tiers for a dashboard project."
                },
                tokens_used=38,
                latency_ms=220
            )
        ),
        SessionStepCreate(
            step_number=4,
            step_type="tool_call",
            content="[To Developer] Set up React project with TypeScript",
            metadata=Metadata(
                tool_name="send_message",
                tool_args={
                    "to": "developer-agent",
                    "message": "Set up a React TypeScript project for the weather dashboard UI."
                },
                tokens_used=35,
                latency_ms=210
            )
        ),
        SessionStepCreate(
            step_number=5,
            step_type="tool_result",
            content="[From Researcher] Found OpenWeather API (free tier: 60 calls/min), WeatherAPI.com (free: 1M calls/month), and Tomorrow.io (free: 500 calls/day).",
            metadata=Metadata(
                tool_name="receive_message",
                tool_args={"from": "researcher-agent"},
                tokens_used=85,
                latency_ms=480
            )
        ),
        SessionStepCreate(
            step_number=6,
            step_type="tool_result",
            content="[From Developer] React TypeScript project created with Vite. Folder structure: src/components, src/api, src/types.",
            metadata=Metadata(
                tool_name="receive_message",
                tool_args={"from": "developer-agent"},
                tokens_used=75,
                latency_ms=420
            )
        ),
        SessionStepCreate(
            step_number=7,
            step_type="observation",
            content="Good progress. Researcher found APIs, developer set up project. Now need to choose an API and design the dashboard.",
            metadata=Metadata(tokens_used=48, latency_ms=290)
        ),
        SessionStepCreate(
            step_number=8,
            step_type="decision",
            content="I'll choose OpenWeather API for its reliability and decent free tier. Now need UI design.",
            metadata=Metadata(tokens_used=42, latency_ms=320)
        ),
        SessionStepCreate(
            step_number=9,
            step_type="tool_call",
            content="[To Researcher] Get OpenWeather API documentation and example response format",
            metadata=Metadata(
                tool_name="send_message",
                tool_args={
                    "to": "researcher-agent",
                    "message": "Get OpenWeather API docs and example of current weather response format."
                },
                tokens_used=40,
                latency_ms=230
            )
        ),
        SessionStepCreate(
            step_number=10,
            step_type="tool_call",
            content="[To Developer] Design dashboard layout with current weather, forecast, and location search",
            metadata=Metadata(
                tool_name="send_message",
                tool_args={
                    "to": "developer-agent",
                    "message": "Design a clean dashboard layout showing current weather, 5-day forecast, and location search."
                },
                tokens_used=45,
                latency_ms=240
            )
        ),
        SessionStepCreate(
            step_number=11,
            step_type="tool_result",
            content="[From Researcher] OpenWeather API: Current weather endpoint returns temperature, humidity, wind, conditions. Example response provided.",
            metadata=Metadata(
                tool_name="receive_message",
                tool_args={"from": "researcher-agent"},
                tokens_used=68,
                latency_ms=410
            )
        ),
        SessionStepCreate(
            step_number=12,
            step_type="tool_result",
            content="[From Developer] Created Figma mockup: Header with search, main card for current weather, grid for 5-day forecast, footer for details.",
            metadata=Metadata(
                tool_name="receive_message",
                tool_args={"from": "developer-agent"},
                tokens_used=72,
                latency_ms=390
            )
        ),
        SessionStepCreate(
            step_number=13,
            step_type="observation",
            content="Excellent! Have API docs and UI design. Need to implement API integration and build components.",
            metadata=Metadata(tokens_used=42, latency_ms=280)
        ),
        SessionStepCreate(
            step_number=14,
            step_type="decision",
            content="I'll have developer build the components while I write the backend API proxy (to hide API key).",
            metadata=Metadata(tokens_used=48, latency_ms=310)
        ),
        SessionStepCreate(
            step_number=15,
            step_type="tool_call",
            content="[To Developer] Build CurrentWeather component using the mockup design",
            metadata=Metadata(
                tool_name="send_message",
                tool_args={
                    "to": "developer-agent",
                    "message": "Build CurrentWeather component showing temperature, conditions, and basic metrics."
                },
                tokens_used=42,
                latency_ms=220
            )
        ),
        SessionStepCreate(
            step_number=16,
            step_type="tool_call",
            content="Creating FastAPI backend with weather proxy endpoint",
            metadata=Metadata(
                tool_name="write_code",
                tool_args={
                    "language": "python",
                    "file": "backend/main.py",
                    "content": "FastAPI app with /api/weather endpoint"
                },
                tokens_used=85,
                latency_ms=520
            )
        ),
        SessionStepCreate(
            step_number=17,
            step_type="tool_result",
            content="[From Developer] CurrentWeather component built with TypeScript interfaces matching API response.",
            metadata=Metadata(
                tool_name="receive_message",
                tool_args={"from": "developer-agent"},
                tokens_used=62,
                latency_ms=380
            )
        ),
        SessionStepCreate(
            step_number=18,
            step_type="tool_result",
            content="Backend created successfully with proxy endpoint that forwards to OpenWeather API.",
            metadata=Metadata(
                tool_name="write_code",
                tokens_used=48,
                latency_ms=450
            )
        ),
        SessionStepCreate(
            step_number=19,
            step_type="observation",
            content="Components and backend ready. Need to integrate and test the full flow.",
            metadata=Metadata(tokens_used=38, latency_ms=270)
        ),
        SessionStepCreate(
            step_number=20,
            step_type="decision",
            content="I'll have developer complete integration and run tests, while I document the project.",
            metadata=Metadata(tokens_used=45, latency_ms=310)
        ),
        SessionStepCreate(
            step_number=21,
            step_type="tool_call",
            content="[To Developer] Integrate components with backend API and run end-to-end test",
            metadata=Metadata(
                tool_name="send_message",
                tool_args={
                    "to": "developer-agent",
                    "message": "Connect components to backend API and test full weather fetch flow."
                },
                tokens_used=42,
                latency_ms=230
            )
        ),
        SessionStepCreate(
            step_number=22,
            step_type="tool_call",
            content="Writing project documentation",
            metadata=Metadata(
                tool_name="write_document",
                tool_args={
                    "title": "Weather Dashboard Documentation",
                    "sections": ["Setup", "API Keys", "Development", "Deployment"]
                },
                tokens_used=52,
                latency_ms=380
            )
        ),
        SessionStepCreate(
            step_number=23,
            step_type="tool_result",
            content="[From Developer] Integration complete! Dashboard shows real weather data for default location.",
            metadata=Metadata(
                tool_name="receive_message",
                tool_args={"from": "developer-agent"},
                tokens_used=58,
                latency_ms=350
            )
        ),
        SessionStepCreate(
            step_number=24,
            step_type="tool_result",
            content="Documentation written with setup instructions and architecture overview.",
            metadata=Metadata(
                tool_name="write_document",
                tokens_used=42,
                latency_ms=320
            )
        ),
        SessionStepCreate(
            step_number=25,
            step_type="decision",
            content="Project completed successfully! Weather dashboard is functional with documentation.",
            metadata=Metadata(tokens_used=38, latency_ms=290)
        )
    ]
    
    await recorder.record_batch_steps(session.id, steps)
    await recorder.complete_session(session.id)
    
    return session


async def create_replays_for_session(db: AsyncSession, session: AgentSession) -> List[Replay]:
    """Create demo replays for a session."""
    replay_engine = ReplayEngine(db)
    replays = []
    
    # Replay 1: Different model
    replay1 = await replay_engine.create_replay(session.id, ReplayCreate(
        name="Replay with GPT-4",
        start_step=1,
        end_step=10,
        replay_config=ReplayConfig(
            model="gpt-4-turbo",
            temperature=0.7,
            system_prompt="You are an expert debugger with attention to detail."
        )
    ))
    
    # Replay 2: Different temperature
    replay2 = await replay_engine.create_replay(session.id, ReplayCreate(
        name="Replay with lower temperature",
        start_step=5,
        end_step=15,
        replay_config=ReplayConfig(
            model=session.model,
            temperature=0.2,  # More deterministic
            system_prompt="Be concise and focused in your debugging."
        )
    ))
    
    # Replay 3: Different constraints
    replay3 = await replay_engine.create_replay(session.id, ReplayCreate(
        name="Replay with constraints",
        start_step=10,
        end_step=20,
        replay_config=ReplayConfig(
            model=session.model,
            temperature=0.7,
            system_prompt="Debug the ImportError.",
            constraints=["Don't create new files", "Only use existing modules"]
        )
    ))
    
    replays.extend([replay1, replay2, replay3])
    
    # Execute replays
    for replay in replays:
        try:
            await replay_engine.execute_replay(replay.id)
        except Exception as e:
            print(f"Warning: Could not execute replay {replay.id}: {e}")
    
    await db.commit()
    return replays


async def seed_database():
    """Seed the database with demo data."""
    await init_db()
    async with AsyncSessionLocal() as db:
        # Create sessions
        print("Creating debug session...")
        debug_session = await create_debug_session(db)
        
        print("Creating research session...")
        research_session = await create_research_session(db)
        
        print("Creating multi-agent session...")
        multi_agent_session = await create_multi_agent_session(db)
        
        # Create replays for debug session
        print("Creating demo replays...")
        await create_replays_for_session(db, debug_session)
        
        await db.commit()
        
        # Print summary
        sessions_result = await db.execute(select(AgentSession))
        sessions = sessions_result.scalars().all()
        
        replays_result = await db.execute(select(Replay))
        replays = replays_result.scalars().all()
        
        steps_result = await db.execute(select(SessionStep))
        steps = steps_result.scalars().all()
        
        print("\n" + "="*50)
        print("SEED DATA SUMMARY")
        print("="*50)
        print(f"Sessions created: {len(sessions)}")
        print(f"Total steps: {len(steps)}")
        print(f"Replays created: {len(replays)}")
        
        for session in sessions:
            print(f"\n- {session.name}: {session.total_steps} steps, "
                  f"{session.total_decisions} decisions")
        
        print("\nSeed data loaded successfully!")
        print("="*50)


if __name__ == "__main__":
    asyncio.run(seed_database())