from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import uuid
from typing import Dict, List
import json

from app.config import settings
from app.database import init_db, close_db
from app.routes import sessions, steps, replay, compare, dashboard
from app.engine.recorder import SessionRecorder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup and shutdown events."""
    # Startup
    print("Starting Agent Replay...")
    await init_db()
    
    if settings.enable_seed_data:
        # Import here to avoid circular imports
        from seed_data import seed_database
        try:
            await seed_database()
            print("Seed data loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load seed data: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down Agent Replay...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="Agent Replay",
    description="AI Agent Decision Playback Engine - Record, Rewind, Replay",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Create templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(sessions.router)
app.include_router(steps.router)
app.include_router(replay.router)
app.include_router(compare.router)
app.include_router(dashboard.router)

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a WebSocket to a specific session."""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnect a WebSocket from a session."""
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast a message to all connections for a session."""
        if session_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.active_connections[session_id].remove(connection)


manager = ConnectionManager()


@app.get("/")
async def root():
    """Root endpoint redirects to dashboard."""
    return {
        "message": "Agent Replay API",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agent-replay",
        "version": "1.0.0"
    }


@app.get("/dashboard")
async def dashboard_page():
    """Serve dashboard HTML page."""
    # This would normally render a template
    # For now, return a simple response
    return {
        "message": "Dashboard endpoint - UI would be served here",
        "ui_endpoints": {
            "sessions": "/static/html/dashboard.html",  # Example
            "timeline": "/static/html/timeline.html"
        }
    }


@app.websocket("/ws/sessions/{session_id}/live")
async def websocket_session_live(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live session streaming."""
    await manager.connect(websocket, session_id)
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "message": "Connected to live session stream"
    })
    
    try:
        while True:
            # Wait for messages from client (if needed)
            data = await websocket.receive_text()
            
            # Process client message (could be commands, filters, etc.)
            try:
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": "now"
                    })
                
                elif message.get("type") == "subscribe":
                    # Client wants to subscribe to specific events
                    await websocket.send_json({
                        "type": "subscribed",
                        "events": message.get("events", ["step_added"])
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)


@app.post("/api/v1/ingest/trace")
async def ingest_trace(trace_data: dict):
    """Endpoint for ingesting traces from various AI platforms."""
    try:
        # This would use the TraceIngestionAdapter to convert trace data
        # For now, return a placeholder response
        return {
            "message": "Trace ingestion endpoint",
            "status": "under_construction",
            "trace_id": str(uuid.uuid4()),
            "steps_created": 0
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


@app.get("/api/v1/export/{session_id}")
async def export_session(session_id: str, format: str = "json"):
    """Export a session in various formats."""
    # This would export the session data
    # For now, return a placeholder
    return {
        "session_id": session_id,
        "format": format,
        "status": "export_would_be_generated_here",
        "download_url": f"/api/v1/export/{session_id}/download?format={format}"
    }


# Exception handlers
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    from fastapi.responses import JSONResponse as JR
    return JR(
        content={
            "error": "Not Found",
            "path": str(request.url.path),
            "message": "The requested resource was not found",
        },
        status_code=404,
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    from fastapi.responses import JSONResponse as JR
    return JR(
        content={
            "error": "Internal Server Error",
            "path": str(request.url.path),
            "message": "An unexpected error occurred",
        },
        status_code=500,
    )


# Middleware for logging (simplified)
@app.middleware("http")
async def log_requests(request, call_next):
    # Simple request logging
    print(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )