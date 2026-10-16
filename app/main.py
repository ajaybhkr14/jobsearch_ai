from fastapi import FastAPI
import uvicorn

from app.routes import jobs
from app.services.vector_service import init_connection_pool

app = FastAPI(
    title="AI Job Matcher API",
    description="Layered AI-agentic backend to parse resumes, scrape jobs, and compute match scores.",
    version="1.0.0"
)

# Register modular routers
app.include_router(jobs.router)


@app.on_event("startup")
async def startup_event():
    """Pre-warm the database connection pool on application launch."""
    await init_connection_pool()


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Service is healthy and ready."}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)