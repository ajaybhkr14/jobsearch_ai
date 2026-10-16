import os
import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import both LangChain model wrappers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.schemas import JobPostingSchema, JobEvaluation
from app.services.scraper_service import scrape_job_board
from app.services.vector_service import save_job_to_db

load_dotenv()

# ==========================================
# 1. THE LLM FACTORY (Dynamic Model Selection)
# ==========================================
def get_llm():
    """
    Reads the .env file and returns the correct LangChain model.
    If you switch LLM_PROVIDER in the future, the rest of the code doesn't change.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        print("🧠 Using LLM: OpenAI (gpt-4o-mini)")
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        print("🧠 Using LLM: Google Gemini (gemini-2.0-flash)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}. Use 'gemini' or 'openai'.")


# ==========================================
# 2. DEFINING THE GRAPH MEMORY (State)
# ==========================================
class WorkflowState(TypedDict):
    """
    This is the 'suitcase' that gets carried from node to node.
    Each node opens it, reads what it needs, and puts new data inside.
    """
    target_url: str
    resume_text: str
    scraped_job: Optional[JobPostingSchema]
    job_db_id: Optional[int]
    evaluation: Optional[JobEvaluation]
    error_message: Optional[str]


# ==========================================
# 3. THE WORKER NODES (The Execution Steps)
# ==========================================

async def node_scrape(state: WorkflowState):
    """Node 1: Visits the website and extracts the data into JSON."""
    print(f"➡️ NODE 1: Scraping {state['target_url']}")
    
    # Call our browser-use scraper from Phase 3
    job_data = await scrape_job_board(state["target_url"])
    
    if job_data is None:
        # If scraping fails, put an error in the suitcase
        return {"error_message": "Scraping failed or was blocked."}
    
    # If successful, pack the Pydantic object into the suitcase
    return {"scraped_job": job_data}


async def node_save_to_db(state: WorkflowState):
    """Node 2: Takes the scraped JSON and saves it + its vectors into Postgres."""
    print("➡️ NODE 2: Generating Vectors and Saving to Database...")
    job_data = state["scraped_job"]
    
    # Call our Postgres asyncpg function from Phase 2
    new_id = await save_job_to_db(job_data)
    
    # Pack the new database ID into the suitcase
    return {"job_db_id": new_id}


async def node_evaluate(state: WorkflowState):
    """Node 3: The LLM reads the resume and the job text to score the match."""
    print("➡️ NODE 3: Evaluating candidate fit using LLM...")
    job_data = state["scraped_job"]
    resume = state["resume_text"]
    
    # 1. Get the correct LLM based on the .env file
    llm = get_llm()
    
    # 2. Force the LLM to output our exact Pydantic schema
    structured_llm = llm.with_structured_output(JobEvaluation)
    
    # 3. Create the prompt for the AI
    prompt = f"""
    You are an expert technical recruiter. 
    Compare this candidate's resume to the job description below.
    
    CANDIDATE RESUME:
    {resume}
    
    JOB TITLE: {job_data.title}
    JOB DESCRIPTION:
    {job_data.raw_markdown}
    
    Analyze the requirements and determine the fit score (0-100), missing skills, and a 1-sentence reasoning.
    """
    
    # 4. Trigger the LLM and get the validated Python object back
    evaluation: JobEvaluation = await structured_llm.ainvoke(prompt)
    
    print(f"✅ Evaluation Complete! Match Score: {evaluation.fit_score}/100")
    
    # Pack the final report into the suitcase
    return {"evaluation": evaluation}


# ==========================================
# 4. THE ROUTER (Traffic Control)
# ==========================================
def route_after_scrape(state: WorkflowState) -> str:
    """Checks the suitcase. If there's an error, abort. If not, continue to DB."""
    if state.get("error_message"):
        print("🚦 ROUTER: Error detected. Aborting workflow.")
        return "end_workflow"
    print("🚦 ROUTER: Scrape successful. Proceeding to database.")
    return "continue_to_db"


# ==========================================
# 5. WIRING IT ALL TOGETHER
# ==========================================
def build_graph():
    """Compiles the nodes and edges into an executable state machine."""
    workflow = StateGraph(WorkflowState)
    
    # 1. Add the worker functions as nodes
    workflow.add_node("scrape", node_scrape)
    workflow.add_node("save_db", node_save_to_db)
    workflow.add_node("evaluate", node_evaluate)
    
    # 2. Where does the program start?
    workflow.set_entry_point("scrape")
    
    # 3. Where does it go after scraping? (Depends on the router)
    workflow.add_conditional_edges(
        "scrape",
        route_after_scrape,
        {
            "continue_to_db": "save_db",
            "end_workflow": END  # Special LangGraph command to stop completely
        }
    )
    
    # 4. Simple linear steps
    workflow.add_edge("save_db", "evaluate")
    workflow.add_edge("evaluate", END)
    
    # 5. Compile into an executable app
    return workflow.compile()

# This variable 'app_graph' is what our FastAPI server will import and run
app_graph = build_graph()