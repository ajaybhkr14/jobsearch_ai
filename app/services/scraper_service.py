import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from app.schemas import JobPostingSchema

load_dotenv()

# 1. Configure the Persistent Browser
# We reuse your local Chrome installation so websites see your logged-in cookies.
# IMPORTANT: Close your actual Chrome browser before running this script.
browser = Browser(
    config=BrowserConfig(
        headless=False,  # Must be False so anti-bot systems see a real browser rendering
        # Mac path to Chrome. Windows users change to: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)

# 2. Initialize the "Brain" (The LLM)
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


async def scrape_job_board(target_url: str) -> Optional[JobPostingSchema]:
    """
    Directs the AI agent to navigate to a URL, read the job post, and return structured data.
    """
    print(f"🤖 Deploying AI Agent to scrape: {target_url}")
    
    # 3. Formulate the Mission Prompt
    # We don't need to over-explain the fields because our Pydantic Schema descriptions handle that.
    task_prompt = f"""
    1. Navigate directly to this URL: {target_url}
    2. Read the main job description on the page.
    3. Extract the required job details to match the output schema.
    4. If a field like salary or remote status is not explicitly mentioned, leave it empty or default.
    5. Do not click away to other pages; just read the current page and finish.
    """
    
    # 4. Instantiate the Agent
    agent = Agent(
        task=task_prompt,
        llm=get_llm(),
        browser=browser,
        response_model=JobPostingSchema, # Forces the agent to output our exact Python class
    )
    
    try:
        # 5. Execute the Agent Loop
        result = await agent.run()
        
        # 6. Extract and validate the final answer
        if result.is_done():
            # result.final_result() automatically returns a populated JobPostingSchema object
            extracted_job: JobPostingSchema = result.final_result()
            print(f"✅ Successfully extracted data for: {extracted_job.title} at {extracted_job.company_name}")
            return extracted_job
        else:
            print(f"❌ Agent failed to complete the task for {target_url}")
            return None
            
    except Exception as e:
        print(f"⚠️ Error during scraping: {str(e)}")
        return None

# Quick test block to verify it works independently
if __name__ == "__main__":
    async def run_test():
        test_url = "https://www.linkedin.com/jobs/view/3900012345" # Replace with a real active job link
        job_data = await scrape_job_board(test_url)
        if job_data:
            print("\nFinal JSON Output:")
            print(job_data.model_dump_json(indent=2))
            
    asyncio.run(run_test())