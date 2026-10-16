# app/vector_service.py
import os
import asyncpg
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.schemas import JobPostingSchema
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Dynamic Embedding Factory
# ==========================================
def get_embedding_model():
    """
    Returns the appropriate embedding model based on LLM_PROVIDER in .env.
    Note: Ensure vector dimensions match your DB schema.
    - OpenAI text-embedding-3-small outputs 1536 dimensions.
    - Gemini text-embedding-004 outputs 768 dimensions.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        print("🔢 Using Embedding Model: OpenAI (text-embedding-3-small)")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        print("🔢 Using Embedding Model: Google Gemini (models/text-embedding-004)")
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'.")


# Initialize global embedding model instance
embedding_model = get_embedding_model()


def generate_embedding(text: str) -> list[float]:
    """Generates vector embeddings for a given text using the selected provider."""
    return embedding_model.embed_query(text)

db_pool = None

async def init_connection_pool():
    """Creates a pool of reusable database connections."""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            database=os.getenv("DB_NAME", "postgres"),
            min_size=1,
            max_size=10
        )
    return db_pool

async def save_job_to_db(job: JobPostingSchema) -> int:
    """
    Saves the extracted job data and its vector embedding into Postgres.
    """
    conn = await get_db_connection()
    try:
        # We combine the title, skills, and full description so the vector captures the entire context
        text_to_embed = f"{job.title} at {job.company_name}. Skills: {' '.join(job.required_skills)}. {job.raw_markdown}"
        
        # Call OpenAI to get the numbers
        vector = generate_embedding(text_to_embed)
        
        query = """
            INSERT INTO jobs (
                title, company_name, salary_min, salary_max, 
                is_remote, required_skills, raw_markdown, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id;
        """
        
        # Execute the SQL insertion
        # str(vector) formats the Python list into '[0.12, -0.45, ...]' which Postgres understands
        job_id = await conn.fetchval(
            query,
            job.title,
            job.company_name,
            job.salary_min,
            job.salary_max,
            job.is_remote,
            job.required_skills,
            job.raw_markdown,
            str(vector) 
        )
        return job_id
    finally:
        await conn.close()

async def find_matching_jobs(resume_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Converts your resume into a vector and asks Postgres to find the mathematically closest jobs.
    """
    conn = await get_db_connection()
    try:
        # First, turn your resume into numbers
        resume_vector = generate_embedding(resume_text)
        
        # The <=> operator computes Cosine Distance in pgvector
        # A smaller distance means the meaning is closer.
        query = """
            SELECT id, title, company_name, salary_min, required_skills,
                   (embedding <=> $1) AS similarity_distance
            FROM jobs
            ORDER BY similarity_distance ASC
            LIMIT $2;
        """
        
        # Run the search
        rows = await conn.fetch(query, str(resume_vector), limit)
        
        # Convert Postgres rows into standard Python dictionaries
        return [dict(row) for row in rows]
    finally:
        await conn.close()