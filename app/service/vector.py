# app/vector_service.py
import os
import asyncpg
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from app.schemas import JobPostingSchema
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI Embedding Model
# This requires OPENAI_API_KEY to be set in your .env file
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def generate_embedding(text: str) -> List[float]:
    """
    Takes a string of text and asks OpenAI to convert it into a vector (array of 1536 numbers).
    """
    return embedding_model.embed_query(text)

async def get_db_connection():
    """Establishes an async connection to Postgres."""
    return await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        database=os.getenv("DB_NAME", "postgres")
    )

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