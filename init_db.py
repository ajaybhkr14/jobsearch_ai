# init_db.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    
    # Set dimensions based on provider
    if provider == "gemini":
        vector_dim = 768
        print("📐 Vector Provider: Gemini (768 dimensions)")
    else:
        vector_dim = 1536
        print("📐 Vector Provider: OpenAI (1536 dimensions)")

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )
    cursor = conn.cursor()

    # 1. Enable vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Re-create or create table with dynamic dimension
    # (If table exists with wrong dimension, we drop and recreate)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            company_name VARCHAR(255) NOT NULL,
            salary_min INT,
            salary_max INT,
            is_remote BOOLEAN DEFAULT FALSE,
            required_skills TEXT[],
            raw_markdown TEXT,
            embedding vector({vector_dim}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. Create HNSW Index
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw_idx 
        ON jobs USING hnsw (embedding vector_cosine_ops);
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ PostgreSQL initialized with pgvector ({vector_dim}d) and HNSW index!")

if __name__ == "__main__":
    init_db()