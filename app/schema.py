from pydantic import BaseModel, Field
from typing import Optional, List

class JobPostingSchema(BaseModel):
    """
    This defines the shape of the data we extract from the target job websites.
    """
    title: str = Field(description="The exact job title (e.g., Senior Backend Engineer)")
    company_name: str = Field(description="The name of the hiring company")
    
    # We use Optional because many job postings don't list a salary
    salary_min: Optional[int] = Field(None, description="Minimum yearly salary in USD")
    salary_max: Optional[int] = Field(None, description="Maximum yearly salary in USD")
    
    is_remote: bool = Field(default=False, description="True if the job allows remote work")
    
    required_skills: List[str] = Field(
        default_factory=list, 
        description="List of programming languages, frameworks, or tools required (e.g., Python, AWS, Node.js)"
    )
    
    raw_markdown: str = Field(description="The full, cleaned text of the job description")

class JobEvaluation(BaseModel):
    """
    This defines the shape of the data the LLM returns AFTER evaluating your resume against a job.
    """
    fit_score: int = Field(description="Match score from 0 to 100")
    missing_skills: List[str] = Field(description="Skills required by the job that are missing from the resume")
    recommend_apply: bool = Field(description="True if the score is > 85 and major requirements are met")
    reasoning: str = Field(description="A one-sentence explanation of why the score was given")