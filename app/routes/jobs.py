from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Optional
import json

from app.schemas import JobMatchRequest, JobMatchResponse
from app.services.pdf_service import extract_text_from_pdf
from app.services.orchestrator_service import app_graph

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs & Matching"])


@router.post("/scrape-and-match", response_model=List[JobMatchResponse])
async def evaluate_jobs_from_text(request: JobMatchRequest):
    """Trigger job matching using plain-text resume input."""
    return await _process_job_matching(request.resume_text, request.target_urls)


@router.post("/upload-and-match", response_model=List[JobMatchResponse])
async def evaluate_jobs_from_pdf(
    file: UploadFile = File(...),
    target_urls: str = Form(..., description="JSON array of target URLs, e.g. ['https://...']")
):
    """Trigger job matching using an uploaded PDF resume."""
    # Parse target_urls from form-data string
    try:
        urls_list = json.loads(target_urls)
        if not isinstance(urls_list, list):
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="target_urls must be a valid JSON list of strings.")

    # 1. Delegate PDF extraction to PDF Service
    resume_text = await extract_text_from_pdf(file)

    # 2. Delegate workflow execution to Orchestrator
    return await _process_job_matching(resume_text, urls_list)


async def _process_job_matching(resume_text: str, urls: List[str]) -> List[JobMatchResponse]:
    """Helper method executing the LangGraph pipeline for a resume against URLs."""
    results = []
    
    for url in urls:
        initial_state = {
            "target_url": url,
            "resume_text": resume_text
        }
        
        try:
            final_state = await app_graph.ainvoke(initial_state)
            
            if final_state.get("error_message"):
                results.append(JobMatchResponse(
                    url=url,
                    status=f"FAILED: {final_state['error_message']}",
                    db_id=None,
                    match_score=None,
                    missing_skills=None,
                    reasoning=None
                ))
                continue
            
            eval_data = final_state["evaluation"]
            results.append(JobMatchResponse(
                url=url,
                status="SUCCESS",
                db_id=final_state.get("job_db_id"),
                match_score=eval_data.fit_score,
                missing_skills=eval_data.missing_skills,
                reasoning=eval_data.reasoning
            ))
            
        except Exception as e:
            results.append(JobMatchResponse(
                url=url,
                status=f"SYSTEM_ERROR: {str(e)}",
                db_id=None,
                match_score=None,
                missing_skills=None,
                reasoning=None
            ))
            
    return results