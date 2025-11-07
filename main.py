# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import uvicorn

from core_matcher import initialize_vector_store, process_resume_match
from chat_memory_strategy import ChatMemoryManager, calculate_match_score


load_dotenv()  
# Global variables
try:
    vector_store = initialize_vector_store(
        'sample_job_dataset - Sheet1.csv', 
        os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
except Exception as e:
    print(f"Error initializing vector store: {e}")
    vector_store = None

# Chat memory managers by session
chat_managers = {}

app = FastAPI(title="Resume Matcher API", description="LangChain & Gemini Backend")

# Define request/response models
class MatchResponse(BaseModel):
    match_report: str
    top_jobs: List[dict]
    session_id: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    resume_text: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str
    skills_required: str

class ScoreResponse(BaseModel):
    match_score: float

# 2. ENDPOINTS

@app.post("/api/match-resume", response_model=MatchResponse)
async def match_resume(resume_file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not initialized. Check job data file.")

    # Read the uploaded PDF/text file content
    try:
        content = await resume_file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    # Pass the content to the core logic
    try:
        report, jobs_data = process_resume_match(
            resume_content=content,
            vector_store=vector_store,
            llm_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Generate session ID for chat continuity
        import uuid
        session_id = str(uuid.uuid4())
        
        # Store the parsed resume text for chat context
        from core_matcher import parse_resume
        resume_text = parse_resume(content)
        
        # Initialize chat manager for this session with resume context
        chat_managers[session_id] = ChatMemoryManager(
            vector_store=vector_store,
            llm_api_key=os.getenv("GEMINI_API_KEY"),
            resume_context=resume_text
        )
        
        return MatchResponse(
            match_report=report, 
            top_jobs=jobs_data,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")

@app.post("/api/analyze-resume-text", response_model=MatchResponse)
async def analyze_resume_text(request: TextAnalysisRequest):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not initialized. Check job data file.")

    try:
        # Convert text to bytes to use with existing process_resume_match function
        resume_bytes = request.resume_text.encode('utf-8')
        
        report, jobs_data = process_resume_match(
            resume_content=resume_bytes,
            vector_store=vector_store,
            llm_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Generate session ID for chat continuity
        import uuid
        session_id = str(uuid.uuid4())
        
        # Initialize chat manager for this session with resume context
        chat_managers[session_id] = ChatMemoryManager(
            vector_store=vector_store,
            llm_api_key=os.getenv("GEMINI_API_KEY"),
            resume_context=request.resume_text
        )
        
        return MatchResponse(
            match_report=report, 
            top_jobs=jobs_data,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    if request.session_id not in chat_managers:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a resume first.")
    
    chat_manager = chat_managers[request.session_id]
    result = chat_manager.chat(request.message)
    
    # Format source documents
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "job_title": doc.metadata.get("title", "Unknown"),
            "job_id": doc.metadata.get("job_id", "Unknown"),
            "skills": doc.metadata.get("skills_required", ""),
            "experience": doc.metadata.get("experience_level", "")
        })
    
    return ChatResponse(
        answer=result["answer"],
        sources=sources
    )

@app.post("/api/calculate-score", response_model=ScoreResponse)
async def calculate_match_score_endpoint(request: ScoreRequest):
    try:
        score = calculate_match_score(
            request.resume_text,
            request.job_description,
            request.skills_required
        )
        return ScoreResponse(match_score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating score: {e}")

@app.delete("/api/chat-session/{session_id}")
async def clear_chat_session(session_id: str):
    if session_id in chat_managers:
        chat_managers[session_id].clear_memory()
        del chat_managers[session_id]
    return {"message": "Chat session cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)