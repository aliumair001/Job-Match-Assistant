# core_matcher.py

import os
import tempfile
import pandas as pd
from typing import List, Dict, Any
from io import BytesIO

# ------------------------------------------------------------------------------
# LangChain & AI Components (LCEL & Modern Imports)
# ------------------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 


from langchain_google_genai import ChatGoogleGenerativeAI

# Import system prompt (single master)
from system_prompt import SYSTEM_PROMPT


def _save_temp_file(file_content: bytes, suffix: str = ".pdf") -> str:
    """Save uploaded content bytes to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(file_content)
    return path


def parse_resume(resume_content: bytes) -> str:
    """
    Parses PDF or text resume content into text.
    Falls back to UTF-8 decoding if PDF parsing fails.
    """
    if not resume_content:
        return "No resume content provided."

    is_pdf = resume_content.startswith(b"%PDF")
    resume_text = ""

    if is_pdf:
        temp_file_path = _save_temp_file(resume_content, suffix=".pdf")
        try:
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            resume_text = "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            resume_text = f"Error parsing PDF: {e}. Trying simple text decode..."
            try:
                resume_text = resume_content.decode("utf-8", errors="ignore")
            except Exception:
                resume_text = "Could not parse or decode resume content."
        finally:
            os.remove(temp_file_path)
    else:
        resume_text = resume_content.decode("utf-8", errors="ignore")

    # Clean up and normalize whitespace
    return " ".join(resume_text.split())


# ------------------------------------------------------------------------------
# 2. VECTOR STORE INITIALIZATION (Job Dataset)
# ------------------------------------------------------------------------------

def initialize_vector_store(csv_path: str, embedding_model_name: str = "all-MiniLM-L6-v2") -> FAISS:
    """Loads job dataset, converts to LangChain Documents, and builds FAISS vector store."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Job dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    job_documents = []

    for _, row in df.iterrows():
        content = (
            f"Job Title: {row.get('title', '')}\n"
            f"Job ID: {row.get('job_id', '')}\n"
            f"Experience Level: {row.get('experience_level', '')}\n"
            f"Description: {row.get('description', '')}\n"
            f"Skills Required: {row.get('skills_required', '')}"
        )
        doc = Document(page_content=content, metadata=row.to_dict())
        job_documents.append(doc)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_documents(job_documents, embeddings)

    return vector_store


# ------------------------------------------------------------------------------
# 3. CORE MATCHING LOGIC (Gemini 2.5 Flash + FAISS using LCEL)
# ------------------------------------------------------------------------------

def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

def process_resume_match(
    resume_content: bytes,
    vector_store: FAISS,
    llm_api_key: str
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Core function to process a resume and generate:
      - Match report (from Gemini)
      - Top matched job metadata (from FAISS)
    
    This function uses the modern LCEL pattern.
    """

    # Step 1: Parse resume text
    resume_text = parse_resume(resume_content)
    if resume_text.startswith("Error") or resume_text.startswith("No resume"):
        return f"Error in resume processing: {resume_text}", []

    # Step 2: Retriever setup
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Since LCEL RAG chain runs retrieval internally, we run it separately 
    # to extract the top job metadata for the function's return value.
    top_jobs = retriever.invoke(resume_text)

    if not top_jobs:
        return "No relevant jobs found in dataset.", []
    
    # Extract metadata early for return value
    top_jobs_data = [doc.metadata for doc in top_jobs]

    # Step 3: Gemini LLM setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=llm_api_key,
        temperature=0.2
    )

    # Step 4: Prompt setup
    # LCEL RAG prompt requires two input keys: context and query
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT), 
        ("human",
         "Candidate Resume (Query):\n{query}\n\n"
         "Relevant Job Details (Context):\n{context}\n\n"
         "Generate the match report following the FINAL OUTPUT STRUCTURE TEMPLATE.")
    ])

   
    
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Step 6: Execute the chain using the resume text as the main input (query)
    try:
        report = rag_chain.invoke(resume_text)
        return report, top_jobs_data

    except Exception as e:
        # NOTE: The top_jobs_data is still returned on error if retrieval succeeded.
        return f"Error during LCEL chain execution: {e}", top_jobs_data




if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    JOB_DATA_FILE = "sample_job_dataset - Sheet1.csv"
    EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("❌ ERROR: Gemini API key not found in environment (.env)")
        exit(1)

    print("🔹 Initializing FAISS vector store...")
    try:
        vector_store = initialize_vector_store(JOB_DATA_FILE, EMBED_MODEL)
        print("✅ FAISS index ready!\n")
    except Exception as e:
        print(f"❌ Error initializing vector store: {e}")
        exit(1)


    MOCK_RESUME = (
        "5 years of experience as a Data Engineer. Skilled in Python, SQL, AWS, "
        "ETL pipelines, Docker, CI/CD, and data cleaning with Pandas. "
        "Experience using FastAPI for deploying ML models."
    ).encode("utf-8")

    print("🔹 Processing Resume Match via Gemini (using LCEL)...\n")
    report, jobs = process_resume_match(MOCK_RESUME, vector_store, GEMINI_API_KEY)

    print("🧠 ===== MATCH REPORT =====")
    print(report)
    print("\n💼 ===== TOP JOBS =====")
    for j in jobs:
        print(f"- {j.get('job_id', 'N/A')}: {j.get('title', 'Unknown')} ({j.get('experience_level', '-')})")