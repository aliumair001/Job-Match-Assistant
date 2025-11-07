# app.py
import streamlit as st
import requests
import json
import time
import io

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust port if necessary

st.set_page_config(
    page_title="Resume Match Assistant", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "top_jobs" not in st.session_state:
    st.session_state.top_jobs = []
if "resume_context" not in st.session_state:
    st.session_state.resume_context = ""
if "match_report" not in st.session_state:
    st.session_state.match_report = ""

# --- UI Layout ---
st.title(" LangChain Resume Match Assistant")
st.markdown("Upload your resume or paste your resume text to get a detailed match analysis ")

# Sidebar for additional features
with st.sidebar:
    st.header(" Chat with Assistant")
    st.markdown("After analyzing your resume, you can ask questions like:")
    st.markdown("- 'Am I fit for the Data Analyst role?'")
    st.markdown("- 'Which job is best for my experience?'")
    st.markdown("- 'What skills am I missing for ML Engineer?'")
    
    if st.button("Clear Chat History"):
        if st.session_state.session_id:
            try:
                requests.delete(f"{API_BASE_URL}/api/chat-session/{st.session_state.session_id}")
            except:
                pass
        st.session_state.chat_history = []
        st.rerun()

# Tab interface for different input methods
tab1, tab2 = st.tabs(["📁 Upload Resume File", "📝 Paste Resume Text"])

with tab1:
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or Text)", 
        type=["pdf", "txt"], 
        accept_multiple_files=False,
        key="file_uploader"
    )

    if uploaded_file and st.button("Analyze & Match Resume", key="analyze_file"):
        st.info("Analysis in progress... This may take a moment as the system is generating a detailed  report.")
        
        files = {"resume_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        try:
            # Send the file to the FastAPI backend
            response = requests.post(f"{API_BASE_URL}/api/match-resume", files=files)
            
            if response.status_code == 200:
                # Success
                data = response.json()
                
                st.session_state.session_id = data.get('session_id')
                st.session_state.top_jobs = data.get('top_jobs', [])
                st.session_state.match_report = data.get('match_report', '')
                
                # Store resume context
                if uploaded_file.type == "application/pdf":
                    st.session_state.resume_context = "PDF resume uploaded and analyzed"
                else:
                    uploaded_file.seek(0)
                    st.session_state.resume_context = uploaded_file.read().decode("utf-8")
                
                st.success("Analysis Complete!")
                
                # Display match report with streaming effect
                st.markdown("---")
                st.markdown("##  Match Report Summary")
                
                report_placeholder = st.empty()
                full_report = data['match_report']
                displayed_report = ""
                
                # Stream the report text
                for char in full_report:
                    displayed_report += char
                    report_placeholder.markdown(displayed_report)
                    time.sleep(0.01)  # Adjust speed as needed
                
                st.markdown("---")
                
                with st.expander(" See Top 3 Retrieved Job Details"):
                    for i, job in enumerate(data['top_jobs']):
                        st.subheader(f"Job {i+1}: {job.get('title', 'N/A')} (ID: {job.get('job_id', 'N/A')})")
                        st.write(f"**Experience Level:** {job.get('experience_level', 'N/A')}")
                        st.write(f"**Skills Required:** {job.get('skills_required', 'N/A')}")
                        st.write(f"**Description:** {job.get('description', 'N/A')}")
                        
                        # Calculate and display match score
                        if st.session_state.resume_context and st.session_state.resume_context != "PDF resume uploaded and analyzed":
                            try:
                                score_response = requests.post(f"{API_BASE_URL}/api/calculate-score", json={
                                    "resume_text": st.session_state.resume_context,
                                    "job_description": job.get('description', ''),
                                    "skills_required": job.get('skills_required', '')
                                })
                                if score_response.status_code == 200:
                                    score_data = score_response.json()
                                    st.write(f"**Match Score:** {score_data['match_score']}%")
                            except:
                                pass
                        
                        st.markdown("---")
                
            else:
                error_detail = response.json().get("detail", "Unknown error occurred.")
                st.error(f"Backend API Error ({response.status_code}): {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI Backend at {API_BASE_URL}. Ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with tab2:
    # Text input section
    resume_text = st.text_area(
        "Paste your resume text here:",
        height=200,
        placeholder="Paste your resume text here...\nExample: 5 years experience in Python, SQL, Data Analysis. Skilled in machine learning, data visualization, and cloud technologies.",
        key="text_resume"
    )
    
    if st.button("Analyze Resume Text", key="analyze_text") and resume_text:
        st.info("Analysis in progress... This may take a moment as the system is generating a detailed LLM report.")
        
        try:
            # Send the text to the FastAPI backend
            response = requests.post(f"{API_BASE_URL}/api/analyze-resume-text", json={
                "resume_text": resume_text
            })
            
            if response.status_code == 200:
                # Success
                data = response.json()
                
                st.session_state.session_id = data.get('session_id')
                st.session_state.top_jobs = data.get('top_jobs', [])
                st.session_state.match_report = data.get('match_report', '')
                st.session_state.resume_context = resume_text
                
                st.success("Analysis Complete!")
                
                # Display match report with streaming effect
                st.markdown("---")
                st.markdown("## 📄 Match Report Summary")
                
                report_placeholder = st.empty()
                full_report = data['match_report']
                displayed_report = ""
                
                # Stream the report text
                for char in full_report:
                    displayed_report += char
                    report_placeholder.markdown(displayed_report)
                    time.sleep(0.01)  # Adjust speed as needed
                
                st.markdown("---")
                
                with st.expander(" See Top 3 Retrieved Job Details"):
                    for i, job in enumerate(data['top_jobs']):
                        st.subheader(f"Job {i+1}: {job.get('title', 'N/A')} (ID: {job.get('job_id', 'N/A')})")
                        st.write(f"**Experience Level:** {job.get('experience_level', 'N/A')}")
                        st.write(f"**Skills Required:** {job.get('skills_required', 'N/A')}")
                        st.write(f"**Description:** {job.get('description', 'N/A')}")
                        
                        # Calculate and display match score
                        try:
                            score_response = requests.post(f"{API_BASE_URL}/api/calculate-score", json={
                                "resume_text": resume_text,
                                "job_description": job.get('description', ''),
                                "skills_required": job.get('skills_required', '')
                            })
                            if score_response.status_code == 200:
                                score_data = score_response.json()
                                st.write(f"**Match Score:** {score_data['match_score']}%")
                        except:
                            pass
                        
                        st.markdown("---")
                
            else:
                error_detail = response.json().get("detail", "Unknown error occurred.")
                st.error(f"Backend API Error ({response.status_code}): {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI Backend at {API_BASE_URL}. Ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Chat Interface (only show after resume analysis)
if st.session_state.session_id and st.session_state.top_jobs:
    st.markdown("---")
    st.markdown("##  Chat with Job Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your job matches, skills, or improvements..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                response = requests.post(f"{API_BASE_URL}/api/chat", json={
                    "session_id": st.session_state.session_id,
                    "message": prompt
                })
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data["answer"]
                    
                    # Stream the response
                    for char in assistant_response:
                        full_response += char
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    
                    # Show sources if available
                    if data.get("sources"):
                        with st.expander("View Source Jobs"):
                            for source in data["sources"]:
                                st.write(f"**{source['job_title']}** (ID: {source['job_id']})")
                                st.write(f"Skills: {source['skills']}")
                                st.write(f"Experience: {source['experience']}")
                                st.markdown("---")
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Could not connect to backend. Please ensure the server is running."
                message_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Display session info in sidebar for debugging
with st.sidebar:
    st.markdown("---")
    st.markdown("### Session Info")
    if st.session_state.session_id:
        st.write(f"Session: {st.session_state.session_id[:8]}...")
        st.write(f"Chat messages: {len(st.session_state.chat_history)}")
    else:
        st.write("No active session")