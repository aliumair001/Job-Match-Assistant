

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from system_prompt import SYSTEM_PROMPT


class ChatMemoryManager:
    def __init__(self, vector_store: FAISS, llm_api_key: str, resume_context: str = "", k: int = 5):
        self.vector_store = vector_store
        self.llm_api_key = llm_api_key
        self.resume_context = resume_context  # Store the resume context
        self.chat_history = []
        self.k = k  # Keep last k messages
    
    def _get_recent_history(self) -> str:
        """Get recent conversation history as formatted string"""
        if not self.chat_history:
            return "No previous conversation."
        
        recent_messages = self.chat_history[-self.k:]
        history_text = ""
        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        return history_text
    
    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        self.chat_history.append({"role": role, "content": content})
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history = []
    
    def chat(self, question: str) -> dict:
        """Process chat question with memory context using LCEL pattern"""
        try:
            # Get recent conversation history
            recent_history = self._get_recent_history()
            
            # Create the LLM (same as core_matcher.py)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.llm_api_key,
                temperature=0.2
            )
            
            retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
            )
            
            def format_docs(docs):
                """Helper function to format retrieved documents"""
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Build conversational prompt with memory AND resume context
            CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=f"RESUME CONTEXT (User's uploaded resume):\n{self.resume_context}\n\nPREVIOUS CONVERSATION:\n{recent_history}"),
                HumanMessage(
                    "Current Question: {question}\n\n"
                    "Relevant Job Details:\n{context}\n\n"
                    "Please answer based on the user's resume context above and the relevant job details."
                )
            ])
            
            # Build the LCEL Chain (same pattern as core_matcher.py)
            conversational_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | CONVERSATIONAL_PROMPT
                | llm
                | StrOutputParser()
            )
            
            # Execute the chain
            answer = conversational_chain.invoke(question)
            
            # Get source documents for reference
            source_docs = retriever.invoke(question)
            
            # Add to history
            self.add_message("user", question)
            self.add_message("assistant", answer)
            
            return {
                "answer": answer,
                "source_documents": source_docs
            }
            
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            self.add_message("assistant", error_msg)
            return {
                "answer": error_msg,
                "source_documents": []
            }


def calculate_match_score(resume_text: str, job_description: str, skills_required: str) -> float:
    """
    Calculate a simple match score between resume and job requirements
    """
    if not resume_text or not job_description:
        return 0.0
        
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()
    skills_lower = skills_required.lower() if skills_required else ""
    
    # Extract skills list
    skills_list = [skill.strip().lower() for skill in skills_lower.split(",")] if skills_lower else []
    
    # Calculate matches
    job_keywords = set(job_lower.split())
    resume_keywords = set(resume_lower.split())
    
    # Basic keyword matching
    job_match = len(job_keywords.intersection(resume_keywords)) / len(job_keywords) if job_keywords else 0
    skills_match = sum(1 for skill in skills_list if skill in resume_lower) / len(skills_list) if skills_list else 0
    
    # Weighted score (skills are more important)
    total_score = (job_match * 0.4) + (skills_match * 0.6)
    
    return min(round(total_score * 100, 2), 100.0)