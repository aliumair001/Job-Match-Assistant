# system_prompt.py

SYSTEM_PROMPT = """
You are an expert Resume Reviewer and Job Match Assistant. Your role is to analyze resumes against job descriptions and provide detailed, actionable feedback.

**CRITICAL INSTRUCTIONS:**
1. Always compare the candidate's resume against the provided job context
2. Be specific about matched skills and missing requirements
3. Provide concrete improvement suggestions
4. Recommend the best fitting job with clear reasoning
5. Maintain professional but helpful tone

**FINAL OUTPUT STRUCTURE TEMPLATE:**
## Match Analysis Summary

###  Matched Skills:
- [List specific skills from resume that match job requirements]

###  Missing/Weak Areas:
- [List missing skills or experience gaps]

###  Improvement Suggestions:
- [Actionable advice to improve job fit]

###  Recommended Job Fit:
- [Clear recommendation with reasoning]

**Always reference specific job titles and requirements in your analysis.**
"""

CHAT_MEMORY_STRATEGY_DESCRIPTION = """
The Chat Memory strategy is designed for a multi-turn conversation in the optional Streamlit Chat Interface.
Since the core task is a Retrieval-Augmented Generation (RAG) process over retrieved job documents, the memory must balance short-term history with long-term persistence of the initial match context.

**Implementation Details (Focus on LangChain):**

1.  **Initial Match Context (Permanent RAG Base):** The output from the initial process (Resume Upload -> Top 3 Jobs -> Match Report) establishes the core context. The **FAISS Vector Store** acts as the *permanent long-term memory* of the job data.

2.  **Conversation Buffer (Short-Term Memory):** A **ConversationBufferWindowMemory** with a small window size (e.g., K=5) is used.
    * **Goal:** To retain the last few turns of dialogue, allowing the user to ask follow-up questions without repeating the previous question's context.
    * **Benefit:** Prevents the prompt from growing too large and consuming unnecessary tokens.

3.  **Dynamic Context Injection (Conversational Retrieval Chain):** For every new user question in the chat, a **ConversationalRetrievalChain** is employed.
    * **Step 1: Condensation:** The user's new question and the short-term history are processed by a small LLM chain (`CondenseQuestionChain`). This creates a *standalone, context-aware query* (e.g., "Given the user's last question was about the Data Analyst role, and their history shows they are a Python developer, how do their skills match the Data Analyst job?").
    * **Step 2: Re-Retrieval:** This condensed query is used to search the **FAISS Vector Store** again. This ensures that the response to the follow-up question is always grounded in the specific, relevant job documents.
    * **System Prompt Persistence:** The detailed **SYSTEM_PROMPT** (from `system_prompt.py`) is injected on *every turn* to maintain the LLM's role, tone, and strict output constraints.
"""