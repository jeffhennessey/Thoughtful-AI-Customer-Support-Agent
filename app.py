import streamlit as st
import pandas as pd
import os
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

# Hardcoded dataset
data = {
    "questions": [
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]
}
df = pd.DataFrame(data["questions"])

# Text preprocessing for robust matching
def preprocess_text(text):
    text = text.lower().strip()
    return re.sub(r'[^\w\s]', '', text)

# Setup embeddings and vector store
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") -- changed to CPU only for Streamlit
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
docs = [Document(page_content=preprocess_text(row["question"]), metadata={"answer": row["answer"]}) for _, row in df.iterrows()]
vectorstore = FAISS.from_documents(docs, embeddings)

# Setup web search tool
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for searching the web. Supports operators like site:."
    )
]

# Streamlit sidebar (no API key input)
st.sidebar.title("Agent Settings")
use_predefined = st.sidebar.checkbox("Use predefined answers", value=True)

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.sidebar.warning("OpenAI API Key not found in environment. Web search disabled; using predefined answers only.")
    llm = None
    agent = None
else:
   # llm = OpenAI(temperature=0.7) - deprecated. Changing to ChatOpenAI
   from langchain_openai import ChatOpenAI

   llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

# Main UI
st.title("Thoughtful AI Customer Support Agent")
st.markdown("Ask about Thoughtful AI. Uses predefined responses or web search based on settings.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            # Fallback to predefined answers if no API key
            try:
                processed_prompt = preprocess_text(prompt)
                similar_docs = vectorstore.similarity_search_with_score(processed_prompt, k=1)
                doc, score = similar_docs[0]
                if score < 0.4:
                    response = doc.metadata["answer"]
                else:
                    response = "No close match found in predefined answers, and web search is disabled (no API key)."
            except Exception as e:
                response = f"An error occurred: {str(e)}. Please try again."
        else:
            try:
                # Input validation
                if len(prompt) > 500:
                    response = "Input is too long. Please keep under 500 characters."
                elif not prompt.strip():
                    response = "Please enter a valid question."
                else:
                    with st.spinner("Processing your question..."):
                        if use_predefined:
                            processed_prompt = preprocess_text(prompt)
                            similar_docs = vectorstore.similarity_search_with_score(processed_prompt, k=1)
                            doc, score = similar_docs[0]
                            if score < 0.4:
                                response = doc.metadata["answer"]
                            else:
                                response = f"This answer was generated using external search: {agent.run(f'Answer using web search: {prompt}')}"
                        else:
                            response = agent.run(f"Use web search with 'site:thoughtful.ai': {prompt}")
            except Exception as e:
                response = f"An error occurred: {str(e)}. Please try again."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})