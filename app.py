import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.documents import Document

# Preprocess text
def preprocess_text(text):
    return text.lower().strip()

# Initialize Streamlit app
st.title("Thoughtful AI Customer Support Agent")
st.write("Ask about Thoughtful AI's automation agents (e.g., EVA, CAM, PHIL) or other topics.")
use_predefined = st.sidebar.checkbox("Use Predefined Answers", value=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load predefined Q&A data
data = {
    "question": [
        "What does EVA do?",
        "What is CAM's role?",
        "Who is PHIL?"
    ],
    "answer": [
        "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time.",
        "CAM automates the claims process, ensuring accurate and timely submission to payers.",
        "PHIL automates payment posting, reconciling payments with claims and patient records."
    ]
}
df = pd.DataFrame(data)
df["question"] = df["question"].apply(preprocess_text)

# Setup embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
docs = [Document(page_content=row["question"], metadata={"answer": row["answer"]}) for _, row in df.iterrows()]
vectorstore = FAISS.from_documents(docs, embeddings)

# Setup web search tool
search = DuckDuckGoSearchRun()
tools = [Tool(name="Web Search", func=search.run, description="Useful for answering questions not in the predefined knowledge base")]

# Setup LLM and agent
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"), openai_proxy=None)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

# Handle user input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not prompt.strip():
        response = "Please enter a valid question."
    elif len(prompt) > 500:
        response = "Input is too long. Please keep it under 500 characters."
    else:
        with st.spinner("Processing..."):
            try:
                if use_predefined:
                    query_embedding = embeddings.embed_query(preprocess_text(prompt))
                    docs = vectorstore.similarity_search_by_vector(query_embedding, k=1)
                    if docs and docs[0].metadata["answer"]:
                        response = docs[0].metadata["answer"]
                    else:
                        response = "No predefined answer found."
                else:
                    if os.getenv("OPENAI_API_KEY"):
                        response = agent.run(f"Search the web for: {prompt} site:thoughtful.ai")
                    else:
                        response = "Web search disabled: No OpenAI API key provided."
            except Exception as e:
                response = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)