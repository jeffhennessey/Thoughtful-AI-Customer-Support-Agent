# Thoughtful AI Customer Support Agent

This project is a Python-based AI-powered customer support agent built for Thoughtful AI, designed to answer questions about its automation agents (e.g., EVA, CAM, PHIL). It uses a predefined knowledge base for common queries and falls back to web search for broader questions, ensuring accurate and responsive answers.

## Features
- **Non-Exact Question Matching**: Uses FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`) for robust, similarity-based matching of user questions to predefined answers.
- **Web Search Fallback**: Integrates LangChain with DuckDuckGo for real-time web searches (site-restricted to thoughtful.ai when toggled), handling queries not in the knowledge base.
- **User-Friendly UI**: Built with Streamlit, featuring a chat interface, persistent chat history, and a sidebar for settings (OpenAI API key, toggle for predefined answers).
- **Robust Error Handling**: Validates inputs (e.g., max 500 characters, non-empty), handles missing API keys, and gracefully manages exceptions with fallback messages.
- **Loading State**: Displays a spinner during processing for better UX.

## Setup Instructions
1. Fork or view this repo on GitHub.
2. Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.
3. In the sidebar, enter an OpenAI API key (optional; required for web search fallback; predefined answers work without it).
4. Test the hosted app:
   - Ask "What does EVA do?" (checkbox checked) for a predefined response.
   - Uncheck the checkbox and ask "Thoughtful AI mission" for a web search-based answer.
   - Try edge cases (e.g., empty input, long input) to see error handling.

## Technologies
- **UI**: Streamlit (chat interface, sidebar)
- **Data**: Pandas (managing predefined Q&A dataset)
- **AI**: LangChain (embeddings, agents, DuckDuckGo search), FAISS (vector store), HuggingFace (embeddings), OpenAI (LLM for web search)
- **Environment**: Python 3.11
- **Development Time**: Built in ~25-30 minutes per the challenge constraints.

## Edge Cases Handled
- **Input Validation**: Rejects empty or overly long inputs (>500 characters).
- **No API Key**: Warns user and uses predefined answers only.
- **Poor Matches**: Falls back to web search with a clear note when similarity score exceeds threshold (0.4).
- **Errors**: Catches exceptions and displays user-friendly messages.

## Notes
- Hosted on Streamlit Cloud for easy review without local setup.
- Predefined answers work without an API key, ensuring accessibility.
- Optimized for minimal storage usage.

For issues, contact me.