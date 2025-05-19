import streamlit as st
import requests

# Define the URL of your FastAPI backend
# Assuming it's running locally on port 8000
BACKEND_URL = "http://localhost:8000/chat"

st.title("Smart Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send message to backend and get response
    try:
        response = requests.post(BACKEND_URL, json={"message": prompt})
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        backend_response = response.json()["response"]
    except requests.exceptions.RequestException as e:
        backend_response = f"Error communicating with backend: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(backend_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": backend_response})

# Instructions to run the application:
# 1. Make sure you are in the smart-chatbot directory.
# 2. Make sure your GOOGLE_API_KEY environment variable is set.
# 3. Make sure you have run knowledge_base/process.py at least once
#    to create the FAISS index (if you want to use the knowledge base).
# 4. In one terminal, run the FastAPI backend:
#    uvicorn app.main:app --reload
# 5. In another terminal, run the Streamlit frontend:
#    streamlit run frontend/app.py
