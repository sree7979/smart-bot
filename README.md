# Smart Chatbot

A RAG-based smart chatbot built with LangChain, LangGraph, FastAPI, and Streamlit, utilizing Google Generative AI.

## Description

This project implements a smart chatbot that leverages Retrieval Augmented Generation (RAG) to provide informed responses. It uses LangGraph to orchestrate the conversation flow, LangChain for interacting with language models and tools, FastAPI for the backend API, and Streamlit for the interactive frontend. The chatbot can retrieve knowledge from a local knowledge base to answer user queries.

## Features

*   **Retrieval Augmented Generation (RAG):** Integrates external knowledge to provide more accurate and contextually relevant responses.
*   **LangGraph Orchestration:** Uses LangGraph to manage the conversation states and tool usage.
*   **FastAPI Backend:** Provides a robust and scalable API for chatbot interaction.
*   **Streamlit Frontend:** Offers an easy-to-use and interactive chat interface.
*   **Google Generative AI:** Utilizes Google's powerful language models for generating responses.

## Setup

To set up and run the smart chatbot locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sree7979/smart-bot.git
    cd smart-bot
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google API Key:**

    Obtain a Google API Key from the Google Cloud Console. Set it as an environment variable:

    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

    Replace `"YOUR_API_KEY"` with your actual API key.

5.  **Process the knowledge base:**

    Run the processing script to create the FAISS index from the knowledge base data. This is necessary for the RAG functionality.

    ```bash
    python knowledge_base/process.py
    ```

    Make sure you have added your knowledge data to `knowledge_base/data/my_ai_knowledge.txt`.

## Usage

To run the smart chatbot, you need to start both the FastAPI backend and the Streamlit frontend.

1.  **Start the FastAPI backend:**

    Open a terminal in the `smart-bot` directory and run:

    ```bash
    uvicorn app.main:app --reload
    ```

    The backend will run on `http://localhost:8000`.

2.  **Start the Streamlit frontend:**

    Open a *new* terminal in the `smart-bot` directory and run:

    ```bash
    streamlit run frontend/app.py
    ```

    The Streamlit app will open in your web browser. You can now interact with the smart chatbot through the web interface.


