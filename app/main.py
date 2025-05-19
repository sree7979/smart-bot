from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from .chatbot.agent import app as chatbot_agent # Import the compiled LangGraph agent

# Define the request body model
class ChatRequest(BaseModel):
    message: str

# Define the response body model
class ChatResponse(BaseModel):
    response: str

app = FastAPI()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to interact with the chatbot.
    """
    user_message = request.message

    # Create a HumanMessage from the user's input
    inputs = {"messages": [HumanMessage(content=user_message)]}

    # Invoke the LangGraph agent
    # We are getting the final message from the agent's output stream
    # In a real application, you might want to handle the stream differently
    response = None
    for output in chatbot_agent.stream(inputs):
        for key, value in output.items():
            # Assuming the final response is in the 'messages' key from the 'call_llm' node
            if key == 'call_llm' and 'messages' in value:
                 # Get the content of the last message from the LLM
                 last_message = value['messages'][-1]
                 response = last_message.content
                 # We can break here assuming the last output from call_llm is the final response
                 break
        if response is not None:
            break


    if response is None:
        response = "Sorry, I could not generate a response."

    return ChatResponse(response=response)

# To run the FastAPI application:
# 1. Make sure you are in the smart-chatbot directory.
# 2. Make sure your GOOGLE_API_KEY environment variable is set.
# 3. Run the following command:
#    uvicorn app.main:app --reload
