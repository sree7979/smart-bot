import os
from typing import List, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from .tools import retrieve_knowledge # Import the retrieval tool

# Set up Google API Key
# Make sure you have the GOOGLE_API_KEY environment variable set
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Uncomment and replace if not using environment variable

# Define the state of the graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda existing, new: existing + new]
    # Add a field to store retrieved knowledge
    knowledge: str = None

# Define the Gemini LLM
# Ensure you have the 'GOOGLE_API_KEY' environment variable set
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# Define the tools the agent can use
tools = [retrieve_knowledge]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the nodes of the graph

def call_llm(state: AgentState):
    """
    Invokes the LLM to generate a response based on the messages and retrieved knowledge.
    """
    messages = state['messages']
    knowledge = state.get('knowledge')

    # Include retrieved knowledge in the prompt if available
    if knowledge:
        system_message = f"You are a helpful assistant. Use the following retrieved knowledge to answer the user's question:\n\n{knowledge}\n\nIf the knowledge does not contain the answer, say so."
        messages = [HumanMessage(content=system_message)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """
    Invokes the tool based on the LLM's decision.
    """
    messages = state['messages']
    last_message = messages[-1]

    # Find the tool call in the last message
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']

    # Invoke the tool
    if tool_name == "retrieve_knowledge":
        # Assuming the tool expects a 'query' argument
        query = tool_args.get('query', messages[-2].content) # Use tool arg or previous human message
        knowledge = retrieve_knowledge.invoke(query)
        return {"knowledge": knowledge}
    else:
        # Handle other tools if added later
        return {"messages": [HumanMessage(content=f"Unknown tool: {tool_name}")]}

# Define the conditional edge logic

def decide_next_step(state: AgentState):
    """
    Decides whether to continue or end the conversation based on the LLM's response.
    If the LLM suggests a tool call, route to the tool node. Otherwise, end.
    """
    last_message = state['messages'][-1]
    # If the LLM has tool calls, route to the tool node
    if last_message.tool_calls:
        return "call_tool"
    else:
        # Otherwise, end the conversation
        return END

# Build the LangGraph graph

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tool", call_tool)

# Set the entry point
workflow.set_entry_point("call_llm")

# Add edges
workflow.add_conditional_edges(
    "call_llm",
    decide_next_step,
    {
        "call_tool": "call_tool",
        END: END
    }
)

# After calling a tool, always return to the LLM to generate a response
workflow.add_edge("call_tool", "call_llm")

# Compile the graph
app = workflow.compile()

# Example usage (for testing the agent independently)
if __name__ == "__main__":
    # Make sure you have run knowledge_base/process.py first,
    # set your GOOGLE_API_KEY environment variable,
    # and have some data in knowledge_base/data/.

    # Example conversation
    inputs = {"messages": [HumanMessage(content="What is LangGraph?")]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("----")
            print(value)
        print("\n---\n")

    inputs = {"messages": [HumanMessage(content="Tell me about the capital of France.")]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("----")
            print(value)
        print("\n---\n")
