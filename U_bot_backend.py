from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from huggingface_hub import InferenceClient
import streamlit as st

# ---------------- STATE ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- HF TOKEN ----------------
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# ---------------- SMALL MODEL ----------------
client = InferenceClient(
    model="google/gemma-2b-it",
    token=HF_TOKEN
)

# ---------------- NODE ----------------
def chat_node(state: ChatState):

    messages = state['messages']

    prompt = ""

    for msg in messages:
        if msg.type == "human":
            prompt += f"User: {msg.content}\n"
        elif msg.type == "ai":
            prompt += f"Assistant: {msg.content}\n"

    prompt += "Assistant:"

    response = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.3
    )

    return {
        "messages": [AIMessage(content=response)]
    }

# ---------------- GRAPH ----------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# ---------------- MEMORY ----------------
memory = MemorySaver()

chatbot = graph.compile(checkpointer=memory)