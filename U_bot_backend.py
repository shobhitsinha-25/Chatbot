from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# ---------------- STATE ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- HF MODEL ----------------
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95,
    provider="hf-inference"   
)

model = ChatHuggingFace(llm=llm)

# ---------------- NODE ----------------
def chat_node(state: ChatState):

    messages = state['messages']
    response = model.invoke(messages)

    return {'messages': [response]}

# ---------------- GRAPH ----------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# ---------------- MEMORY ----------------
memory = MemorySaver()

chatbot = graph.compile(checkpointer=memory)