import streamlit as st
from U_bot_backend import chatbot
from langchain_core.messages import HumanMessage

st.title("U-BOT Chatbot")

CONFIG = {"configurable": {"thread_id": "user-1"}}

# ----------- SESSION STATE -----------
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# ----------- LOAD OLD MSGS ----------
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------- USER INPUT -------------
user_input = st.chat_input("Type your message...")

if user_input:

    # Store User Message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Send to Backend LangGraph
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG
    )

    ai_reply = response["messages"][-1].content

    # Store AI Reply
    st.session_state.message_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    with st.chat_message("assistant"):
        st.markdown(ai_reply)