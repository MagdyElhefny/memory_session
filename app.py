"""
Streamlit interface for the LangGraph healthy-food assistant.

Run: streamlit run app.py

Sidebar lets you switch "User" — try asking as 'alice', mentioning a food
you like, then switch to 'bob': bob won't know about it (long-term memory is
per-user). Switch back to 'alice' and start a NEW conversation (button in
sidebar): she still remembers, because long-term memory outlives the thread.
"""
import uuid
import streamlit as st
from langgraph.types import Command

from graph import graph
from memory import get_preferences
from util import extract_text

st.set_page_config(page_title="Healthy Food Buddy", page_icon="🥦")
st.title("🥦 Healthy Food Buddy")
st.caption("A LangGraph demo: messages · reducers · short/long-term memory · human-in-the-loop · tools")


def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_interrupt = None


with st.sidebar:
    st.header("👤 User")
    user_id = st.text_input("User id", value=st.session_state.get("user_id", "alice"))
    if user_id != st.session_state.get("user_id"):
        st.session_state.user_id = user_id
        new_thread()
    if "thread_id" not in st.session_state:
        new_thread()

    if st.button("🔄 New conversation (same user)"):
        new_thread()
    st.caption(f"thread_id: `{st.session_state.thread_id[:8]}…`")
    st.caption("↳ short-term memory (this chat) resets. Long-term memory below does not.")

    st.divider()
    st.subheader("🧠 Long-term memory")
    st.caption(f"What I remember about **{user_id}**, across all conversations:")
    prefs = get_preferences(user_id)
    if prefs:
        for p in prefs:
            icon = "✅" if p["sentiment"] == "likes" else "🚫"
            st.write(f"{icon} {p['sentiment']} **{p['food']}**")
    else:
        st.caption("_Nothing saved yet — tell me you like/dislike a food!_")

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def show_ai_reply(result: dict):
    reply = extract_text(result["messages"][-1].content)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)


# --- Human-in-the-loop approval UI ---
if st.session_state.get("pending_interrupt"):
    payload = st.session_state.pending_interrupt
    with st.chat_message("assistant"):
        st.warning(f"🤔 **Approval needed:** {payload['message']}")
        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve & remember", use_container_width=True)
        reject = col2.button("❌ Reject", use_container_width=True)

    if approve or reject:
        result = graph.invoke(Command(resume={"approved": approve}), config=config)
        st.session_state.pending_interrupt = None
        if "__interrupt__" in result:
            st.session_state.pending_interrupt = result["__interrupt__"][0].value
        else:
            show_ai_reply(result)
        st.rerun()

else:
    user_input = st.chat_input("Ask about healthy food, or tell me a food you like/dislike…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking…"):
            result = graph.invoke(
                {"messages": [("user", user_input)], "user_id": user_id, "search_results": []},
                config=config,
            )

        if "__interrupt__" in result:
            st.session_state.pending_interrupt = result["__interrupt__"][0].value
            st.rerun()
        else:
            show_ai_reply(result)