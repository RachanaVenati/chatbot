import os
import time
import requests
import base64
import streamlit as st
from dotenv import load_dotenv
import tiktoken

# Load .env variables
load_dotenv()
IONOS_API_TOKEN = os.getenv("IONOSKEY")
COLLECTION_ID = "ca216380-01e4-487b-9217-006de4399736"
MODEL_ID = "0b6c4a15-bb8d-4092-82b0-f357b77c59fd"

# Token counting
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Truncate context
def truncate_context_to_fit_tokens(text, max_tokens=3000, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    truncated = tokens[:max_tokens]
    return enc.decode(truncated)

# Get recent conversation from Streamlit session
def get_recent_conversation():
    history = st.session_state.chat_history[-10:]  # last 5 turns
    return "\n".join([
        f"{'User' if role == 'user' else 'Assistant'}: {msg}"
        for role, msg in history if msg.strip()
    ])

# LLaMA API call
def llama_completion(prompt):
    endpoint = f"https://inference.de-txl.ionos.com/models/{MODEL_ID}/predictions"
    headers = {
        "Authorization": f"Bearer {IONOS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    body = {"properties": {"input": prompt}}
    response = requests.post(endpoint, json=body, headers=headers).json()
    return response["properties"]["output"].strip()

# Retrieve context documents
def retrieve_context(query, limit=3):
    endpoint = f"https://inference.de-txl.ionos.com/collections/{COLLECTION_ID}/query"
    headers = {
        "Authorization": f"Bearer {IONOS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    body = {"query": query, "limit": limit}
    relevant_documents = requests.post(endpoint, json=body, headers=headers)
    return [
        base64.b64decode(entry['document']['properties']['content']).decode()
        for entry in relevant_documents.json()['properties']['matches']
    ]

# Generate response

def generate_response(question, context):
    truncated_context = truncate_context_to_fit_tokens(context)
    history = get_recent_conversation()
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the user's question truthfully and in one sentence.\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Context:\n{truncated_context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return llama_completion(prompt)

# Fallback response if context fails
def fallback_general_response(question):
    history = get_recent_conversation()
    prompt = (
        f"You are an assistant that helps users based on general knowledge and prior conversation.\n"
        f"Never hallucinate. Respond with 'I don't know' if unsure.\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"User's question: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot (LLaMA)", layout="centered")
st.title("\U0001F9E0 RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for role, message in st.session_state.chat_history:
    avatar = "\U0001F9D1‍\U0001F4BB" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
                st.markdown(f"""
            <div style='background-color:#ffffff;padding:1rem;border-radius:8px;
                        border:1px solid #ccc; font-family:"Segoe UI", sans-serif;
                        font-size:16px; color:#000000;'>
                {message}
            </div>
        """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask your question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user", avatar="\U0001F9D1‍\U0001F4BB"):
        st.markdown(
                        f"""
            <div style='background-color:#ffffff;padding:1rem;border-radius:8px;
                        border:1px solid #ccc; font-family:"Segoe UI", sans-serif;
                        font-size:16px; color:#000000;'>
                {user_input}
            </div>
        """
, unsafe_allow_html=True
        )

    st.session_state.chat_history.append(("bot", ""))
    with st.chat_message("assistant", avatar="🤖"):
        msg_placeholder = st.empty()
        with st.spinner("Retrieving context and generating response..."):
            docs = retrieve_context(user_input)
            if docs:
                final_context = "\n".join(docs)
                full_answer = generate_response(user_input, final_context)
            else:
                full_answer = fallback_general_response(user_input)

        # Typing effect
        typed_text = ""
        for char in full_answer:
            typed_text += char
            msg_placeholder.markdown(
                            f"""
            <div style='background-color:#ffffff;padding:1rem;border-radius:8px;
                        border:1px solid #ccc; font-family:"Segoe UI", sans-serif;
                        font-size:16px; color:#000000;'>
                {typed_text}
            </div>
        """, unsafe_allow_html=True
            )
            time.sleep(0.015)

        msg_placeholder.markdown(
                        f"""
            <div style='background-color:#ffffff;padding:1rem;border-radius:8px;
                        border:1px solid #ccc; font-family:"Segoe UI", sans-serif;
                        font-size:16px; color:#000000;'>
                {typed_text}
            </div>
        """, unsafe_allow_html=True)

        st.session_state.chat_history[-1] = ("bot", full_answer)
