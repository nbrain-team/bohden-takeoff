import streamlit as st
import asyncio
import os
from together import Together
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API key from environment variable
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

# Set page title
st.set_page_config(page_title="AI Construction Chatbot", page_icon="🤖")

st.title("🏗️ AI Construction Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Function to generate AI response
async def generate_response(user_input):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content

# Get user input
if user_input := st.chat_input("Type your message..."):
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot_response = loop.run_until_complete(generate_response(user_input))

    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
