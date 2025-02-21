import os
from together import Together
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

def get_chatbot_response(user_input):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content
