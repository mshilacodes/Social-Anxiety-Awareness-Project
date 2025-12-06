import streamlit as st
import google.generativeai as genai
from functions import get_secret

api_key = get_secret("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash-lite")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history.append(("assistant", "Hi! How may I help you?"))

user_message = st.chat_input("Type your message...")

for role,message in st.session_state.chat_history:
    st.chat_message(role).write(message)

if user_message:
    st.chat_message("user").write(user_message)
    st.session_state.chat_history.append(("user", user_message))

    system_prompt = f""" 
    You are are  friendly and a mental health nurse
    Always suggest activities related to the problem they have shared.
    If the user asks concerning questions, suggest nearest hospital
    But if symptons are mild suggest activites user can do
    """
full_input = f"{system_prompt}\n\nUser message:\n\"\"\"{user_message}\"\"\"

context = [
    *[
        {"role":role, "parts":[{"text":msg}]} for role, msg in st.session_state.chat_history
    ],
    {"role": "user","parts":[{"text": full_input}]}
]

response = model.genrate_content(context)
assistant_reply = response.text

st.chat_message("assistant").write(assistant_reply)
st.session_state.chat_history.append(("assistant", assistant_reply))


