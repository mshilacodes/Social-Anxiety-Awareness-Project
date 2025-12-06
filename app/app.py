import streamlit as st
import google.genai as genai
from google.genai.types import GenerateContentConfig
from functions import get_secret


st.set_page_config(page_title="Leph Anxiety Support", page_icon="💬")

api_key = get_secret("API_KEY")
client = genai.Client(api_key=api_key)

def gemini_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=GenerateContentConfig(
            max_output_tokens=500,
        ),
        contents=prompt
    )
    return response.text

 
ANXIETY_QUESTIONS = [
    "Over the last 2 weeks, how often have you felt nervous, anxious, or on edge?",
    "Over the last 2 weeks, how often were you unable to stop or control worrying?",
    "How often have you worried too much about different things?",
    "How often have you had trouble relaxing?",
    "How often have you felt so restless that it was hard to sit still?",
    "How often have you become easily annoyed or irritable?",
    "How often have you felt afraid, as if something awful might happen?"
]

OPTIONS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
SCORES = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}

 
if "mode" not in st.session_state:
    st.session_state.mode = "intro"   # intro → anxiety_test → chat
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


 
def restart():
    st.session_state.mode = "intro"
    st.session_state.question_index = 0
    st.session_state.answers = []
    st.session_state.chat_history = []



if st.session_state.mode == "intro":

    st.title("💬 Leph Anxiety Support")
    st.write("How are you feeling today?")

    feeling = st.slider(
        "Choose how you feel",
        1, 10, 5,
        help="1 = Very Low, 10 = Very High"
    )

    if st.button("Continue"):
        st.session_state.mode = "anxiety_test"



elif st.session_state.mode == "anxiety_test":

    q_index = st.session_state.question_index
    st.title("🧠 Anxiety Screening")

    st.write(f"**Question {q_index+1} of {len(ANXIETY_QUESTIONS)}**")
    st.write(ANXIETY_QUESTIONS[q_index])

    choice = st.radio("Select an answer:", OPTIONS)

    if st.button("Next"):
        st.session_state.answers.append(choice)
        st.session_state.question_index += 1

        if st.session_state.question_index >= len(ANXIETY_QUESTIONS):
            st.session_state.mode = "results"



elif st.session_state.mode == "results":

    total_score = sum(SCORES[a] for a in st.session_state.answers)

    st.title("📊 Your Anxiety Assessment Results")
    st.write(f"Your total score is **{total_score}**.")

    # Get Gemini summary
    summary_prompt = f"""
    A user completed the GAD-7 anxiety questionnaire.
    Total score: {total_score}.

    Provide a supportive, friendly, 1-paragraph explanation 
    of what their score might indicate, and gentle self-care suggestions.
    Avoid medical claims or diagnosing.
    """

    result_text = gemini_response(summary_prompt)
    st.write(result_text)

    if st.button("Start Chatting"):
        st.session_state.mode = "chat"


elif st.session_state.mode == "chat":

    st.title("💬 Chat with Leph")

    # Display chat history
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Leph:** {text}")

    user_input = st.text_input("Say something…")

    if st.button("Send"):
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))

            ai_reply = gemini_response(user_input)
            st.session_state.chat_history.append(("assistant", ai_reply))

    st.button("Restart", on_click=restart)
