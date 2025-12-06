import streamlit as st
from google import genai
from google.genai import Client
from google.genai.types import GenerateContentConfig
import os
from functions import get_secret


st.set_page_config(page_title = "Leph Anxiety Support")

api_key = os.getenv("API_KEY")
client = Client(api_key=api_key)

def gemini_response(prompt):
    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        config=GenerateContentConfig(
            max_output_tokens=500,
        ),
        contents=prompt
    )
    return response.text


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mode" not in st.session_state:
    st.session_state.mode = "intro"

if "anxiety_answers" not in st.session_state:
    st.session_state.anxiety_answers = []

if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

#Anxiety Questions
ANXIETY_QUESTIONS =[
"In the past week, how often have you felt nervous or on the edge? (0=never, 3=nearly everyday)",
"How often have you been unable to stop or control worrying? (0-3)",
"How often have you been restless? (0-3)",
"How often have you felt easily irritated or annoyed? (0-3)",
"How often have you felt afraid that something awful might happen? (0-3)"
]
#Auto Intro 

if len(st.session_state.chat_history) == 0:
    st.session_state.chat_history.append(("assistant",
        "Hi there \nHow are you feeling today on a scale of  **1 to 10**?\n"
        "(1= very low, 10 = feeling great)"
                                          
    ))
    st.session_state.mode ="anxiety_scale"


for role,message in st.session_state.chat_history:
    st.chat_message(role).write(message)

#User Input
user_message = st.chat_input("Type your message...")


if user_message:
    st.chat_message("user").write(user_message)
    st.session_state.chat_history.append(("user", user_message))

    if st.session_state.mode == "anxiety scale":
        try: 
            scale_value = int(user_message)
            if 1 <= scale_value <= 10:
                if scale_value <= 6:
                    st.session_state.chat_history.append(("assistant",
                        "Thank you for sharing"
                        "I'd like to ask a few short questions to understand how you're feeling.\n\n"
                        "Here's the first question: \n\n"
                        f"{ANXIETY_QUESTIONS[0]}"
                        
                     ))
                    st.session_state.mode = "chat"
                else:
                    st.session.chat_history.append(("assistant"
                    "I'm glad you're feeling ok today! "
                    "If you want to talk about anything or ask for support, I'm here"
                    
                    ))
                    st.session_state.mode = "chat" 
            else:
                st.session_state.chat_history.append(("assistant", "Please choose a number between **1 and 10**"))
        except:
            st.session_state.chat_history.append(("assistant", "Please enter a number **4**, **7**, or **10**."))
    
    elif st.session_state.mode == "anxiety_questions":
        try:
            score = int(user_message)
            if 0 <= score <= 3:
                st.session_state.anxiety_answers.append(score)
                st.session_state.current_question_index +=1

                if st.session_stat.current_question_index < len(ANXIETY_QUESTIONS):
                    next_q = ANXIETY_QUESTIONS[st.session_state.current_question_index]
                    st.session_state.chat_history.append(("assistant", next_q))
                else:
                    total = sum(st.session_state.anxiety_answers)

                    if total <= 4:
                        level = "Minimal anxiety"
                        suggestion = "Try light breathing exercises, short walks, or journaling."
                    elif total<=9:
                        level = "Mild anciety"
                        suggestion = "Try gounding techniques like 5-4-3-2-1, warm tea, or mindful stretching." 
                    elif total <=14:
                        level = "Moderate anxiety"
                        suggestion = "Consider longer mindfulness sessions, talking to a trusted friend, or structured routines."
                    else:
                        level = "Sever anxiety"
                        suggestion = (
                            "It may help you to talk to a mental health professional"
                            "if you feel unsafe"
                        )
                    st.session_state.chat_history.append(("assistant",
                        f"Your score **{total}**, which suggests **{level}**.\n\n"
                        f"Here are some suppotive, non-medical suggestions: \n- {suggestion}\n\n"
                        "You can now chat with me about anything you'd like"                                      
                    ))

                    st.session_state.mode="chat"
        except:
            st.session_state.chat_history.append(("assistant", "Please answer with a number from **0 to 3**."))
    elif st.session_state.mode =="chat":

        system_prompt = f""" 
        You are are  friendly and a mental health nurse
        Always suggest activities related to the problem they have shared.
        If the user asks concerning questions, suggest nearest hospital
        But if symptons are mild suggest activites user can do
        """
        
        full_input = f"{system_prompt}\n\nUser message: \n\"\"\"{user_message}\"\"\""

        context = [
            *[
                {"role":role, "parts":[{"text":msg}]} for role, msg in st.session_state.chat_history
            ],
            {"role": "user","parts":[{"text": full_input}]}
        ]


        response = client.models.genrate_content(
            model="gemini-2.0-flash",
            conents=full_input
        )

       


        assistant_reply = response.text

        st.chat_message("assistant").write(assistant_reply)
        st.session_state.chat_history.append(("assistant", assistant_reply))


