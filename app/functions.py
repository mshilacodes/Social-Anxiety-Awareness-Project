import streamlit as st
import os
from dotenv import load_dotenv

def get_secret(key):

    try:
        return st.sectrets[key]
    except Exception:
        load_dotenv()
        return os.genenv(key)