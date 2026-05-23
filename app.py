import streamlit as st
from google import genai
import os

from dotenv import load_dotenv

from db import save_message, get_history, init_db
from cache import get_cached_response, set_cached_response

load_dotenv()
init_db()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

st.set_page_config(
    page_title="AI Chat App",
    page_icon="🤖"
)

st.title("🤖 Gemini 2.5 Flash Chat")

history = get_history()

for role, message in history:

    with st.chat_message(role):

        st.markdown(message)

prompt = st.chat_input("Escribe algo...")

if prompt:

    with st.chat_message("user"):

        st.markdown(prompt)

    save_message("user", prompt)

    cached = get_cached_response(prompt)

    if cached:

        response_text = cached

    else:

        with st.spinner("Pensando..."):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            response_text = response.text

            set_cached_response(prompt, response_text)

    with st.chat_message("assistant"):

        st.markdown(response_text)

    save_message("assistant", response_text)
