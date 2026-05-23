import streamlit as st
import pandas as pd
from google import genai
import os
from dotenv import load_dotenv
from db import save_message, get_history, init_db
from cache import get_cached_response, set_cached_response

load_dotenv()
init_db()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="AI Data Chat", page_icon="🤖")
st.title("🤖 Gemini 2.5 Flash Chat")

# --- NUEVA FUNCIÓN: CARGA DE DATOS ---
df_info = ""
uploaded_file = st.sidebar.file_uploader("Cargar CSV para análisis", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("Datos cargados:", df.shape)
    df_info = f"Los datos tienen estas estadísticas:\n{df.describe().to_string()}\n\n"

# Cargar historial de BD
history = get_history()
for role, message in history:
    with st.chat_message(role):
        st.markdown(message)

# Lógica del Chat
prompt = st.chat_input("Escribe algo...")

if prompt:
    # Combinar prompt con datos si existen
    final_prompt = df_info + prompt
    
    with st.chat_message("user"):
        st.markdown(prompt)
    save_message("user", prompt)

    # Lógica de Caché
    cached = get_cached_response(prompt) # Usamos el prompt original como llave

    if cached:
        response_text = cached
    else:
        with st.spinner("Pensando..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt
            )
            response_text = response.text
            set_cached_response(prompt, response_text) # Guardamos en caché

    with st.chat_message("assistant"):
        st.markdown(response_text)
    save_message("assistant", response_text)
