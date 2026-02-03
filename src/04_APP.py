import streamlit as st
import requests
from PIL import Image
import os

# Configuración de la página
st.set_page_config(page_title="RAG Multimodal Viajes", layout="wide")

st.title("✈️ Asistente de Viajes (Multimodal)")
st.markdown("Pregunta sobre Japón o España. El sistema buscará en PDFs y te mostrará fotos relevantes.")

# URL de tu API (Asegúrate de ejecutar 03_API.py en otra terminal)
API_URL = "http://localhost:8000/chat"

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si el mensaje tiene imágenes asociadas, mostrarlas
        if "images" in message and message["images"]:
            cols = st.columns(len(message["images"]))
            for idx, img_path in enumerate(message["images"]):
                try:
                    # Ajuste de ruta si es necesario (depende de desde dónde ejecutes streamlit)
                    # Si la API devuelve rutas relativas, aquí podrías necesitar ajustarlas
                    image = Image.open(img_path)
                    cols[idx].image(image, caption=os.path.basename(img_path), use_container_width=True)
                except Exception as e:
                    cols[idx].error(f"No se pudo cargar imagen: {img_path}")

# Input del usuario
if prompt := st.chat_input("Ej: ¿Qué ver en Kioto en primavera?"):
    # 1. Guardar y mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Llamar a la API
    with st.chat_message("assistant"):
        with st.spinner("Consultando guías y buscando fotos..."):
            try:
                payload = {"query": prompt, "top_k": 3}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["respuesta"]
                    sources = data["fuentes"]
                    images = data["imagenes"]
                    
                    # Mostrar respuesta texto
                    st.markdown(answer)
                    
                    # Mostrar fuentes
                    if sources:
                        st.caption(f"📚 Fuentes: {', '.join(sources)}")
                    
                    # Mostrar imágenes recuperadas
                    if images:
                        st.markdown("---")
                        st.subheader("📸 Imágenes relacionadas encontradas:")
                        cols = st.columns(len(images))
                        for idx, img_path in enumerate(images):
                            try:
                                img = Image.open(img_path)
                                cols[idx].image(img, caption="Recuperado por CLIP", use_container_width=True)
                            except:
                                cols[idx].warning(f"Imagen no encontrada: {img_path}")
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "images": images # Guardamos rutas para repintar luego
                    })
                    
                else:
                    st.error(f"Error en API: {response.status_code}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")