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

# ==========================================
# 1. MOSTRAR HISTORIAL (Con estilo de 1 foto)
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Si el mensaje tiene imágenes guardadas
        if "images" in message and message["images"]:
            # Solo mostramos la primera para mantener coherencia
            img_path = message["images"][0]
            try:
                image = Image.open(img_path)
                st.image(image, caption="Imagen recuperada", use_column_width=True)
            except Exception as e:
                st.error(f"No se pudo cargar imagen del historial: {img_path}")

# ==========================================
# 2. INPUT USUARIO Y NUEVA RESPUESTA
# ==========================================
if prompt := st.chat_input("Ej: ¿Qué ver en Kioto en primavera?"):
    # Guardar y mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Llamar a la API
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
                    
                    # --- CAMBIO AQUÍ: Mostrar solo la MEJOR imagen ---
                    if images:
                        st.markdown("---")
                        st.subheader("📸 Imagen destacada:")
                        
                        # Tomamos la primera imagen (Top 1)
                        mejor_imagen = images[0]
                        
                        try:
                            img = Image.open(mejor_imagen)
                            # use_column_width=True hace que ocupe todo el ancho
                            st.image(img, caption="Mejor coincidencia visual", use_column_width=True)
                        except:
                            st.warning(f"Imagen no encontrada: {mejor_imagen}")
                    # -------------------------------------------------
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "images": images # Guardamos la lista (aunque luego mostremos solo la 1ª)
                    })
                    
                else:
                    st.error(f"Error en API: {response.status_code}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")