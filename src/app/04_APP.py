import streamlit as st
import requests
from PIL import Image
import os
import base64

# --- DEFINICIÓN DE AVATAR ---
ROBOT_ICON = "./img/robot.png"
TURISTA_ICON = "./img/turista.png"

# Configuración de la página
st.set_page_config(page_title="RAG Multimodal Viajes", layout="wide")

# --- FUNCIÓN PARA CARGAR EL DISEÑO ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    with open('style.css', 'r') as f:
        style = f.read()
    # CAMBIO IMPORTANTE: Usamos .replace en vez de .format
    st.markdown(f'<style>{style.replace("{img_bg}", bin_str)}</style>', unsafe_allow_html=True)

# Cargamos el fondo
try:
    set_background('./img/background.png')
except Exception as e:
    st.warning(f"No se pudo cargar el fondo: {e}")

# --- CABECERA CON LOGO ---
# Definimos 3 columnas. La del medio (5) es más ancha para el logo.
# Las de los lados (1) actúan como márgenes para centrarlo.
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    try:
        # width=450 fija el tamaño en pixeles. Puedes subir o bajar este número.
        # Quitamos 'use_container_width=True' porque eso es lo que lo estiraba al máximo.
        st.image("./img/logo.png", width=450) 
    except:
        st.title("Cicerón: Tu compañero de viaje")

# Mensaje de bienvenida
st.info("¡Hola! Soy Cicerón, tu guía turístico. Cuéntame, ¿a dónde quieres ir?")

# --- FIN CONFIGURACIÓN DISEÑO ---

# URL de tu API (Asegúrate de ejecutar 03_API.py en otra terminal)
API_URL = "http://localhost:8000/chat"

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 1. MOSTRAR HISTORIAL (Con estilo de 1 foto)
# ==========================================
contenedor_chat = st.container(border=True)
for message in st.session_state.messages:
    avatar_img = ROBOT_ICON if message["role"] == "assistant" else None
    turista_img = TURISTA_ICON if message["role"] == "user" else None
    with st.chat_message(message["role"],avatar=avatar_img):
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
    with st.chat_message("user",avatar=TURISTA_ICON):
        st.markdown(prompt)

    # Llamar a la API
    with st.chat_message("assistant",avatar=ROBOT_ICON):
        with st.spinner("Consultando guías y buscando fotos..."):
            
            # --- NUEVA LÓGICA DE HISTORIAL ---
            # Preparamos el historial para enviar (limitado a los últimos 6 mensajes)
            historial_para_api = []
            
            # Iteramos sobre los mensajes previos. 
            # Nota: Excluimos el último (que es el prompt actual) para no duplicarlo, 
            # ya que se envía en el campo "query".
            mensajes_previos = st.session_state.messages[:-1] 
            
            for msg in mensajes_previos[-6:]:
                # Filtramos solo role y content, quitamos las imágenes para no romper el JSON
                historial_para_api.append({"role": msg["role"], "content": msg["content"]})

            # Creamos el payload con la memoria incluida
            payload = {
                "query": prompt, 
                "top_k": 3,
                "historial": historial_para_api  # <--- ENVIAMOS LO QUE SABEMOS
            }
            # ---------------------------------

            try:
                # Usamos el payload completo
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
                    
                    # Mostrar solo la MEJOR imagen
                    if images:
                        st.markdown("---")
                        st.subheader("📸 Imagen destacada:")
                        
                        # Tomamos la primera imagen (Top 1)
                        mejor_imagen = images[0]
                        
                        try:
                            img = Image.open(mejor_imagen)
                            st.image(img, caption="Mejor coincidencia visual", use_column_width=True)
                        except:
                            st.warning(f"Imagen no encontrada: {mejor_imagen}")
                    
                    # Guardar en historial de Streamlit
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "images": images # Guardamos la lista
                    })
                    
                else:
                    st.error(f"Error en API: {response.status_code}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")