import streamlit as st
import requests
from PIL import Image
import os
import sys
import logging
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils

utils.setup_logging()
logger = logging.getLogger("RAG_Multimodal_APP")

# --- CONFIGURACIÓN DE CONSTANTES ---
ROBOT_ICON = "./img/robot.png"
TURISTA_ICON = "./img/turista.png"
LOGO2_PATH = "./img/logo-2.png"
API_BASE_URL = "http://localhost:8000"
API_CHAT_URL = f"{API_BASE_URL}/chat"

try:
    img_favicon = Image.open(LOGO2_PATH)
except Exception as e:
    logger.error(f"No se pudo cargar el favicon desde {LOGO2_PATH}: {e}")
    img_favicon = None 

st.set_page_config(
    page_title="Cicerón ¿Listo para viajar?", # <-- NUEVO TÍTULO
    page_icon=img_favicon,                    # <-- NUEVO ICONO
    layout="wide"
)

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

# Inicializar historial de chat SI NO EXISTE
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# BARRA LATERAL (PANEL DE CONTROL)
# ==========================================
with st.sidebar:
    st.title("PANEL DE CONTROL")
    st.markdown("---") # Línea separadora
    
    # Botón 1: Borrar Historial
    # Al hacer clic, vaciamos la lista y recargamos
    if st.button("Borrar Historial"):
        logger.info(f"SOLICITUD DE BORRADO. Contenido actual del historial ({len(st.session_state.messages)} mensajes): {st.session_state.messages}")
        st.session_state.messages = []
        logger.info("Historial borrado exitosamente. Estado actual: []")
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True) # Espacio extra

    # Indicador de estado 
    st.caption("Estado del sistema:")
    
    # Intentamos conectar con el endpoint /health (común en FastAPI)
    # Si tu API no tiene /health, puedes probar con API_BASE_URL solamente
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        
        if response.status_code == 200:
            st.success("Conectado")
        else:
            st.error(f"Error API: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Servidor fuera de servicio")
        st.caption("Asegúrate de que el servidor esté activo")
    except Exception as e:
        st.error("Error de conexión")

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

# ==========================================
# 1. MOSTRAR HISTORIAL (Con estilo de 1 foto)
# ==========================================

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
                st.image(image, caption="Imagen recuperada", width=800)
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
            
            # Preparamos el historial para enviar (limitado a los últimos 6 mensajes)
            historial_para_api = []
            # Iteramos sobre los mensajes previos. 
            # Nota: Excluimos el último (que es el prompt actual) para no duplicarlo, 
            # ya que se envía en el campo "query".
            mensajes_previos = st.session_state.messages[:-1] 
            
            for msg in mensajes_previos[-6:]:
                # Filtramos solo role y content, quitamos las imágenes para no romper el JSON
                historial_para_api.append({"role": msg["role"], "content": msg["content"]})
                logger.info(f"ENVIANDO CONSULTA: '{prompt}'")
                logger.info(f"CONTEXTO (Historial adjunto): {historial_para_api}")

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
                    logger.info(f"RESPUESTA RECIBIDA: {answer[:100]}...")

                    # Mostrar respuesta texto
                    st.markdown(answer)
                    
                    # Mostrar fuentes
                    if sources:
                        st.caption(f"Fuentes: {', '.join(sources)}")
                    
                    # Mostrar solo la MEJOR imagen
                    if images:
                        # Tomamos la primera imagen (Top 1)
                        mejor_imagen = images[0]
                        
                        try:
                            img = Image.open(mejor_imagen)
                            st.image(img, caption="Mejor coincidencia visual", width=800)
                        except:
                            st.warning(f"Imagen no encontrada: {mejor_imagen}")
                    
                    # Guardar en historial de Streamlit
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "images": images 
                    })
                    
                else:
                    msg_error = f"Error en API: {response.status_code}"
                    logger.error(msg_error)
                    st.error(msg_error)
            except Exception as e:
                msg_error = f"Error de conexión: {e}"
                logger.error(msg_error)
                st.error(msg_error)