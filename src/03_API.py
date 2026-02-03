import os
import sys
import logging
from typing import List, Optional, Dict
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_API")

# ===========================
# 0. CONFIGURACIÓN
# ===========================
load_dotenv()

app = FastAPI(title="RAG Multimodal API", version="3.0")

# RUTAS Y NOMBRES (Deben coincidir con 01_Ingesta.py)
# [cite: 2] La ingesta usó "../chroma_db_multimodal"
DB_DIR = "../chroma_db_multimodal" 
COLLECTION_NAME = "documentos_multimodal" # Prefijo base

# Modelos definidos en ingesta [cite: 2]
MODELO_TEXTO_NAME = "intfloat/multilingual-e5-base"
MODELO_IMAGEN_NAME = "clip-ViT-B-32"

client_llm = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

# ===========================
# 1. CARGA DE RECURSOS (SINGLETON)
# ===========================
@lru_cache(maxsize=1)
def get_models():
    logger.info("Cargando modelos de Embedding (Texto + Imagen)...")
    # Modelo para búsqueda de texto
    mod_txt = SentenceTransformer(MODELO_TEXTO_NAME)
    # Modelo para búsqueda de imagen (CLIP) - Necesario para convertir query texto -> espacio imagen
    mod_img = SentenceTransformer(MODELO_IMAGEN_NAME)
    return mod_txt, mod_img

@lru_cache(maxsize=1)
def get_collections():
    logger.info(f"Conectando a ChromaDB en: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Recuperamos las DOS colecciones creadas en la ingesta
    try:
        col_txt = client.get_collection(name=f"{COLLECTION_NAME}_texto")
        col_img = client.get_collection(name=f"{COLLECTION_NAME}_imagen")
    except Exception as e:
        logger.error(f"Error accediendo a colecciones. ¿Ejecutaste la ingesta?: {e}")
        raise e
        
    return col_txt, col_img

# ===========================
# 2. LOGICA RAG
# ===========================

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

class ChatResponse(BaseModel):
    respuesta: str
    fuentes: List[str]      # Nombres de PDFs
    imagenes: List[str]     # Rutas a los archivos de imagen
    debug_info: Dict

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    query = request.query
    debug_info = {}
    
    # 1. Cargar recursos
    model_txt, model_clip = get_models()
    col_txt, col_img = get_collections()
    
    # 2. Búsqueda de TEXTO (Retrieval)
    # E5 necesita el prefijo "query: " para funcionar bien
    emb_txt = model_txt.encode(f"query: {query}", normalize_embeddings=True).tolist()
    
    res_txt = col_txt.query(
        query_embeddings=[emb_txt],
        n_results=request.top_k
    )
    
    # 3. Búsqueda de IMÁGENES (Multimodal Retrieval)
    # Usamos CLIP para proyectar el texto del usuario al espacio visual
    # Ejemplo: "Playa bonita" -> Vector CLIP -> Buscar fotos similares
    emb_img = model_clip.encode(query, normalize_embeddings=True).tolist()
    
    res_img = col_img.query(
        query_embeddings=[emb_img],
        n_results=2 # Traemos las 2 imágenes más relevantes
    )
    
    # 4. Procesar Resultados
    contexto_textual = []
    lista_fuentes = set()
    lista_imagenes_path = []
    
    # Procesar Texto recuperado
    if res_txt['ids'] and res_txt['documents']:
        for i, doc in enumerate(res_txt['documents'][0]):
            meta = res_txt['metadatas'][0][i]
            # Usamos 'origen_pdf' como se definió en la ingesta [cite: 2]
            nombre_pdf = meta.get('origen_pdf', 'desconocido')
            lista_fuentes.add(nombre_pdf)
            
            contexto_textual.append(
                f"--- Fuente: {nombre_pdf} ---\n{doc}"
            )

    # Procesar Imágenes recuperadas
    if res_img['ids'] and res_img['metadatas']:
        for i, meta in enumerate(res_img['metadatas'][0]):
            # Recuperamos el path absoluto guardado en la ingesta [cite: 2]
            path_img = meta.get('imagen_path')
            nombre_img = meta.get('imagen_nombre')
            
            if path_img and os.path.exists(path_img):
                lista_imagenes_path.append(path_img)
                # Opcional: Avisar al LLM que hay una imagen disponible
                contexto_textual.append(f"[SISTEMA: Se muestra al usuario la imagen {nombre_img} relacionada con el tema]")
    
    # 5. Generación de Respuesta (LLM)
    contexto_str = "\n\n".join(contexto_textual)
    
    system_prompt = """Eres un asistente turístico experto en Japón y España.
    Responde a la pregunta usando SOLO la información del contexto proporcionado.
    Si el contexto menciona que se muestra una imagen, puedes referenciarla como 'Como puedes ver en la imagen...'."""
    
    try:
        response = client_llm.chat.completions.create(
            model=os.getenv("MODELO_LLM", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{contexto_str}"}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error generando respuesta: {e}"
    
    debug_info["contexto_usado"] = contexto_textual
    
    return ChatResponse(
        respuesta=answer,
        fuentes=list(lista_fuentes),
        imagenes=lista_imagenes_path,
        debug_info=debug_info
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)