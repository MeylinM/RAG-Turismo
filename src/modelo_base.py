import os
import logging
from functools import lru_cache
from typing import List, Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rrf import buscar_texto_hibrido

load_dotenv()

logger = logging.getLogger("RAG_Model")

# Configuración

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db_multimodal"
COLLECTION_NAME = "documentos_multimodal"

MODELO_TEXTO_NAME = "intfloat/multilingual-e5-base"
MODELO_IMAGEN_NAME = "clip-ViT-B-32"

client_llm = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

# -------------------------
# Carga de modelos y colecciones
# -------------------------
@lru_cache(maxsize=1)
def get_models():
    logger.info("Cargando modelos de Embedding (Texto + Imagen)...")
    mod_txt = SentenceTransformer(MODELO_TEXTO_NAME)
    mod_img = SentenceTransformer(MODELO_IMAGEN_NAME)
    return mod_txt, mod_img

@lru_cache(maxsize=1)
def get_collections():
    logger.info(f"Conectando a ChromaDB en: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    logger.warning(f"Usando DB_DIR = {DB_DIR}")
    print("DB_DIR =", DB_DIR)

    col_txt = client.get_collection(name=f"{COLLECTION_NAME}_texto")
    col_img = client.get_collection(name=f"{COLLECTION_NAME}_imagen")
    return col_txt, col_img

# -------------------------
# Función principal de RAG
# -------------------------
def generar_respuesta(query: str, top_k: int = 3) -> Dict:
    model_txt, model_clip = get_models()
    col_txt, col_img = get_collections()
    
    # Búsqueda de texto
    res_txt = buscar_texto_hibrido(query, col_txt, model_txt, top_k=top_k)

    # Búsqueda de imágenes
    emb_img = model_clip.encode(query, normalize_embeddings=True).tolist()
    res_img = col_img.query(query_embeddings=[emb_img], n_results=2)

    # Procesamiento
    contexto_textual = []
    lista_fuentes = set()
    lista_imagenes_path = []

    if res_txt['ids'] and res_txt['documents']:
        for i, doc in enumerate(res_txt['documents'][0]):
            meta = res_txt['metadatas'][0][i]
            nombre_pdf = meta.get('origen_pdf', 'desconocido')
            lista_fuentes.add(nombre_pdf)
            contexto_textual.append(f"--- Fuente: {nombre_pdf} ---\n{doc}")

    if res_img['ids'] and res_img['metadatas']:
            PROYECTO_ROOT = Path(__file__).resolve().parent.parent 
            CARPETA_IMAGENES = PROYECTO_ROOT / "data" / "imagenes_extraidas"

            for i, meta in enumerate(res_img['metadatas'][0]):
                nombre_img = meta.get('imagen_nombre')
                
                # Construimos la ruta real ignorando la que viene mal de la DB
                full_path = CARPETA_IMAGENES / nombre_img

                if full_path.exists():
                    lista_imagenes_path.append(str(full_path))
                    contexto_textual.append(f"[SISTEMA: Se muestra la imagen {nombre_img}]")
                    logger.info(f"✅ Imagen encontrada: {full_path}")
                else:
                    logger.warning(f"❌ No existe en: {full_path}")
    # Generar respuesta LLM
    contexto_str = "\n\n".join(contexto_textual)
    system_prompt = """Eres un asistente turístico experto en Japón y España.
    Responde usando SOLO la información del contexto. Si se muestra una imagen, puedes mencionarla."""

    response = client_llm.chat.completions.create(
        model=os.getenv("MODELO_LLM", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{contexto_str}"}
        ],
        temperature=0.3
    )
    answer = response.choices[0].message.content

    return {
        "respuesta": answer,
        "fuentes": list(lista_fuentes),
        "imagenes": lista_imagenes_path,
        "contexto_usado": contexto_textual
    }
