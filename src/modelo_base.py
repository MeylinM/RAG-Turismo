# src/modelo_base.py

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# --- TUS MÓDULOS AVANZADOS (Arquitectura Modular) ---
from rrf import obtener_ranking_bm25, fusionar_rrf
from reranker import RerankingSystem
from query_rewriting import QueryRewriter 
from semantic_router import SemanticRouter

# Configuración inicial
load_dotenv()
logger = logging.getLogger("RAG_Orquestador")

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db_multimodal"
COLLECTION_TEXTO = "documentos_multimodal_texto"
COLLECTION_IMAGEN = "documentos_multimodal_imagen"

# --- FUNCIONES DE SEGURIDAD (NUEVO) ---
def validar_seguridad(query: str) -> bool:
    """Detecta intentos básicos de Prompt Injection."""
    frases_prohibidas = [
        "ignora tus instrucciones", "olvida tus reglas", "ignore previous instructions",
        "actúa como", "tu nuevo rol es", "system override"
    ]
    query_lower = query.lower()
    for frase in frases_prohibidas:
        if frase in query_lower:
            return False
    return True

# --- INICIALIZACIÓN GLOBAL (Se ejecuta una vez al arrancar) ---
# 1. Base de Datos
client_chroma = chromadb.PersistentClient(path=str(DB_DIR))
col_texto = client_chroma.get_collection(COLLECTION_TEXTO)
col_imagen = client_chroma.get_collection(COLLECTION_IMAGEN)

# 2. Modelos de Embeddings (Ligeros)
model_emb_texto = SentenceTransformer("intfloat/multilingual-e5-base")
model_emb_imagen = SentenceTransformer("clip-ViT-B-32")

# 3. Cliente LLM
client_llm = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))

# 4. Instancias de tus Técnicas Avanzadas
rewriter = QueryRewriter()
router = SemanticRouter()
reranker = RerankingSystem()

def generar_respuesta(query_usuario: str, top_k: int = 3, historial: list = None):
    """
    Pipeline RAG Avanzado (SAA 100%):
    1. Seguridad -> 2. Rewriting -> 3. Routing -> 4. Hybrid Search (RRF) -> 5. Reranking -> 6. Generation
    """
    
    # ==========================================
    # PASO 0: SEGURIDAD ANTES DE NADA (NUEVO)
    # ==========================================
    if not validar_seguridad(query_usuario):
        return {
            "respuesta": "Lo siento, no puedo procesar esa solicitud por motivos de seguridad.",
            "fuentes": [],
            "imagenes": [],
            "contexto_usado": "Bloqueado por seguridad"
        }

    # Inicialización del historial si es None
    if historial is None:
        historial = []

    # ==========================================
    # PASO 1: QUERY REWRITING (Mejora la pregunta)
    # ==========================================
    query_para_rewriting = query_usuario

    # 2. Si hay historial, la actualizamos para incluir el contexto
    if historial:
        # Tomamos los últimos mensajes para dar contexto
        ultimos_mensajes = historial[-6:]
        texto_historial = " ".join([m["content"] for m in ultimos_mensajes])
        
        # Creamos una query compuesta: "Contexto previo... + Pregunta nueva"
        query_para_rewriting = f"Contexto de la conversación: {texto_historial}. Pregunta actual del usuario: {query_usuario}"
        
        logger.info(f"Memoria inyectada en búsqueda: {query_para_rewriting[:100]}...")

    query_optimizada = rewriter.reescribir(query_para_rewriting)
    # ==========================================
    # PASO 2: SEMANTIC ROUTING (Filtra por país)
    # ==========================================
    filtros_metadata = router.detectar_filtros(query_optimizada)
    
    # ==========================================
    # PASO 3: RETRIEVAL HÍBRIDO (RRF)
    # ==========================================
    
    # 3.1 Vector Search (Semántico) - APLICANDO FILTROS
    emb_query = model_emb_texto.encode(f"query: {query_optimizada}").tolist()
    
    # Pedimos 20 candidatos iniciales
    res_sem = col_texto.query(
        query_embeddings=[emb_query],
        n_results=20,
        where=filtros_metadata # <--- APLICAMOS ROUTING AQUÍ
    )
    ids_semanticos = res_sem['ids'][0] if res_sem['ids'] else []

    # 3.2 Keyword Search (BM25)
    # Nota: Traemos ids globales para BM25 y dejamos que RRF decida
    todos_docs = col_texto.get() 
    ids_bm25 = obtener_ranking_bm25(
        query_optimizada, 
        todos_docs['ids'], 
        todos_docs['documents'], 
        top_n=20
    )

    # 3.3 Fusión RRF (Cruce de listas)
    ranked_tuples = fusionar_rrf(ids_semanticos, ids_bm25, k=60)
    top_ids_fusionados = [x[0] for x in ranked_tuples[:15]] # Top 15 para reranking

    if not top_ids_fusionados:
        # Fallback si no encuentra nada
        return {
            "respuesta": "Lo siento, no he encontrado información relevante en mis guías.", 
            "fuentes": [], 
            "imagenes": [], 
            "contexto_usado": ""
        }

    # Recuperamos el contenido completo de los ganadores del RRF
    docs_candidatos = col_texto.get(ids=top_ids_fusionados, include=["documents", "metadatas"])

    # ==========================================
    # PASO 4: RE-RANKING (Cross-Encoder)
    # ==========================================
    lista_para_rerank = []
    for i in range(len(docs_candidatos['ids'])):
        # Protección extra por si falta metadata
        meta = docs_candidatos['metadatas'][i] if docs_candidatos['metadatas'][i] else {}
        
        lista_para_rerank.append({
            "texto": docs_candidatos['documents'][i],
            "metadata": meta,
            "id": docs_candidatos['ids'][i]
        })

    # El reranker selecciona los definitivos (top_k)
    mejores_docs = reranker.rerank(query_optimizada, lista_para_rerank, top_k=top_k)

    # ==========================================
    # PASO 5: RECUPERACIÓN MULTIMODAL (Imágenes)
    # ==========================================
    emb_img = model_emb_imagen.encode(query_usuario).tolist()
    res_img = col_imagen.query(query_embeddings=[emb_img], n_results=3)
    
    rutas_imagenes = []
    if res_img['ids'] and res_img['ids'][0]:
         for meta in res_img['metadatas'][0]:
             if meta and 'imagen_path' in meta:
                 # --- CORRECCIÓN DE RUTAS ---
                 nombre_archivo = os.path.basename(meta['imagen_path'])
                 ruta_absoluta = BASE_DIR / "data" / "imagenes_extraidas" / nombre_archivo
                 
                 if ruta_absoluta.exists():
                     rutas_imagenes.append(str(ruta_absoluta))
                 else:
                     logger.warning(f"Imagen no encontrada en disco: {ruta_absoluta}")

    # ==========================================
    # PASO 6: GENERACIÓN CON LLM (Con corrección de error)
    # ==========================================
    
    contexto_texto = "\n\n".join([
        f"--- Fuente: {d['documento']['metadata'].get('nombre_archivo', 'Doc Desconocido')} (Score: {d['score_rerank']:.2f}) ---\n{d['documento']['texto']}" 
        for d in mejores_docs
    ])

    prompt_sistema = """
Eres Cicerón, un guía turístico EXPERTO, amable y directo especializado en ESPAÑA y JAPÓN.
Tu objetivo es responder a las dudas de los turistas basándote EXCLUSIVAMENTE en el fragmento de texto proporcionado.

REGLAS DE ESTILO:
1. Habla de forma natural y fluida, como una persona, no como un robot.
2. PROHIBIDO usar frases como "Según el texto", "En el documento", "La información dice", "Se menciona que".
   - MAL: "Según el texto, el museo abre a las 10."
   - BIEN: "El museo abre a las 10."
3. Sé conciso. Ve al grano. No des rodeos.
4. Usa **negritas** para resaltar lugares, horarios y precios importantes.

REGLAS DE CONTENIDO:
1. Usa SOLO la información del contexto. NO inventes nada que no esté escrito ahí.
2. Si la respuesta NO está en el contexto, NO intentes adivinar, ni inferir, ni usar tu conocimiento general.
3. Si no tienes la información exacta, responde SOLAMENTE: "Lo siento, mis fuentes actuales no tienen esa información específica sobre [tema]."
4. Si el contexto menciona varias opciones, menciónalas todas claramente.
"""
    
    prompt_usuario = f"Pregunta viajero: {query_usuario}\n\nContexto:\n{contexto_texto}"

    # Construcción de mensajes con historial
    messages_finales = [{"role": "system", "content": prompt_sistema}]
    
    # Añadimos el historial previo (si existe)
    if historial:
        messages_finales.extend(historial)
        
    # Añadimos la pregunta actual con el contexto inyectado
    messages_finales.append({"role": "user", "content": prompt_usuario})

    response = client_llm.chat.completions.create(
        model=os.getenv("MODELO_LLM", "llama3-70b-8192"),
        messages=messages_finales,
        temperature=0.0
    )

    # ==========================================
    # RETORNO SEGURO
    # ==========================================
    lista_fuentes = list(set([
        d['documento']['metadata'].get('nombre_archivo', 'Desconocido') 
        for d in mejores_docs
    ]))

    return {
        "respuesta": response.choices[0].message.content,
        "fuentes": lista_fuentes,
        "imagenes": rutas_imagenes,
        "contexto_usado": contexto_texto
    }