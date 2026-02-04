"""
================================================================================
02_evaluacion_retrieval.py - Evaluación automática de chunks con LLM
================================================================================
"""

import os
import sys
import random
import json
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # Conecta con LLM compatible
import logging
from dotenv import load_dotenv
import time
from pathlib import Path

# Añadir la carpeta padre (src/) al path para encontrar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# ============================================================================ 
# CONFIGURACIÓN GENERAL
# ============================================================================ 

ENV_PATH = utils.project_root() / ".env"
load_dotenv(dotenv_path=ENV_PATH)
utils.setup_logging()
logger = logging.getLogger("eval_auto")

# Base de datos ChromaDB
DB_DIR = str(utils.project_root() / "chroma_db_multimodal")  # tu DB real
COLLECTION_NAME = "documentos_multimodal_texto"  # colección de chunks de texto
GOLDEN_SET_FILE = str(utils.project_root() / "data" / "golden_set_automatico.jsonl")

logger.info(f"DB_DIR: {DB_DIR}")
logger.info(f"COLLECTION_NAME: {COLLECTION_NAME}")
logger.info(f"GOLDEN_SET_FILE: {GOLDEN_SET_FILE}")

# LLM y embeddings
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
MODELO_LLM = os.getenv("MODELO_LLM", "deepseek-r1:8b")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "intfloat/multilingual-e5-base")

# ============================================================================ 
# FUNCIONES PARA GENERAR PREGUNTAS
# ============================================================================ 

def generar_pregunta_para_chunk(texto_chunk, metadata, client_llm):
    """Genera una pregunta basada en un chunk usando LLM."""

    contexto_extra = f"Categoría: {metadata.get('category', 'General')}"
    if 'subcategory' in metadata:
        contexto_extra += f", Subcategoría: {metadata['subcategory']}"

    system_content = f"""Eres un profesor experto creando preguntas de examen sobre {contexto_extra}.

INSTRUCCIONES:
1. Lee el texto y formula UNA pregunta clara y específica.
2. La pregunta debe ser NATURAL, como si una persona sin conocimientos del tema la hiciera.
3. NO uses frases como "según el texto", "del documento", "proporcionado".
4. Devuelve SOLO la pregunta, sin explicaciones.
"""

    user_content = f"""TEXTO:
{texto_chunk[:1500]}

PREGUNTA:"""

    try:
        response = client_llm.chat.completions.create(
            model=MODELO_LLM,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=150
        )

        pregunta = response.choices[0].message.content.strip().replace('"', '')
        logger.info(f"Pregunta generada: {pregunta[:80]}...")
        return pregunta

    except Exception as e:
        logger.error(f"Error generando pregunta: {e}")
        return None

def crear_golden_set_automatico(collection, client_llm, num_preguntas=5):
    """Crea un golden set automáticamente para evaluación."""
    logger.info(f"Generando Golden Set Automático ({num_preguntas} preguntas)...")
    
    all_data = collection.get()
    all_ids = all_data['ids']
    all_docs = all_data['documents']
    all_metas = all_data['metadatas']
    
    if len(all_ids) == 0:
        logger.error("La base de datos está vacía.")
        return []

    indices = random.sample(range(len(all_ids)), min(num_preguntas, len(all_ids)))
    
    golden_set = []
    
    for i, idx in enumerate(indices):
        chunk_id = all_ids[idx]
        texto = all_docs[idx]
        meta = all_metas[idx]
        
        pregunta = generar_pregunta_para_chunk(texto, meta, client_llm)
        if pregunta:
            entry = {
                "id": f"q_{i}",
                "query": pregunta,
                "relevant_ids": [chunk_id],
                "texto_original": texto[:100],
                "metadata": {
                    "source": meta.get('source', 'unknown'),
                    "category": meta.get('category', 'General'),
                    "subcategory": meta.get('subcategory', '')
                }
            }
            golden_set.append(entry)
            time.sleep(2)  # Pausa corta para no saturar el LLM

    # Guardar golden set
    with open(GOLDEN_SET_FILE, 'w', encoding='utf-8') as f:
        for entry in golden_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Golden Set guardado en: {GOLDEN_SET_FILE}")
    return golden_set

# ============================================================================ 
# FUNCIONES DE EVALUACIÓN
# ============================================================================ 

def evaluar_retrieval(collection, model_emb, golden_set, top_k=3):
    """Evalúa la recuperación de chunks (Hit Rate y MRR)."""
    logger.info(f"Iniciando Evaluación (Top-{top_k})...")

    aciertos = 0
    mrr_sum = 0

    for item in golden_set:
        pregunta = item['query']
        target_ids = item['relevant_ids']

        query_formateada = f"query: {pregunta}"

        query_emb = utils.generar_embeddings(model_emb, [query_formateada])
        results = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )
        recuperados_ids = results['ids'][0]

        acierto = any(tid in recuperados_ids for tid in target_ids)
        if acierto:
            aciertos += 1
            for rank, rid in enumerate(recuperados_ids):
                if rid in target_ids:
                    mrr_sum += 1.0 / (rank + 1)
                    break

    total = len(golden_set)
    hit_rate = aciertos / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    return hit_rate, mrr

# ============================================================================ 
# MAIN
# ============================================================================ 

def main():
    logger.info("="*80)
    logger.info(" EVALUACIÓN AUTOMÁTICA DE RAG (PDFs) ")
    logger.info("="*80)

    # Conectar a ChromaDB
    if not os.path.exists(DB_DIR):
        logger.error(f"No existe BD en {DB_DIR}")
        return
    client_db = chromadb.PersistentClient(path=DB_DIR)
    collection = client_db.get_collection(COLLECTION_NAME)

    # Conectar a LLM y embeddings
    client_llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    model_emb = SentenceTransformer(MODELO_EMBEDDINGS)

    # Cargar o generar Golden Set
    golden_set = []
    if os.path.exists(GOLDEN_SET_FILE):
        opcion = input("Existe un Golden Set anterior. ¿Usarlo (s) o generar nuevo (n)? [s/n]: ")
        if opcion.lower() == 's':
            with open(GOLDEN_SET_FILE, 'r', encoding='utf-8') as f:
                golden_set = [json.loads(line) for line in f]
            logger.info(f"Leídas {len(golden_set)} preguntas.")

    if not golden_set:
        num = int(input("¿Cuántas preguntas generar para la prueba? (Rec: 5-10): "))
        golden_set = crear_golden_set_automatico(collection, client_llm, num_preguntas=num)

    # Ejecutar evaluación
    if golden_set:
        hit_rate, mrr = evaluar_retrieval(collection, model_emb, golden_set, top_k=3)
        logger.info("\n" + "="*60)
        logger.info("RESULTADOS DEL EXAMEN")
        logger.info("="*60)
        logger.info(f"Preguntas Evaluadas : {len(golden_set)}")
        logger.info(f"Hit Rate @3        : {hit_rate*100:.1f}%")
        logger.info(f"MRR Score          : {mrr:.3f}")
        if hit_rate < 0.5:
            logger.info("CONSEJO: Hit Rate bajo. Revisar chunking o embeddings.")
        else:
            logger.info("RESULTADO: Sistema funciona bien")
        logger.info("="*60)

if __name__ == "__main__":
    main()
