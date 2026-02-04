# src/busqueda_hibrida.py

import logging
from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import utils   # Importamos tu utils

logger = logging.getLogger("Hibrida")

def obtener_ranking_bm25(query: str, all_ids: List[str], all_docs: List[str], top_n: int) -> List[str]:
    """Usa BM25 para encontrar documentos por palabras exactas."""
    # 1. Limpiamos y tokenizamos el corpus usando tu función de utils
    tokenized_corpus = [utils.limpiar_texto_basico(doc).split() for doc in all_docs]
    
    # 2. Inicializamos BM25
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 3. Limpiamos y tokenizamos la query
    query_tokens = utils.limpiar_texto_basico(query).split()
    
    # 4. Obtenemos puntuaciones
    scores = bm25.get_scores(query_tokens)
    
    # 5. Ordenamos y devolvemos los IDs de los mejores
    top_indices = np.argsort(-scores)[:top_n]
    return [all_ids[i] for i in top_indices]

def fusionar_rrf(ranking_sem: List[str], ranking_bm25: List[str], k: int = 60) -> List[tuple]:
    """Combina dos rankings basándose en la posición (RRF)."""
    scores = {}
    
    # Ranking Semántico
    for pos, doc_id in enumerate(ranking_sem):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + pos + 1)
        
    # Ranking BM25
    for pos, doc_id in enumerate(ranking_bm25):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + pos + 1)
    
    # Ordenar por score RRF descendente
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def buscar_texto_hibrido(query: str, coleccion_txt, modelo_txt, top_k: int = 3):
    """Función principal que mezcla Semántica + BM25."""
    
    # --- PASO 1: Obtener todos los documentos de la colección ---
    # Necesario para que BM25 pueda calcular frecuencias de palabras
    data = coleccion_txt.get(include=["documents", "metadatas"])
    if not data['ids']:
        return {"ids": [], "documents": [], "metadatas": []}

    all_ids = data['ids']
    all_docs = data['documents']
    all_metas = data['metadatas']

    # --- PASO 2: Búsqueda Semántica (Similitud) ---
    # Usamos tu función de generar_embeddings
    q_emb = utils.generar_embeddings(modelo_txt, [f"query: {query}"])
    res_sem = coleccion_txt.query(query_embeddings=q_emb, n_results=10)
    ranking_sem = res_sem['ids'][0]

    # --- PASO 3: Búsqueda BM25 (Palabras Clave) ---
    ranking_bm25 = obtener_ranking_bm25(query, all_ids, all_docs, top_n=10)

    # --- PASO 4: Fusión RRF ---
    fusionados = fusionar_rrf(ranking_sem, ranking_bm25)[:top_k]
    
    # --- PASO 5: Reconstruir formato para que el modelo base lo entienda ---
    # Creamos un mapa de ID -> índice para recuperar texto y meta rápido
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
    
    ids_finales = []
    docs_finales = []
    metas_finales = []
    
    for doc_id, _ in fusionados:
        idx = id_to_idx[doc_id]
        ids_finales.append(doc_id)
        docs_finales.append(all_docs[idx])
        metas_finales.append(all_metas[idx])

    return {
        "ids": [ids_finales],
        "documents": [docs_finales],
        "metadatas": [metas_finales]
    }