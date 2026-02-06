# src/reranker.py
import os
import logging
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# Cargar variables
load_dotenv()
logger = logging.getLogger("Reranker")

class RerankingSystem:
    def __init__(self):
        # Cargamos el modelo definido en tu .env (o el default)
        model_name = os.getenv("MODELO_RERANKER", "BAAI/bge-reranker-v2-m3")
        logger.info(f" Cargando modelo Cross-Encoder: {model_name}...")
        
        try:
            # Usamos device='cpu' o 'cuda' según disponibilidad. 
            # CrossEncoders pueden ser pesados, cuidado con la memoria.
            self.model = CrossEncoder(model_name)
            logger.info(" Modelo Reranker cargado correctamente.")
        except Exception as e:
            logger.error(f" Error cargando Reranker: {e}")
            raise e

    def rerank(self, query: str, docs: list, top_k: int = 3):
        """
        Reordena una lista de documentos basada en su relevancia con la query.
        
        Args:
            query: La pregunta del usuario.
            docs: Lista de diccionarios o tuplas (texto, metadatos, id).
            top_k: Número de documentos a devolver después del reordenamiento.
        """
        if not docs:
            return []

        # Preparar pares para el modelo: [[query, doc1], [query, doc2], ...]
        # Asumimos que docs viene como una lista de tuplas/objetos donde 
        # el primer elemento es el texto del documento.
        
        # Extraemos solo el texto para el modelo
        textos = [d['texto'] for d in docs]
        sentence_pairs = [[query, text] for text in textos]

        # Predecir scores (scores más altos = mayor relevancia)
        scores = self.model.predict(sentence_pairs)

        # Combinar scores con los documentos originales
        resultados_con_score = []
        for i, doc in enumerate(docs):
            resultados_con_score.append({
                "documento": doc, # Guardamos el objeto original completo
                "score_rerank": float(scores[i])
            })

        # Ordenar descendente por score
        ranking_nuevo = sorted(resultados_con_score, key=lambda x: x['score_rerank'], reverse=True)

        # Loguear para depuración (puedes quitar esto en prod)
        logger.info(f" Top 1 Rerank Score: {ranking_nuevo[0]['score_rerank']:.4f}")
        
        # Devolver solo los top_k
        return ranking_nuevo[:top_k]