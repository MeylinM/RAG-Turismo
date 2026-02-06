import logging
from typing import Dict, Optional

logger = logging.getLogger("SemanticRouter")

class SemanticRouter:
    def __init__(self):
        # Definimos reglas basadas en tus metadatos (metadatos_pdfs.json)
        self.reglas_ubicacion = {
            "japon": ["japón", "japon", "tokio", "tokyo", "akihabara", "shibuya", "sushi", "ramen"],
            "espana": ["españa", "spain", "madrid", "barcelona", "tapas", "paella", "rural"]
        }

    def detectar_filtros(self, query: str) -> Optional[Dict]:
        """
        Analiza la query y devuelve un filtro de metadatos compatible con ChromaDB.
        """
        query_lower = query.lower()
        filtros = {}

        # 1. Detección de Ubicación (Japón vs España)
        es_japon = any(keyword in query_lower for keyword in self.reglas_ubicacion["japon"])
        es_espana = any(keyword in query_lower for keyword in self.reglas_ubicacion["espana"])

        if es_japon and not es_espana:
            # Filtro para documentos de Japón (ajusta los valores exactos a tus metadatos)
            filtros["ubicacion"] = {"$in": ["Tokio, Japón", "Japón"]}
            logger.info("📍 Router: Destino detectado -> JAPÓN")
            
        elif es_espana and not es_japon:
            # Filtro para documentos de España
            filtros["ubicacion"] = "España"
            logger.info("📍 Router: Destino detectado -> ESPAÑA")
        
        else:
            logger.info("📍 Router: Destino ambiguo o mixto -> Búsqueda global")

        # 2. (Opcional) Aquí podrías añadir detección de categorías (Gastronomía, Alojamiento...)
        # if "comer" in query_lower or "restaurante" in query_lower:
        #     filtros["categoria"] = ...

        return filtros if filtros else None