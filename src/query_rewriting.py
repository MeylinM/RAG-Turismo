import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("QueryRewriter")

class QueryRewriter:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )
        self.model = os.getenv("MODELO_LLM", "gpt-3.5-turbo")

    def reescribir(self, query_original: str) -> str:
        """
        Transforma una consulta vaga en una optimizada para búsqueda vectorial.
        Ej: "sitios japo madrid" -> "restaurantes de comida japonesa en Madrid"
        """
        try:
            prompt = f"""Actúa como un experto en recuperación de información. 
            Tu tarea es reescribir la consulta del usuario para mejorar los resultados en una búsqueda semántica de turismo.
            Hazla específica, elimina ruido y enfoca las palabras clave.
            
            Consulta original: "{query_original}"
            
            Responde SOLO con la consulta reescrita, sin explicaciones ni comillas."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Creatividad baja para ser precisos
                max_tokens=100
            )
            
            query_nueva = response.choices[0].message.content.strip()
            logger.info(f"🔄 Rewriting: '{query_original}' -> '{query_nueva}'")
            return query_nueva
            
        except Exception as e:
            logger.error(f"⚠️ Fallo en rewriting, usando original. Error: {e}")
            return query_original