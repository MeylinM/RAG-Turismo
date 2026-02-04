# api.py
import time
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from modelo_base import generar_respuesta  # <-- tu lógica RAG separada

app = FastAPI(title="RAG Turismo Japón-España", version="1.0")

# ==========================
# MODELOS DE REQUEST / RESPONSE
# ==========================
class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

class ChatResponse(BaseModel):
    respuesta: str
    fuentes: List[str]      # Nombres de PDFs
    imagenes: List[str]     # Rutas a imágenes
    debug_info: Dict        # Información de depuración

# ==========================
# ENDPOINTS
# ==========================

@app.get("/health")
def health_check():
    """
    Endpoint para verificar que la API está viva.
    """
    return {"status": "ok", "version": "1.0 RAG Turismo"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal para preguntas de turismo.
    """
    t_start = time.time()
    res = generar_respuesta(request.query, top_k=request.top_k)

    return ChatResponse(
        respuesta=res["respuesta"],
        fuentes=res["fuentes"],
        imagenes=res["imagenes"],
        debug_info={"contexto_usado": res["contexto_usado"], "tiempo_segundos": time.time() - t_start}
    )

# ==========================
# EJECUCIÓN LOCAL
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
