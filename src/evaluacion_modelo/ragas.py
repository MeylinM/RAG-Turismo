"""
================================================================================
05_evaluacion_ragas_completa.py
================================================================================
OBJETIVO:
    Evaluar tu RAG COMPLETO (Pipeline Avanzado: Rewriting -> Routing -> RRF -> Rerank).
    
    Usa el concepto de "LLM-as-a-Judge" para puntuar:
    1. Fidelidad (¿Alucinó?)
    2. Relevancia (¿Respondió la pregunta?)
    3. Exactitud (¿Coincide con el Ground Truth?)
    4. Multimodalidad (¿Recuperó imágenes?)
"""

import json
import os
import sys
import logging
import time
import re
import csv
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/evaluacion_modelo
src_dir = os.path.dirname(current_dir)                   # .../src
project_root = os.path.dirname(src_dir)                  # .../RAG-Turismo
data_dir = os.path.join(project_root, "data")            # .../RAG-Turismo/data

sys.path.append(src_dir)
import utils
from modelo_base import generar_respuesta

utils.setup_logging()
logger = logging.getLogger("Evaluador_Completo")

# Configuración
ENV_PATH = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=ENV_PATH)

GOLDEN_SET_PATH = os.path.join(data_dir, "golden_set_turismo.jsonl")
RESULTADOS_CSV = os.path.join(current_dir, "reporte_evaluacion_final3.csv")

# Configuración del Juez (Usamos OpenAI o el que tengas en .env para juzgar)
client_juez = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"), 
    api_key=os.getenv("LLM_API_KEY")
)
MODELO_JUEZ = os.getenv("MODELO_LLM", "gpt-4o-mini") # O el modelo que uses

# ============================================================================
# FUNCIONES DEL JUEZ (Prompts de Evaluación)
# ============================================================================

def evaluar_fidelidad(respuesta, contexto, client):
    """ Verifica si la respuesta se basa SOLO en el contexto (Detecta alucinaciones) """
    prompt = f"""
    Eres un auditor de IA. Evalúa si la siguiente RESPUESTA está basada EXCLUSIVAMENTE en el CONTEXTO proporcionado.
    
    CONTEXTO:
    {contexto}
    
    RESPUESTA:
    {respuesta}
    
    Si la respuesta contiene información que NO está en el contexto, es una alucinación (0).
    Si toda la información está respaldada por el contexto, es correcto (1).
    
    Responde SOLO con un número: 0 o 1.
    """
    try:
        res = client.chat.completions.create(
            model=MODELO_JUEZ, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        val = res.choices[0].message.content.strip()
        return 1 if "1" in val else 0
    except: return 0

def evaluar_exactitud(respuesta_ia, ground_truth, client):
    """ Compara la respuesta de la IA con la respuesta ideal humana """
    prompt = f"""
    Compara la RESPUESTA DE LA IA con la RESPUESTA CORRECTA (Ground Truth).
    
    RESPUESTA CORRECTA: {ground_truth}
    RESPUESTA IA: {respuesta_ia}
    
    Evalúa del 1 al 5 qué tan bien coinciden semánticamente.
    1: Totalmente incorrecto.
    5: Dice exactamente lo mismo (aunque use otras palabras).
    
    Responde SOLO con el número (1-5).
    """
    try:
        res = client.chat.completions.create(
            model=MODELO_JUEZ, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        val = re.search(r'\d', res.choices[0].message.content)
        return int(val.group()) if val else 1
    except: return 1

def evaluar_multimodalidad(imagenes_recuperadas):
    """ Evalúa si el sistema trajo imágenes (Requisito del reto) """
    # Si la lista de imágenes no está vacía, es un éxito multimodal
    return 1 if len(imagenes_recuperadas) > 0 else 0

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print(" EVALUACIÓN DEL SISTEMA RAG COMPLETO (Juez IA)")
    print("="*60)
    
    # 1. Cargar Golden Set
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"No encuentro {GOLDEN_SET_PATH}. Ejecuta primero el creador de ground truth.")
        return
        
    casos = []
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            casos.append(json.loads(line))
            
    print(f" Casos a evaluar: {len(casos)}")
    
    resultados = []
    metricas_totales = {"fidelidad": 0, "exactitud": 0, "multimodal": 0, "tiempo": 0}
    
    # 2. Iterar y Evaluar
    for caso in tqdm(casos, desc="Evaluando"):
        query = caso['query']
        truth = caso['ground_truth']
        
        # --- A) EJECUTAR TU RAG REAL (Pipeline completo) ---
        start_t = time.time()
        try:
            # Llamamos a modelo_base.py
            resultado_rag = generar_respuesta(query, top_k=3)
            
            respuesta_ia = resultado_rag["respuesta"]
            contexto_usado = resultado_rag["contexto_usado"]
            imagenes = resultado_rag["imagenes"]
            
        except Exception as e:
            logger.error(f"Error generando respuesta para '{query}': {e}")
            respuesta_ia = "ERROR"
            contexto_usado = ""
            imagenes = []
        
        tiempo = time.time() - start_t
        
        # --- B) EL JUEZ EVALÚA ---
        score_fid = evaluar_fidelidad(respuesta_ia, contexto_usado, client_juez)
        score_acc = evaluar_exactitud(respuesta_ia, truth, client_juez)
        score_img = evaluar_multimodalidad(imagenes)
        
        # Acumular
        metricas_totales["fidelidad"] += score_fid
        metricas_totales["exactitud"] += score_acc
        metricas_totales["multimodal"] += score_img
        metricas_totales["tiempo"] += tiempo
        
        # Guardar detalle
        resultados.append({
            "pregunta": query,
            "respuesta_ia": respuesta_ia,
            "ground_truth": truth,
            "score_fidelidad": score_fid,
            "score_exactitud": score_acc,
            "tiene_imagenes": "SI" if score_img else "NO",
            "fuentes": str(resultado_rag.get("fuentes", [])),
            "tiempo_seg": round(tiempo, 2)
        })

    # 3. Calcular Promedios
    n = len(casos)
    if n > 0:
        prom_fid = (metricas_totales["fidelidad"] / n) * 100
        prom_acc = metricas_totales["exactitud"] / n
        prom_img = (metricas_totales["multimodal"] / n) * 100
        prom_time = metricas_totales["tiempo"] / n
    else:
        prom_fid = prom_acc = prom_img = prom_time = 0

    # 4. Reporte Final en Consola
    print("\n" + "="*60)
    print(" RESULTADOS FINALES DE LA EVALUACIÓN")
    print("="*60)
    print(f" Total casos: {n}")
    print(f" Exactitud (vs Ground Truth): {prom_acc:.2f} / 5.0")
    print(f" Fidelidad (No Alucinación):  {prom_fid:.1f}%")
    print(f" Tasa Multimodal (Imágenes):  {prom_img:.1f}%")
    print(f" Latencia promedio:           {prom_time:.2f} s")
    print("="*60)

    # 5. Guardar CSV (Para la Rúbrica)
    with open(RESULTADOS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)
        
    print(f" Reporte detallado guardado en: {RESULTADOS_CSV}")

if __name__ == "__main__":
    main()