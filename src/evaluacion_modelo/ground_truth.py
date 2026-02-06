"""
================================================================================
04_crear_ground_truth_multimodal.py
================================================================================
OBJETIVO:
    Crear el "Golden Set" (preguntas y respuestas ideales) leyendo
    directamente de tu base de datos multimodal.
    
    ESTO ES NECESARIO PARA LA RÚBRICA (RA3 - Evaluación con Golden Set).
"""

import json
import os
import sys
import random
import logging
import chromadb
from dotenv import load_dotenv
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS (Update)
# ==========================================
# 1. Ubicación de este script: .../src/evaluacion_modelo
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ubicación de src: .../src
src_dir = os.path.dirname(current_dir)

# 3. Ubicación RAÍZ del proyecto: .../RAG-Turismo
project_root = os.path.dirname(src_dir)

# 4. Carpeta DATA: .../RAG-Turismo/data
data_dir = os.path.join(project_root, "data")

# Añadimos src al path para poder importar utils
sys.path.append(src_dir)
import utils

utils.setup_logging()  
logger = logging.getLogger("crear_ground_truth")

ENV_PATH = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# Guardamos el archivo directamente en la carpeta 'data'
ARCHIVO_SALIDA = os.path.join(data_dir, "golden_set_turismo.jsonl")

DB_DIR = str(utils.project_root() / "chroma_db_multimodal") # Tu DB nueva
COLLECTION_NAME = "documentos_multimodal_texto" # Tu colección de texto

def main():
    print("\n" + "="*70)
    print(" CREADOR DE GROUND TRUTH - TURISMO JAPÓN/ESPAÑA")
    print("="*70)
    
    # 1. Conectar a ChromaDB Multimodal
    if not os.path.exists(DB_DIR):
        print(f" ERROR: No existe la base de datos en {DB_DIR}")
        print(" Ejecuta primero 01_Ingesta_multimodal_metadata.py")
        return
    
    client_db = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client_db.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f" Error conectando a la colección '{COLLECTION_NAME}': {e}")
        return
    
    # 2. Obtener chunks
    print(" Cargando chunks...")
    all_data = collection.get(include=["documents", "metadatas"])
    total_chunks = len(all_data['ids'])
    print(f" Total de chunks disponibles: {total_chunks}")
    
    if total_chunks == 0:
        print(" La base de datos está vacía.")
        return

    # 3. Loop de creación
    indices_disponibles = list(range(total_chunks))
    random.shuffle(indices_disponibles)
    
    contador = 0
    idx_actual = 0
    
    while idx_actual < len(indices_disponibles):
        i = indices_disponibles[idx_actual]
        chunk_id = all_data['ids'][i]
        chunk_text = all_data['documents'][i]
        chunk_meta = all_data['metadatas'][i]
        
        print("\n" + "="*70)
        print(f" CHUNK (Fuente: {chunk_meta.get('nombre_archivo', '?')})")
        print(f" ID: {chunk_id}")
        print("-" * 70)
        print(f"{chunk_text[:500]}...") # Mostrar solo los primeros 500 chars para no saturar
        print("-" * 70)
        
        print("\n OPCIONES:")
        print("   [1] Crear pregunta (Buena para evaluar)")
        print("   [2] Saltar (No es relevante o es un trozo feo)")
        print("   [3] Salir y Guardar")
        
        opcion = input(" > ").strip()
        
        if opcion == '3':
            break
        elif opcion == '2':
            idx_actual += 1
            continue
        elif opcion == '1':
            print("\n Escribe la PREGUNTA que un turista haría sobre este texto:")
            pregunta = input(" PREGUNTA: ").strip()
            
            if not pregunta: continue
            
            print("\n Escribe la RESPUESTA IDEAL (Ground Truth):")
            respuesta = input(" RESPUESTA: ").strip()
            
            if not respuesta: continue
            
            entrada = {
                "query": pregunta,
                "ground_truth": respuesta,
                "relevant_ids": [chunk_id], # Guardamos qué chunk tiene la respuesta
                "metadata": chunk_meta
            }
            
            with open(ARCHIVO_SALIDA, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            
            contador += 1
            print(f" >> Guardado. Llevas {contador} preguntas.")
            idx_actual += 1
        else:
            print(" Opción no válida.")

    print(f"\n FINALIZADO. Archivo guardado en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()