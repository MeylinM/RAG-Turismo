# ============================================================================
# 0. IMPORTS
# ============================================================================

import os
import logging
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import utils

# ============================================================================
# 1. CONFIGURACIÓN (Variables Globales)
# ============================================================================
load_dotenv()
utils.setup_logging()
logger = logging.getLogger("RAG_Multimodal")

# Carpetas
DATA_DIR = Path("../data/pdf")
IMAGENES_DIR = Path("../data/imagenes_extraidas")
DB_DIR = "../chroma_db_multimodal"
COLLECTION_NAME = "documentos_multimodal"

# ========== MODELOS (Late Fusion) ==========
# E5: Mejor modelo multilingüe para texto técnico en español
# Prefijo "query:" y "passage:" mejoran la calidad de búsqueda
MODELO_E5 = "intfloat/multilingual-e5-base"
# Probar
# MODELO_E5 = "intfloat/multilingual-e5-large"

# CLIP: Embedding visual directo para imágenes
MODELO_CLIP = "clip-ViT-B-32"


# Crear carpetas si no existen
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)
if not IMAGENES_DIR.exists():
    IMAGENES_DIR.mkdir()


# ============================================================================
# 2. INICIALIZACIÓN DE MODELOS
# ============================================================================
def cargar_modelos():
    """
    Carga DOS modelos especializados (Late Fusion):
    
    1. E5 Multilingüe: Para texto
       - Entrenado específicamente para búsqueda semántica
       - Excelente rendimiento en español y dominios técnicos
       - Requiere prefijos "query:" y "passage:" para mejor rendimiento
    
    2. CLIP: Para imágenes
       - Crea embeddings visuales directos
       - Puede buscar imágenes con texto en inglés
    """
    logger.info("Cargando modelos de IA (Late Fusion)...")
    
    # 1. Modelo E5 para TEXTO (excelente para español)
    logger.info(f"   Cargando E5 para texto: {MODELO_E5}")
    model_e5 = SentenceTransformer(MODELO_E5)
    logger.info(f"      Dimensión E5: {model_e5.get_sentence_embedding_dimension()}")
    
    # 2. Modelo CLIP para IMÁGENES
    logger.info(f"   Cargando CLIP para imágenes: {MODELO_CLIP}")
    model_clip = SentenceTransformer(MODELO_CLIP)
    logger.info(f"      Dimensión CLIP: {model_clip.get_sentence_embedding_dimension()}")
    
    return model_e5, model_clip

def iniciar_base_datos(reset=False):
    """
    Prepara ChromaDB con DOS colecciones (Late Fusion).
    
    - coleccion_texto: Vectores E5 (texto)
    - coleccion_imagen: Vectores CLIP (imágenes)
    """
    client = chromadb.PersistentClient(path=DB_DIR)
    
    if reset:
        # Borrar ambas colecciones
        for nombre in [f"{COLLECTION_NAME}_texto", f"{COLLECTION_NAME}_imagen"]:
            try:
                client.delete_collection(nombre)
                logger.info(f"Colección {nombre} borrada.")
            except:
                pass
    
    # Crear/obtener colecciones separadas
    col_texto = client.get_or_create_collection(f"{COLLECTION_NAME}_texto")
    col_imagen = client.get_or_create_collection(f"{COLLECTION_NAME}_imagen")
    
    logger.info(f"Colecciones: {COLLECTION_NAME}_texto y {COLLECTION_NAME}_imagen")
    
    return col_texto, col_imagen

# ============================================================================
# 3. FUNCIONES AUXILIARES (Extracción)
# ============================================================================
def extraer_info_pdf(ruta_pdf):
    """
    Abre el PDF y saca TODO: Texto crudo + Imágenes guardadas en disco.
    """
    doc = fitz.open(ruta_pdf)
    texto_completo = ""
    lista_imagenes = []
    nombre_base = os.path.basename(ruta_pdf).replace('.pdf', '')
    
    # Recorrer páginas
    for num_pag, pagina in enumerate(doc, 1):
        # A. Texto
        texto_completo += f"\n--- PÁGINA {num_pag} ---\n" + pagina.get_text()
        
        # B. Imágenes
        for i, img in enumerate(pagina.get_images(full=True)):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                bytes_img = base["image"]
                ext = base["ext"]
                
                nombre_archivo = f"{nombre_base}_pag{num_pag}_img{i}.{ext}"
                ruta_completa = IMAGENES_DIR / nombre_archivo
                
                # Guardar en disco
                with open(ruta_completa, "wb") as f:
                    f.write(bytes_img)
                
                # Guardar info para procesar luego
                lista_imagenes.append({
                    "ruta": str(ruta_completa),
                    "pagina": num_pag,
                    "indice": i,
                    "nombre": nombre_archivo
                })
            except Exception as e:
                pass # Ignorar errores puntuales de extracción

    logger.info(f"PDF Leído: {len(texto_completo)} caracteres y {len(lista_imagenes)} imágenes.")
    return texto_completo, lista_imagenes

# ============================================================================
# 4. CARRIL DE TEXTO (usa E5 Multilingüe)
# ============================================================================
def procesar_carril_texto(texto, nombre_pdf, model_e5, col_texto):
    """
    Procesa texto con E5 Multilingüe.
    
    E5 es uno de los mejores modelos de embeddings para:
    - Texto en español
    - Dominios técnicos (agrario, médico, legal)
    - Búsqueda semántica asimétrica (query corta → documento largo)
    
    IMPORTANTE: E5 requiere prefijo "passage:" para documentos.
    En búsqueda usaremos "query:" para las preguntas.
    """
    if not texto.strip():
        return 0
        
    # Limpieza y chunking
    texto_limpio = utils.limpiar_para_embeddings_pdf(texto)
    chunks = utils.hacer_chunking(texto_limpio, chunk_size=500, overlap=50)
    
    count = 0
    ids = []
    embs = []
    docs = []
    metas = []
    
    for i, chunk in enumerate(chunks):
        # E5 requiere prefijo "passage:" para documentos
        chunk_con_prefijo = f"passage: {chunk}"
        vector = model_e5.encode(chunk_con_prefijo, normalize_embeddings=True).tolist()
        
        doc_id = f"{nombre_pdf}_txt_{i}"
        
        ids.append(doc_id)
        embs.append(vector)
        docs.append(chunk)  # Guardamos SIN prefijo para mostrar
        metas.append({
            "source": nombre_pdf,
            "tipo": "texto",
            "pagina": 0
        })
        count += 1
    
    if count > 0:
        col_texto.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    
    return count

# ============================================================================
# 5. CARRIL DE IMAGEN (usa CLIP - embedding visual directo)
# ============================================================================
def procesar_carril_imagen(lista_imagenes, nombre_pdf, model_clip, col_imagen):
    """
    Procesa imágenes con CLIP: embedding visual DIRECTO.
    
    CLIP crea un vector directamente de los píxeles, preservando
    toda la información visual sin depender de descripciones textuales.
    
    Para búsqueda: usaremos CLIP también para codificar la query,
    permitiendo buscar "gráfico de cultivos" y encontrar imágenes relevantes.
    """
    if not lista_imagenes:
        return 0
    
    count = 0
    ids = []
    embs = []
    docs = []
    metas = []
    
    logger.info(f"Generando embeddings CLIP para {len(lista_imagenes)} imágenes...")
    
    for img_data in tqdm(lista_imagenes):
        try:
            # Cargar imagen
            imagen_pil = Image.open(img_data["ruta"]).convert('RGB')
            
            # CLIP: Embedding visual directo
            vector = model_clip.encode(imagen_pil, normalize_embeddings=True).tolist()
            
            # Metadata descriptiva para UI
            descripcion_metadata = f"[IMAGEN] Página {img_data['pagina']}, archivo: {img_data['nombre']}"
            
            ids.append(f"{nombre_pdf}_img_{img_data['pagina']}_{img_data['indice']}")
            embs.append(vector)
            docs.append(descripcion_metadata)
            metas.append({
                "source": nombre_pdf,
                "tipo": "imagen",
                "pagina": img_data["pagina"],
                "imagen_path": img_data["ruta"],
                "imagen_nombre": img_data["nombre"]
            })
            count += 1
            
        except Exception as e:
            logger.error(f"Error procesando imagen {img_data['nombre']}: {e}")
            
    if count > 0:
        col_imagen.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        
    return count

# ============================================================================
# 6. PROGRAMA PRINCIPAL
# ============================================================================
def main():
    print("\n" + "="*60)
    print(" RAG MULTIMODAL - LATE FUSION (E5 + CLIP)")
    print("="*60)
    
    # 1. Buscar PDFs
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Error: No hay PDFs en {DATA_DIR}")
        return
    
    print(f"\nPDFs encontrados: {len(pdfs)}")
    for pdf in pdfs:
        print(f"   - {pdf.name}")
    
    # 2. Preguntar si borramos BD
    resp = input("\n¿Borrar base de datos y empezar de cero? (s/n): ").lower()
    reset_db = (resp == 's')
    
    # 3. Cargar modelos (Late Fusion: E5 + CLIP)
    model_e5, model_clip = cargar_modelos()
    col_texto, col_imagen = iniciar_base_datos(reset_db)
    
    # 4. Procesar PDFs
    for pdf in pdfs:
        nombre = pdf.name
        print(f"\n{'─'*40}")
        print(f"Procesando: {nombre}")
        print(f"{'─'*40}")
        
        # Fase 1: Extracción
        texto, imagenes = extraer_info_pdf(str(pdf))
        
        # Fase 2: Carril Texto (E5 Multilingüe)
        n_txt = procesar_carril_texto(texto, nombre, model_e5, col_texto)
        print(f"   ✓ Texto: {n_txt} fragmentos (E5 multilingüe)")
        
        # Fase 3: Carril Imagen (CLIP)
        n_img = procesar_carril_imagen(imagenes, nombre, model_clip, col_imagen)
        print(f"   ✓ Imágenes: {n_img} embeddings visuales (CLIP)")

    print("\n" + "="*60)
    print(" PROCESAMIENTO TERMINADO")
    print("="*60)
    print(f"\nBase de datos: {DB_DIR}")
    print(f"Modelo texto:  {MODELO_E5}")
    print(f"Modelo imagen: {MODELO_CLIP}")
    print(f"\nEjecuta: streamlit run 02_streamlit_rag_multimodal_V2.py")

if __name__ == "__main__":
    main()