# ============================================================================
# 0. IMPORTS
# ============================================================================
import os
import logging
import chromadb
import fitz  # PyMuPDF
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================
# Cargar variables de entorno (si tienes un archivo .env)
load_dotenv()

# --- Rutas ---
# Ajusta esto si tus carpetas están en otro lugar
DATA_DIR = Path("../data/pdf")
IMAGENES_DIR = Path("../data/imagenes_extraidas")
DB_DIR = "../chroma_db_multimodal"
COLLECTION_NAME = "documentos_multimodal"

# --- Configuración de Modelos (Late Fusion) ---
MODELO_E5 = "intfloat/multilingual-e5-base" # Texto
MODELO_CLIP = "clip-ViT-B-32"               # Imágenes

# --- FILTROS DE CALIDAD DE IMAGEN (NUEVO) ---
# Evita indexar iconos, líneas separadoras y logotipos pequeños
MIN_ANCHO = 200          # Mínimo 200px de ancho
MIN_ALTO = 200           # Mínimo 200px de alto
MIN_BYTES = 3072         # Mínimo 3KB (evita iconos vectoriales simples)
MAX_RATIO_ASPECTO = 3.5  # Evita barras alargadas (ej. 1000x50px)

# --- FILTROS DE CALIDAD DE IMAGEN (VERSIÓN AGRESIVA) ---
# 1. Filtro de Área Total:
# Exigimos un mínimo de píxeles totales (ancho x alto).
# 150,000 px equivale aprox a una imagen de 400x375.
# Esto elimina iconos que "mienten" sobre sus dimensiones con bordes transparentes.
MIN_PIXELES_TOTALES = 150000

# 2. Filtro de Peso (Bytes):
# Subimos de 3KB a 6KB. Las fotos reales, incluso muy comprimidas,
# suelen pesar más. Los gráficos vectoriales o imágenes planas pesan poco.
MIN_BYTES = 6144  # 6 KB

# 3. Filtro de Proporción:
# Mantenemos esto para eliminar banners muy alargados.
MAX_RATIO_ASPECTO = 3.5

# Crear carpetas si no existen
if not DATA_DIR.exists():
    print(f"ADVERTENCIA: La carpeta {DATA_DIR} no existe. Por favor créala y pon tus PDFs ahí.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
if not IMAGENES_DIR.exists():
    IMAGENES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. FUNCIONES AUXILIARES (Logging y Limpieza)
# ============================================================================
def setup_logging():
    """Configura el sistema de logs para ver qué pasa en la consola"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("RAG_Multimodal")

logger = setup_logging()

def limpiar_texto(texto):
    """Limpia el texto extraído del PDF para mejorar los embeddings"""
    # Reemplazar saltos de línea y múltiples espacios por un solo espacio
    texto = re.sub(r'\s+', ' ', texto)
    # Eliminar caracteres extraños o no imprimibles
    texto = texto.strip()
    return texto

def hacer_chunking(texto, chunk_size=500, overlap=50):
    """Divide el texto largo en trozos más pequeños con solapamiento"""
    palabras = texto.split()
    chunks = []
    if not palabras:
        return []
    
    for i in range(0, len(palabras), chunk_size - overlap):
        chunk = " ".join(palabras[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(palabras):
            break
    return chunks

# ============================================================================
# 3. CARGA DE MODELOS E INICIO DE BD
# ============================================================================
def cargar_modelos():
    logger.info("Cargando modelos de IA (esto puede tardar un poco la primera vez)...")
    
    # 1. Modelo E5 para TEXTO
    logger.info(f"   Cargando E5 para texto: {MODELO_E5}")
    model_e5 = SentenceTransformer(MODELO_E5)
    
    # 2. Modelo CLIP para IMÁGENES
    logger.info(f"   Cargando CLIP para imágenes: {MODELO_CLIP}")
    model_clip = SentenceTransformer(MODELO_CLIP)
    
    return model_e5, model_clip

def iniciar_base_datos(reset=False):
    client = chromadb.PersistentClient(path=DB_DIR)
    
    if reset:
        logger.warning("Borrando colecciones existentes...")
        try:
            client.delete_collection(f"{COLLECTION_NAME}_texto")
            client.delete_collection(f"{COLLECTION_NAME}_imagen")
        except:
            pass
    
    col_texto = client.get_or_create_collection(f"{COLLECTION_NAME}_texto")
    col_imagen = client.get_or_create_collection(f"{COLLECTION_NAME}_imagen")
    
    return col_texto, col_imagen

# ============================================================================
# 4. EXTRACCIÓN DE PDF (CON FILTROS AGRESIVOS)
# ============================================================================
def extraer_info_pdf(ruta_pdf):
    """
    Abre el PDF y saca TODO: Texto crudo + Imágenes guardadas en disco.
    APLICA FILTROS AGRESIVOS para asegurar que solo pasan fotos reales.
    """
    doc = fitz.open(ruta_pdf)
    texto_completo = ""
    lista_imagenes = []
    nombre_base = os.path.basename(ruta_pdf).replace('.pdf', '')
    
    logger.info(f"Procesando archivo: {nombre_base}")
    
    # Recorrer páginas
    for num_pag, pagina in enumerate(doc, 1):
        # A. Texto
        texto_pagina = pagina.get_text()
        texto_completo += f"\n--- PÁGINA {num_pag} ---\n" + texto_pagina
        
        # B. Imágenes
        img_list = pagina.get_images(full=True)
        
        for i, img in enumerate(img_list):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                bytes_img = base["image"]
                ext = base["ext"]
                
                # --- DATOS PARA FILTRADO ---
                width = base["width"]
                height = base["height"]
                size_bytes = len(bytes_img)
                total_pixeles = width * height
                
                # Evitar división por cero o imágenes sin dimensiones
                if total_pixeles == 0:
                    continue
                    
                ratio = width / height if width > height else height / width

                # --- APLICAR FILTROS AGRESIVOS ---
                
                # 1. Filtro de Área (Píxeles Totales)
                # Este es el más efectivo contra iconos con bordes transparentes.
                if total_pixeles < MIN_PIXELES_TOTALES:
                    # logger.debug(f"Descartada por pocos píxeles ({total_pixeles}): p{num_pag}_img{i}")
                    continue
                
                # 2. Filtro de Peso (Bytes)
                # Elimina imágenes grandes pero vacías (marcas de agua suaves, fondos planos).
                if size_bytes < MIN_BYTES:
                    # logger.debug(f"Descartada por peso bajo ({size_bytes}b): p{num_pag}_img{i}")
                    continue

                # 3. Filtro de Proporción (Evitar barras/líneas)
                if ratio > MAX_RATIO_ASPECTO:
                    continue

                # --- GUARDAR SI PASA LOS FILTROS ---
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
                    "nombre": nombre_archivo,
                    "dimensiones": f"{width}x{height}"
                })
            except Exception as e:
                logger.error(f"Error extrayendo imagen en p{num_pag}: {e}")
                pass 

    logger.info(f"   --> Texto extraído: {len(texto_completo)} caracteres")
    logger.info(f"   --> Imágenes VÁLIDAS extraídas: {len(lista_imagenes)}")
    
    return texto_completo, lista_imagenes

# ============================================================================
# 5. CARRIL DE TEXTO (E5)
# ============================================================================
def procesar_carril_texto(texto, nombre_pdf, model_e5, col_texto):
    if not texto.strip():
        return 0
        
    # Limpieza y chunking usando las funciones internas
    texto_limpio = limpiar_texto(texto)
    chunks = hacer_chunking(texto_limpio, chunk_size=500, overlap=50)
    
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
        docs.append(chunk)  # Guardamos SIN prefijo para mostrar luego
        metas.append({
            "source": nombre_pdf,
            "tipo": "texto",
            "chunk_index": i
        })
    
    if ids:
        col_texto.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    
    return len(ids)

# ============================================================================
# 6. CARRIL DE IMAGEN (CLIP)
# ============================================================================
def procesar_carril_imagen(lista_imagenes, nombre_pdf, model_clip, col_imagen):
    if not lista_imagenes:
        return 0
    
    ids = []
    embs = []
    docs = []
    metas = []
    
    logger.info(f"   Generando embeddings para {len(lista_imagenes)} imágenes...")
    
    for img_data in tqdm(lista_imagenes, desc="Embeddings Imagen"):
        try:
            # Cargar imagen
            imagen_pil = Image.open(img_data["ruta"]).convert('RGB')
            
            # CLIP: Embedding visual directo
            vector = model_clip.encode(imagen_pil, normalize_embeddings=True).tolist()
            
            # Metadata descriptiva
            descripcion_metadata = f"[IMAGEN] Archivo: {img_data['nombre']} (Pág {img_data['pagina']})"
            
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
            
        except Exception as e:
            logger.error(f"Error procesando imagen {img_data['nombre']}: {e}")
            
    if ids:
        col_imagen.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        
    return len(ids)

# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print(" RAG MULTIMODAL - INDEXADOR (E5 + CLIP)")
    print("="*60)
    
    # 1. Buscar PDFs
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\n[ERROR] No hay archivos PDF en la carpeta: {DATA_DIR}")
        print("Por favor, pon tus documentos ahí e intenta de nuevo.")
        return
    
    print(f"\nPDFs encontrados: {len(pdfs)}")
    
    # 2. Preguntar si borramos BD
    resp = input("\n¿Borrar base de datos existente y empezar de cero? (s/n): ").lower()
    reset_db = (resp == 's')
    
    # 3. Cargar modelos e iniciar BD
    model_e5, model_clip = cargar_modelos()
    col_texto, col_imagen = iniciar_base_datos(reset_db)
    
    # 4. Procesar PDFs
    total_texto = 0
    total_imagenes = 0
    
    for pdf in pdfs:
        print(f"\n{'─'*40}")
        
        # Fase 1: Extracción
        texto, imagenes = extraer_info_pdf(str(pdf))
        
        # Fase 2: Carril Texto
        n_txt = procesar_carril_texto(texto, pdf.name, model_e5, col_texto)
        total_texto += n_txt
        print(f"   ✓ Texto indexado: {n_txt} fragmentos")
        
        # Fase 3: Carril Imagen
        n_img = procesar_carril_imagen(imagenes, pdf.name, model_clip, col_imagen)
        total_imagenes += n_img
        print(f"   ✓ Imágenes indexadas: {n_img}")

    print("\n" + "="*60)
    print(" PROCESAMIENTO TERMINADO")
    print("="*60)
    print(f"Total Fragmentos Texto: {total_texto}")
    print(f"Total Imágenes:         {total_imagenes}")
    print(f"\nBase de datos guardada en: {DB_DIR}")

if __name__ == "__main__":
    main()