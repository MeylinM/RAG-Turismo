# ============================================================================
# 0. IMPORTS
# ============================================================================

import os
import json #Metadata manual
import logging
from pathlib import Path

from dotenv import load_dotenv
import chromadb # base de datos vectorial
import fitz  # leer PDFs
from PIL import Image # cargar imágenes
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer #E5 y CLIP

import utils

# ============================================================================
# 1. CONFIGURACIÓN GLOBAL
# ============================================================================

load_dotenv()
utils.setup_logging()
logger = logging.getLogger("RAG_Multimodal_Metadata")

# Carpetas
DATA_DIR = Path("../data/pdf")
IMAGENES_DIR = Path("../data/imagenes_extraidas")
METADATA_MANUAL_PATH = Path("../data/metadatos_pdfs.json")

# ChromaDB
DB_DIR = "../chroma_db_multimodal_v2"
COLLECTION_NAME = "documentos_multimodal"

# Modelos
MODELO_E5 = "intfloat/multilingual-e5-base"
MODELO_CLIP = "clip-ViT-B-32"

# Crear carpetas si no existen
IMAGENES_DIR.mkdir(parents=True, exist_ok=True)

# Filtros para imágenes
MIN_ANCHO = 200          # ancho mínimo
MIN_ALTO = 200           # alto mínimo
MIN_BYTES = 6144         # 6 KB mínimo
MAX_RATIO_ASPECTO = 3.5  # evita banners muy alargados
MIN_PIXELES_TOTALES = 150_000  # ancho * alto mínimo

# ============================================================================
# 2. CARGA DE METADATA MANUAL
# ============================================================================

def cargar_metadata_manual():
    """
    Carga el metadata turístico desde JSON y lo indexa por nombre de PDF.
    """
    with open(METADATA_MANUAL_PATH, "r", encoding="utf-8") as f:
        lista_metadata = json.load(f)

    metadata_por_pdf = {}

    for item in lista_metadata:
        nombre_pdf = item["nombre_archivo"]

        metadata_por_pdf[nombre_pdf] = {
            "titulo": item.get("titulo"),
            "ubicacion": item.get("ubicacion"),
            "categoria": item.get("categoria"),
            "palabras_clave": item.get("palabras_clave", []),

            # Derivados útiles para routing
            "pais": "Japón" if "Japón" in item.get("ubicacion", "") else "España",
            "dominio": "turismo"
        }

    logger.info(f"Metadata manual cargado: {len(metadata_por_pdf)} PDFs")
    return metadata_por_pdf

# ============================================================================
# 3. CARGA DE MODELOS (Late Fusion)
# ============================================================================

def cargar_modelos():
    """
    Carga los modelos de embeddings:
    - E5 multilingüe → texto
    - CLIP → imágenes
    """
    logger.info("Cargando modelos de embeddings...")

    # Modelo de TEXTO (E5)
    logger.info(f"Cargando modelo E5: {MODELO_E5}")
    model_e5 = SentenceTransformer(MODELO_E5)
    logger.info(
        f"Modelo E5 cargado (dimensión: {model_e5.get_sentence_embedding_dimension()})"
    )

    # Modelo de IMAGEN (CLIP)
    logger.info(f"Cargando modelo CLIP: {MODELO_CLIP}")
    model_clip = SentenceTransformer(MODELO_CLIP)
    logger.info(
        f"Modelo CLIP cargado (dimensión: {model_clip.get_sentence_embedding_dimension()})"
    )

    return model_e5, model_clip

# ============================================================================
# 4. INICIALIZACIÓN DE CHROMADB
# ============================================================================

def iniciar_chromadb(reset=False):
    """
    Inicializa ChromaDB con dos colecciones:
    - texto (E5)
    - imagen (CLIP)
    """
    client = chromadb.PersistentClient(path=DB_DIR)

    if reset:
        for nombre in [
            f"{COLLECTION_NAME}_texto",
            f"{COLLECTION_NAME}_imagen"
        ]:
            try:
                client.delete_collection(nombre)
                logger.info(f"Colección eliminada: {nombre}")
            except Exception:
                pass

    col_texto = client.get_or_create_collection(
        name=f"{COLLECTION_NAME}_texto"
    )

    col_imagen = client.get_or_create_collection(
        name=f"{COLLECTION_NAME}_imagen"
    )

    logger.info("ChromaDB inicializada (texto + imagen)")
    return col_texto, col_imagen

# ============================================================================
# 5. EXTRACCIÓN DE TEXTO E IMÁGENES DE PDF
# ============================================================================

def extraer_info_pdf(ruta_pdf):
    """
    Extrae todo el texto y las imágenes de un PDF.
    Devuelve:
    - texto completo
    - lista de imágenes con metadata básico
    """
    doc = fitz.open(ruta_pdf)

    texto_completo = ""
    lista_imagenes = []

    nombre_pdf = os.path.basename(ruta_pdf)

    for num_pag, pagina in enumerate(doc, start=1):
        # TEXTO
        texto_pagina = pagina.get_text()
        texto_completo += f"\n--- PÁGINA {num_pag} ---\n{texto_pagina}"

        # IMÁGENES
        for idx_img, img in enumerate(pagina.get_images(full=True)):
            xref = img[0]

            try:
                base = doc.extract_image(xref)
                bytes_img = base["image"]
                ext = base["ext"]
                width = base["width"]
                height = base["height"]
                total_pixeles = width * height
                ratio = (width / height) if width > height else (height / width)
                size_bytes = len(bytes_img)

                # filtros agresivos
                if width < MIN_ANCHO or height < MIN_ALTO:
                    continue
                if total_pixeles < MIN_PIXELES_TOTALES:
                    continue
                if size_bytes < MIN_BYTES:
                    continue
                if ratio > MAX_RATIO_ASPECTO:
                    continue

                # Guardar imagen válida
                nombre_imagen = f"{nombre_pdf}_pag{num_pag}_img{idx_img}.{ext}"
                ruta_imagen = IMAGENES_DIR / nombre_imagen
                with open(ruta_imagen, "wb") as f:
                    f.write(bytes_img)

                lista_imagenes.append({
                    "ruta": str(ruta_imagen),
                    "pagina": num_pag,
                    "indice": idx_img,
                    "nombre": nombre_imagen,
                    "dimensiones": f"{width}x{height}"
                })

            except Exception as e:
                logger.warning(f"Error extrayendo imagen en página {num_pag}: {e}")

    logger.info(
        f"{nombre_pdf}: texto extraído ({len(texto_completo)} caracteres), "
        f"{len(lista_imagenes)} imágenes válidas tras filtros"
    )

    return texto_completo, lista_imagenes

# ============================================================================
# 6. CHUNKING DE TEXTO
# ============================================================================

def dividir_en_chunks(texto, tam_chunk=500, solapamiento=100):
    """
    Divide un texto largo en chunks solapados.
    """
    chunks = []
    inicio = 0
    longitud = len(texto)

    while inicio < longitud:
        fin = inicio + tam_chunk
        chunk = texto[inicio:fin]
        chunks.append(chunk)
        inicio += tam_chunk - solapamiento

    return chunks

# ============================================================================
# 7. EMBEDDINGS DE TEXTO
# ============================================================================

def generar_embedding_texto(texto):
    """
    Genera embedding usando el modelo E5.
    """
    
    texto = "passage: " + texto
    embedding = modelo_texto.encode(texto, normalize_embeddings=True)
    return embedding.tolist()

# ============================================================================
# 8. INDEXAR TEXTO EN CHROMA
# ============================================================================

def indexar_texto_chroma(
    texto,
    metadata_base,
    nombre_pdf
):
    chunks = dividir_en_chunks(texto)

    ids = []
    documentos = []
    embeddings = []
    metadatos = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{nombre_pdf}_chunk_{i}")
        documentos.append(chunk)
        embeddings.append(generar_embedding_texto(chunk))

        metadata = metadata_base.copy()

        # Convertir listas a strings para ChromaDB
        for k, v in metadata.items():
            if isinstance(v, list):
                metadata[k] = ", ".join(v)

        metadata.update({
            "tipo": "texto",
            "chunk_id": i,
            "origen_pdf": nombre_pdf
        })
        metadatos.append(metadata)

    coleccion.add(
        ids=ids,
        documents=documentos,
        embeddings=embeddings,
        metadatas=metadatos
    )

    logger.info(f"{nombre_pdf}: {len(chunks)} chunks indexados")


# ============================================================================
# 9. EMBEDDINGS DE IMÁGENES Y CHROMA
# ============================================================================

def indexar_imagenes_chroma(lista_imagenes, metadata_base, nombre_pdf):
    """
    Genera embeddings de imágenes con CLIP y las guarda en ChromaDB
    """
    if not lista_imagenes:
        logger.info(f"{nombre_pdf}: no hay imágenes para indexar")
        return

    ids = []
    documentos = []
    embeddings = []
    metadatos = []

    for img_data in lista_imagenes:
        try:
            # Cargar imagen
            img_pil = Image.open(img_data["ruta"]).convert("RGB")

            # Embedding CLIP
            vector = modelo_imagen.encode(img_pil, normalize_embeddings=True).tolist()

            ids.append(f"{nombre_pdf}_img_{img_data['pagina']}_{img_data['indice']}")
            documentos.append(f"[IMAGEN] Página {img_data['pagina']}, archivo: {img_data['nombre']}")
            
            # Combinar metadata manual + automática
            metadata = metadata_base.copy()
            # Convertir listas a strings para ChromaDB
            for k, v in metadata.items():
                if isinstance(v, list):
                    metadata[k] = ", ".join(v)

            metadata.update({
                "tipo": "imagen",
                "pagina": img_data["pagina"],
                "imagen_nombre": img_data["nombre"],
                "imagen_path": img_data["ruta"]
            })
            metadatos.append(metadata)

            embeddings.append(vector)

        except Exception as e:
            logger.warning(f"Error procesando imagen {img_data['nombre']}: {e}")

    coleccion_imagen.add(
        ids=ids,
        documents=documentos,
        embeddings=embeddings,
        metadatas=metadatos
    )

    logger.info(f"{nombre_pdf}: {len(lista_imagenes)} imágenes indexadas")

# ============================================================================
# 10. MAIN: PROCESAMIENTO COMPLETO DE TODOS LOS PDF
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print(" RAG MULTIMODAL - INGESTA COMPLETA CON METADATA")
    print("="*60)

    # 1. Cargar metadata manual
    metadata_manual = cargar_metadata_manual()

    # 2. Cargar modelos
    modelo_texto, modelo_imagen = cargar_modelos()

    # 3. Iniciar ChromaDB (reset=True para empezar de cero)
    coleccion, coleccion_imagen = iniciar_chromadb(reset=True)

    # 4. Listar PDFs
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en {DATA_DIR}")
        exit(1)

    print(f"\nPDFs encontrados: {len(pdfs)}")
    for pdf in pdfs:
        print(f"   - {pdf.name}")

    # 5. Procesar cada PDF
    for pdf in pdfs:
        print("\n" + "-"*50)
        print(f"Procesando PDF: {pdf.name}")
        print("-"*50)

        # Extraer texto e imágenes
        texto, imagenes = extraer_info_pdf(str(pdf))

        # Obtener metadata manual para este PDF
        metadata_base = metadata_manual.get(pdf.name, {})
        if not metadata_base:
            print(f"Metadata manual no encontrada para {pdf.name}, se usará vacía")

        # Indexar texto
        indexar_texto_chroma(texto, metadata_base, pdf.stem)

        # Indexar imágenes
        indexar_imagenes_chroma(imagenes, metadata_base, pdf.stem)

    print("\n" + "="*60)
    print(" INGESTA COMPLETA FINALIZADA")
    print("="*60)
    print(f"Base de datos lista en: {DB_DIR}")

