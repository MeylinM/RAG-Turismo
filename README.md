# 🏛️ Cicerón: Asistente Turístico Multimodal e Inteligente

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)
![RAG](https://img.shields.io/badge/RAG-Multimodal-orange.svg)

## 📖 Descripción del Proyecto

**Cicerón** es un sistema avanzado de *Retrieval-Augmented Generation* (RAG) especializado en turismo para **Japón y España**. 

A diferencia de los chatbots convencionales, Cicerón es **Multimodal** (entiende texto e imágenes) y utiliza una arquitectura de agentes para garantizar que la información sea veraz, utilizando guías oficiales y no el conocimiento general alucinado de un LLM.

---

## 🏗️ Arquitectura Técnica

El sistema sigue el patrón de diseño de microservicios para desacoplar la lógica de la interfaz.

### 🛠️ Stack Tecnológico
* **Backend:** `FastAPI` (Gestión de rutas, asincronía y lógica RAG).
* **Frontend:** `Streamlit` (Interfaz de chat con soporte de imágenes).
* **Base de Datos Vectorial:** `ChromaDB` (Colecciones separadas para texto e imágenes).
* **Orquestación:** Python puro (sin frameworks pesados como LangChain para mayor control).

### 🧠 Modelos Implementados
Para lograr la máxima puntuación en precisión, utilizamos una estrategia **Multi-LLM**:

| Componente | Modelo | Función |
| :--- | :--- | :--- |
| **Embeddings** | `intfloat/multilingual-e5-large` | Búsqueda semántica de alta calidad en español. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Reordenamiento (Cross-Encoder) para precisión crítica. |
| **Visión** | `CLIP` / `GPT-4o` | Procesamiento y descripción de imágenes turísticas. |
| **Generación** | `GPT-4o` / `Llama-3` | Respuesta final al usuario. |
| **Router** | `Semantic Router` (Custom) | Clasificación de intención (País/Tema). |

---

## 🚀 Pipeline Avanzado (RAG Flow)

Cicerón implementa 4 técnicas avanzadas (superando las 3 requeridas por la rúbrica):

1.  **Query Rewriting:** Transforma preguntas vagas ("sitios japo madrid") en consultas ricas ("Restaurantes de comida japonesa en Madrid").
2.  **Routing Semántico:** Detecta si la pregunta es sobre Japón o España y filtra la base de datos automáticamente para reducir ruido.
3.  **Reciprocal Rank Fusion (RRF):** Fusiona resultados de **Búsqueda Vectorial** (significado) y **BM25** (palabras clave exactas).
4.  **Cross-Encoder Reranking:** Un modelo especializado re-evalúa los 50 documentos recuperados y selecciona solo los 5 mejores.

---

## 📊 Evaluación y Métricas (Evidencias)

A continuación se presentan las evidencias de rendimiento del sistema, requisito fundamental para la validación técnica.

### 1. Evaluación del Retrieval (Comparativa de Chunks)
Se realizaron pruebas con distintos tamaños de chunk para encontrar el balance entre contexto y precisión.

> **Captura de los resultados del script `02_evaluar_chunks.py`:**

![Evidencia Chunks](img/metrics_chunks.png)  
*(Si no carga la imagen: Hit Rate promedio de 0.88 con chunks de 512 tokens)*

### 2. Evaluación de Generación (RAGAS / LLM-Judge)
Utilizando un "Golden Set" (preguntas con respuestas ideales), evaluamos la fidelidad y relevancia de Cicerón.

> **Captura de los resultados del script `ragas.py`:**

![Evidencia Ragas](img/metrics_ragas.png)

* **Fidelidad:** Mide si el modelo inventa datos. (Objetivo > 90%)
* **Exactitud:** Comparación semántica con la respuesta ideal.
* **Multimodalidad:** Porcentaje de veces que recuperó una imagen correcta.

---

## ⚙️ Instalación y Despliegue

### Requisitos Previos
* Python 3.10+
* Clave de OpenAI (`OPENAI_API_KEY`)

### Paso 1: Configuración Automática
Hemos incluido un script para facilitar la instalación en Windows:
1.  Haz doble clic en el archivo `setup.bat`.
2.  Esto creará el entorno virtual e instalará las dependencias limpias.

### Paso 2: Ejecución Manual
Si prefieres usar la terminal:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Iniciar el Servidor (Backend)
uvicorn src.03_API_Separada:app --reload

# 3. Iniciar la App (Frontend) en otra terminal
streamlit run src/04_APP.py
