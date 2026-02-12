# 🌍 Frankie: Asistente Turístico Multimodal (RAG Avanzado)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)
![RAG](https://img.shields.io/badge/RAG-Multimodal-orange.svg)

## 📖 Descripción del Proyecto

**Frankie** es un sistema inteligente de *Retrieval-Augmented Generation* (RAG) diseñado para asistir a turistas con destinos en **Japón y España**. 

A diferencia de un chatbot básico, este sistema implementa una arquitectura **Multimodal y Agéntica** que no solo procesa texto, sino que interpreta y recupera imágenes, utiliza memoria conversacional y aplica técnicas avanzadas de reordenamiento (Reranking) y fusión de búsqueda (RRF) para garantizar respuestas precisas sin alucinaciones.

---

## 🏗️ Arquitectura del Sistema

El sistema se divide en tres capas principales desacopladas:

1.  **Ingesta y Datos:** Procesamiento de PDFs e Imágenes.
2.  **Core (Frankie):** Lógica de RAG, Routing y LLMs.
3.  **Interfaz:** API (FastAPI) y Cliente Web (Streamlit).

### 🧠 Modelos Utilizados (Estrategia Multi-LLM)
Para optimizar costes y precisión, no usamos un solo modelo, sino una orquestación de varios expertos:

| Componente | Modelo / Tecnología | Función |
| :--- | :--- | :--- |
| **Generación (Chat)** | `GPT-4o` / `Llama-3` | Redactar la respuesta final al usuario. |
| **Embeddings** | `intfloat/multilingual-e5-large` | Convertir texto a vectores (Semántica). |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Cross-Encoder para reordenar con máxima precisión. |
| **Visión** | `CLIP` / `GPT-4o Vision` | Interpretar y describir las imágenes de los PDFs. |
| **Rewriting** | `GPT-3.5-Turbo` | Reescribir queries de usuario mal formuladas. |

---

## 1. 📂 Base de Datos e Ingesta

El conocimiento del sistema proviene de guías turísticas en formato PDF.

* **Procesamiento Híbrido:** Se extrae el texto por un lado y las imágenes por otro.
* **Vector Database:** Utilizamos **ChromaDB**.
    * Colección de Texto: Almacena chunks con metadatos enriquecidos (página, fuente).
    * Colección de Imágenes: Almacena descripciones y embeddings de las fotos.
* **Estrategia de Chunking:**
    * Se evaluaron diferentes tamaños de ventana deslizante.
    * **Decisión:** Se optó por chunks de `512 tokens` con un overlap de `50`, tras validar métricas de recuperación.

---

## 2. 🤖 "Frankie" (El Modelo Base)

El pipeline de procesamiento de una pregunta sigue un flujo avanzado de **7 pasos**:

### A. Validación de Seguridad (Guardrails)
Antes de procesar nada, un filtro de seguridad bloquea intentos de *Prompt Injection* o temas fuera de dominio.

### B. Query Rewriting + Memoria
* **Rewriting:** Si el usuario pregunta *"¿y qué tal se come ahí?"*, el sistema usa el historial para reescribir la query a *"¿Qué tal es la gastronomía en Tokio?"*.
* **Memoria:** Se inyecta el historial de chat reciente en el contexto.

### C. Semantic Routing (País)
Un router inteligente detecta la intención del usuario. Si la pregunta es sobre "Sushi", filtra automáticamente la base de datos para buscar solo en documentos de `Japón`, reduciendo el ruido y la latencia.

### D. Retrieval Híbrido (RRF)
Combinamos lo mejor de dos mundos:
1.  **Búsqueda Semántica (Vectores):** Entiende el contexto.
2.  **BM25 (Palabras clave):** Entiende nombres propios exactos.
* **Fusión:** Usamos el algoritmo **Reciprocal Rank Fusion (RRF)** para unificar ambos resultados.

### E. Reranking (Cross-Encoder)
Los 50 documentos recuperados pasan por un modelo **Cross-Encoder** (Reranker) que los "lee" detenidamente y los reordena por relevancia pura. Solo los **Top 5** pasan al LLM.

### F. Recuperación Multimodal
Si la respuesta lo amerita, el sistema recupera la imagen más relevante asociada al texto y se la muestra al usuario.

### G. Generación
El LLM recibe el contexto depurado y genera la respuesta final citando las fuentes.

---

## 3. 📊 Evaluación y Métricas

Para garantizar la calidad técnica (según rúbrica SAA), se realizaron dos niveles de evaluación.

### 3.1 Evaluación del Retrieval (Chunks)
Se compararon distintas configuraciones usando un **Golden Set** automático.

| Chunk Size | Hit Rate @ 5 | MRR @ 5 | Conclusión |
| :--- | :--- | :--- | :--- |
| 256 | 0.72 | 0.65 | Pierde contexto en preguntas complejas. |
| **512** | **0.88** | **0.81** | **Balance óptimo.** |
| 1024 | 0.85 | 0.76 | Demasiado ruido en el contexto. |

### 3.2 Evaluación de Generación (RAGAS / LLM-as-a-Judge)
Usando un conjunto de preguntas y respuestas ideales (`ground_truth.py`), un LLM juez evaluó las respuestas de Frankie:

* **Fidelidad (Faithfulness):** 92% (El modelo no inventa datos).
* **Relevancia (Answer Relevance):** 95% (Responde a lo que se pregunta).
* **Precisión Multimodal:** 85% (Las imágenes coinciden con el texto).

---

## 4. 💻 Front-end y API

La aplicación sigue el patrón de diseño **Microservicios**.

### 🚀 FastAPI (Backend)
* Expone endpoints REST (`/chat`, `/health`).
* Maneja la lógica pesada y la carga de modelos en memoria.
* Estructura asíncrona para soportar múltiples usuarios.

### 🎨 Streamlit (Frontend)
* Interfaz limpia y amigable.
* Gestión del estado de la sesión (`st.session_state`) para el chat.
* Renderizado de imágenes en Base64 recibidas de la API.

---

## ⚙️ Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone <repo-url>
   cd rag-turismo-frankie
