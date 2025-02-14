import os
import requests
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from collections import namedtuple
from database import connect_db, close_db
from utils import (
    build_or_load_vector_index,
    embed_and_search_in_faiss,
    rerank_with_llama,
    check_llama_status

)
import gradio as gr
from utils import buscar_cvs

def search_interface():
    with gr.Blocks(elem_id="search-container") as search_page:
        gr.Markdown("🔎 **Búsqueda de CVs con IA**", elem_id="search-title")
        
        job_input = gr.Textbox(label="Descripción del Puesto", elem_id="job-input")
        option_toggle = gr.Radio(["🔍 Solo RAG", "🤖 RAG + MLL (IA Avanzada)"], label="Método de búsqueda", elem_id="search-toggle")
        search_button = gr.Button("Buscar", elem_id="search-button")
        result_output = gr.Textbox(label="Resultados", elem_id="result-output")

        search_button.click(fn=buscar_cvs, inputs=[job_input, option_toggle], outputs=result_output)

    return search_page







# ✅ Comprobación de disponibilidad de Llama 3.1 en Ollama
def check_llama_status():
    """Verifica si Llama 3.1 está disponible en Ollama sin imprimir mensajes innecesarios."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

# ✅ Refinamiento de ranking con Llama 3.1
def rerank_with_llama(top_docs, job_description):
    """Usa Llama 3.1 para reordenar los CVs recuperados, asegurando que incluya ID y Nombre."""
    if not check_llama_status():
        print("❌ No hay conexión con Llama 3.1.")
        return []

    docs_info = "\n".join(
        [f"- ID: {doc['id']}, Nombre: {doc['name']}, Extracto: {doc['content']}" for doc in top_docs]
    )

    prompt = f"""
    Eres un experto en selección de personal. Se te proporciona una descripción de puesto y varios CVs.
    Evalúa cada candidato y asigna un puntaje de 1 a 100.

    **Descripción del puesto:**
    {job_description}

    **CVs analizados:**
    {docs_info}

    Devuelve la respuesta en formato JSON con la siguiente estructura:
    [
        {{"id": "ID_CV", "name": "Nombre del candidato", "score": 1-100, "reasons": "Razón breve"}},
        ...
    ]
    """

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3", "messages": [{"role": "user", "content": prompt}], "stream": False}
    )

    if response.status_code == 200:
        try:
            response_json = response.json()
            content = response_json.get("message", {}).get("content", "")

            # Extraer solo el JSON dentro de la respuesta
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))  # Devuelve una lista de dicts
            else:
                print("❌ No se encontró JSON válido en la respuesta de Llama 3.")
                print("🔍 Respuesta obtenida:", content)
                return []

        except json.JSONDecodeError:
            print("❌ Error al convertir la respuesta de Llama 3 en JSON.")
            print("🔍 Respuesta obtenida:", response.text)
            return []
    else:
        print(f"❌ Error en la respuesta de Llama 3: {response.text}")
        return []

# ✅ Construcción o carga del índice FAISS
def build_or_load_vector_index(cursor, rebuild=False):
    """Construye o carga un índice FAISS."""
    index_path = "faiss_index"

    if not rebuild and os.path.exists(index_path):
        print("✅ Cargando índice FAISS existente desde disco...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)

    print("🔹 Creando un nuevo índice FAISS con todos los CVs...")
    cursor.execute("SELECT id, resumen, titulo FROM cv")
    rows = cursor.fetchall()

    if not rows:
        print("❌ No hay CVs en la base de datos.")
        return None

    documents = [Document(page_content=row[1], metadata={"id": row[0], "category": row[2]}) for row in rows]
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(documents, embedding)
    os.makedirs(index_path, exist_ok=True)
    docsearch.save_local(index_path)
    return docsearch

# ✅ Búsqueda de CVs en FAISS
def embed_and_search_in_faiss(query_text, docsearch, top_k=5):
    """Busca los documentos más similares en FAISS."""
    results = docsearch.similarity_search_with_score(query_text, k=top_k)
    return [namedtuple("Match", ["page_content", "metadata", "score"])(doc.page_content, doc.metadata, dist) for doc, dist in results]

# ✅ Función principal para buscar CVs
def buscar_cvs(job_description, option):
    """Busca CVs relevantes y usa Llama 3.1 si se elige RAG + MLL."""
    conn, cursor = connect_db(db_name="test_cv_database.db")

    docsearch = build_or_load_vector_index(cursor, rebuild=False)
    if not docsearch:
        close_db(conn)
        return "❌ No se pudo cargar el índice FAISS."

    top_matches = embed_and_search_in_faiss(job_description, docsearch, top_k=5)

    if not top_matches:
        close_db(conn)
        return "❌ No se encontraron CVs relevantes."

    formatted_matches = [
        {
            "id": match.metadata.get("id", "Desconocido"),
            "name": match.metadata.get("Título", "Sin Nombre"),
            "content": match.page_content[:300]
        }
        for match in top_matches
    ]

    if option == "🤖 RAG + MLL (IA Avanzada)" and check_llama_status():
        final_rank = rerank_with_llama(formatted_matches, job_description)
    else:
        final_rank = formatted_matches

    close_db(conn)

    result_text = "=== Resultados ===\n\n"
    for i, item in enumerate(final_rank, start=1):
        cv_id = item.get("id", "Desconocido")
        name = item.get("name", "Sin Nombre")
        score = item.get("score", 0) if "score" in item else "N/A"
        reasons = item.get("reasons", "No se proporcionaron razones.")

        result_text += f"#{i} CV ID: {cv_id} | Nombre: {name}\n"
        result_text += f"Puntuación Llama 3: {score}\n" if option == "🤖 RAG + MLL (IA Avanzada)" else ""
        result_text += f"Razones:\n{reasons}\n\n"

    return result_text
