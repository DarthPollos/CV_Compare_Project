import os
import requests
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from collections import namedtuple

# Import de tu módulo de DB, ajusta si el path es diferente
from database import connect_db, close_db


###############################################################################
# 1. Verificar estado de LLaMA
###############################################################################
def check_llama_status():
    """Verifica si Llama 3.1 está disponible en Ollama sin imprimir mensajes innecesarios."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


###############################################################################
# 2. Re-rank con LLaMA 3.1
###############################################################################
def rerank_with_llama(top_docs, job_description):
    """
    Usa Llama 3.1 para reordenar los CVs recuperados.
    Solo se envían los CVs ya filtrados y limitados en número para no sobrecargar al LLM.
    """
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


###############################################################################
# 3. Construir o cargar el índice FAISS
###############################################################################
# -- Mantenemos esta función casi igual, pero podemos reutilizar embedding y docsearch
#   si queremos todavía más optimización.
def build_or_load_vector_index(cursor, rebuild=False):
    """Construye o carga un índice FAISS."""
    index_path = "faiss_index"

    # Evita reconstruir el índice si ya existe y no se pide explícitamente
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

    documents = [
        Document(page_content=row[1], metadata={"id": row[0], "category": row[2]}) 
        for row in rows
    ]
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(documents, embedding)
    os.makedirs(index_path, exist_ok=True)
    docsearch.save_local(index_path)
    return docsearch


###############################################################################
# 4. Búsqueda en FAISS
###############################################################################
def embed_and_search_in_faiss(query_text, docsearch, top_k=10):
    """
    Busca los documentos más similares en FAISS con un top_k un poco más grande,
    para luego filtrar manualmente.
    """
    results = docsearch.similarity_search_with_score(query_text, k=top_k)
    # results = [(Document, score), ...]
    # Empaquetamos con namedtuple para mayor legibilidad
    Match = namedtuple("Match", ["page_content", "metadata", "score"])
    return [Match(doc.page_content, doc.metadata, dist) for doc, dist in results]


###############################################################################
# 5. Filtro previo antes de re-rank
###############################################################################
def filter_top_matches(top_matches, distance_threshold=0.7, max_pass=5):
    """
    Filtra los resultados con un umbral de distancia (mientras más bajo, mayor similitud).
    Luego selecciona hasta `max_pass` CVs.
    
    Ajusta 'distance_threshold' a conveniencia en función de las distancias de tu FAISS.
    """
    # 1. Filtra por umbral
    filtered = [m for m in top_matches if m.score < distance_threshold]
    
    # 2. Ordena por score ascendente (mejor similitud primero)
    filtered.sort(key=lambda x: x.score)
    
    # 3. Toma un máximo de max_pass
    return filtered[:max_pass]


###############################################################################
# 6. Función principal para buscar CVs
###############################################################################
def buscar_cvs(job_description, option):
    """
    1) Conecta a DB, carga o crea el índice FAISS.
    2) Busca un top_k mayor (por ej. 10).
    3) Filtra con un umbral y reduce a 'max_pass' (por defecto 5).
    4) Envía SOLO esos CVs filtrados al rerank con LLaMA (si corresponde).
    5) Devuelve un texto con los resultados.
    """

    conn, cursor = connect_db(db_name="test_cv_database.db")

    docsearch = build_or_load_vector_index(cursor, rebuild=False)
    if not docsearch:
        close_db(conn)
        return "❌ No se pudo cargar el índice FAISS."

    # 1. Búsqueda inicial con top_k=10 (por ejemplo)
    top_matches = embed_and_search_in_faiss(job_description, docsearch, top_k=10)

    if not top_matches:
        close_db(conn)
        return "❌ No se encontraron CVs relevantes."

    # 2. Filtrado previo
    #    Ajusta 'distance_threshold' si ves que pocos o demasiados CVs pasan el filtro
    filtered_matches = filter_top_matches(top_matches, distance_threshold=0.7, max_pass=5)
    if not filtered_matches:
        close_db(conn)
        return "❌ Tras el filtrado, no se encontraron CVs suficientemente cercanos."

    # 3. Formatea los CVs filtrados para LLaMA
    formatted_matches = [
        {
            "id": match.metadata.get("id", "Desconocido"),
            "name": match.metadata.get("Título", "Sin Nombre"),
            # Reducimos contenido a 200 chars para no saturar el prompt
            "content": match.page_content[:200]
        }
        for match in filtered_matches
    ]

    # 4. Re-rank con LLaMA (si la opción es RAG + MLL)
    if option == "🤖 RAG + MLL (IA Avanzada)" and check_llama_status():
        final_rank = rerank_with_llama(formatted_matches, job_description)
    else:
        # Si no usamos LLaMA, sólo devolvemos el orden de similitud
        # Podríamos asignar un "score" manual basado en la distancia
        final_rank = []
        for m in filtered_matches:
            final_rank.append({
                "id": m.metadata.get("id", "Desconocido"),
                "name": m.metadata.get("Título", "Sin Nombre"),
                # Podríamos voltear la distancia (score = 1-dist) sólo para referencia
                "score": round(1 - m.score, 2),
                "reasons": "Ranking basado en similitud (sin LLaMA)."
            })

    close_db(conn)

    # 5. Construye el string de resultados
    result_text = "=== Resultados ===\n\n"
    for i, item in enumerate(final_rank, start=1):
        cv_id = item.get("id", "Desconocido")
        name = item.get("name", "Sin Nombre")
        score = item.get("score", 0)
        reasons = item.get("reasons", "No se proporcionaron razones.")

        result_text += f"#{i} CV ID: {cv_id} | Nombre: {name}\n"
        if option == "🤖 RAG + MLL (IA Avanzada)":
            result_text += f"Puntuación Llama 3: {score}\n"
        else:
            result_text += f"Puntuación aproximada (similitud): {score}\n"
        result_text += f"Razones:\n{reasons}\n\n"

    return result_text
