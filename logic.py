import time
from database import connect_db, close_db
from utils import (
    build_or_load_vector_index,
    embed_and_search_in_faiss,
    rerank_with_llama,
    check_llama_status
)

def buscar_cvs(job_description, use_mll):
    """Función para buscar y evaluar CVs con RAG y opcionalmente con MLL."""
    conn, cursor = connect_db(db_name="cv_database.db")


    # Cargar índice FAISS
    docsearch = build_or_load_vector_index(cursor, rebuild=True)

    if not docsearch:
        close_db(conn)
        return "❌ No se pudo cargar el índice FAISS."

    # Buscar los 5 CVs más relevantes
    start_time = time.time()
    top_matches = embed_and_search_in_faiss(job_description, docsearch, top_k=5)
    
    if not top_matches:
        close_db(conn)
        return "❌ No se encontraron CVs relevantes."

    # Formatear los datos antes de enviarlos a Llama 3
    formatted_matches = [
        {
            "id": match.metadata.get("id", "Desconocido"),
            "name": match.metadata.get("Nombre", "Sin Nombre"),
            "content": match.page_content[:300]
        }
        for match in top_matches
    ]

    if use_mll and check_llama_status():
        final_rank = rerank_with_llama(formatted_matches, job_description)
    else:
        final_rank = [{"id": doc["id"], "name": doc["name"], "score": "N/A", "reasons": "Procesamiento básico (RAG)."} for doc in formatted_matches]

    close_db(conn)

    # Calcular tiempo transcurrido
    elapsed_time = round(time.time() - start_time, 2)
    
    # Construir resultado para mostrar
    result_text = f"=== Ranking final refinado con Llama 3.1 ===\n(Tiempo de procesamiento: {elapsed_time}s)\n\n"
    for i, item in enumerate(final_rank, start=1):
        cv_id = item.get("id", "Desconocido")
        name = item.get("name", "Sin Nombre")
        score = item.get("score", "N/A")
        reasons = item.get("reasons", "No se proporcionaron razones.")

        result_text += f"#{i} CV ID: {cv_id} | Nombre: {name}\n"
        result_text += f"Puntuación Llama 3: {score}\n"
        result_text += f"Razones:\n{reasons}\n\n"

    return result_text
