from config import get_api_key
from database import (
    connect_db, insert_cvs_from_dataset, close_db, reset_database
)
from utils import (
    extract_job_requirements,
    evaluate_cv,
    load_dataset,
    filter_cvs_by_category,
    filter_cvs_by_semantic_relevance
)
import time

def main():
    # Paso 1: Solicitar descripción del puesto
    print("Por favor, ingresa la descripción del puesto. Cuando termines, ingresa una línea vacía:")
    job_description_lines = []
    while True:
        line = input()
        if line.strip() == '':
            break
        job_description_lines.append(line)
    job_description = '\n'.join(job_description_lines)

    # Paso 2: Extraer requisitos del puesto usando IA
    api_key = get_api_key()
    job_requirements = extract_job_requirements(job_description, api_key)

    print("\nRequisitos del puesto extraídos:", job_requirements)

    # Paso 3: Preparar la base de datos
    reset_database()
    conn, cursor = connect_db()

    # Paso 4: Cargar y almacenar los CVs en la base de datos
    dataset = load_dataset("Resume.csv")
    insert_cvs_from_dataset(cursor, dataset)
    conn.commit()
    print("CVs insertados en la base de datos.")

    # Paso 5: Filtrar CVs por categorías relevantes
    categories = job_requirements.get("categories", [])
    filtered_cvs = filter_cvs_by_category(cursor, categories)
    if not filtered_cvs:
        print("No se encontraron CVs en las categorías relevantes. Usando todos los CVs disponibles.")
        cursor.execute("SELECT * FROM cv")
        filtered_cvs = cursor.fetchall()

    print(f"Se encontraron {len(filtered_cvs)} CVs relevantes para las categorías especificadas.")

    # Paso 6: Filtrar por relevancia semántica
    threshold = 0.75
    filtered_cvs = filter_cvs_by_semantic_relevance(filtered_cvs, job_description, threshold, api_key)

    print(f"{len(filtered_cvs)} CVs tras filtrar por similitud semántica.")

    # Paso 7: Evaluar y rankear los CVs filtrados
    ranked_cvs = []
    max_cvs_to_process = min(20, len(filtered_cvs))  # Limitar el número de CVs procesados
    for idx, cv in enumerate(filtered_cvs[:max_cvs_to_process], start=1):
        cv_info = {
            "id": cv[0],
            "resume_str": cv[1],
            "category": cv[2].strip().lower() if isinstance(cv[2], str) else ''
        }
        print(f"Procesando CV {idx}/{max_cvs_to_process} (ID: {cv_info['id']}, Categoría: {cv_info['category']})...")
        score, explanation = evaluate_cv(cv_info, job_requirements, api_key)
        ranked_cvs.append((cv_info, score, explanation))

        # Evitar saturar la API (1 segundo de pausa)
        time.sleep(1)

    # Paso 8: Ordenar y mostrar los mejores CVs
    top_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)[:5]
    print("\nLos 5 CVs más cualificados para el puesto:")
    for idx, (cv_info, score, explanation) in enumerate(top_cvs, start=1):
        print(f"\n#{idx} CV")
        print(f"ID: {cv_info['id']}")
        print(f"Categoría: {cv_info['category']}")
        print(f"Puntuación total: {score:.2f}")
        print("Explicación de la puntuación:")
        for line in explanation:
            print(f" - {line}")

    # Paso 9: Cerrar la conexión a la base de datos
    close_db(conn)

if __name__ == "__main__":
    main()
