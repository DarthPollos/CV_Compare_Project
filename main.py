from config import get_api_key
from database import (
    connect_db, insert_cvs_from_dataset, get_all_cvs, close_db, reset_database
)
from utils import extract_job_requirements, evaluate_cv, load_dataset
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
    
    # Paso 5: Recuperar y evaluar los CVs
    cvs = get_all_cvs(cursor)
    
    # Procesar todos los CVs (o limitar el número por motivos prácticos)
    max_cvs = 20  # Puedes ajustar este número según tus necesidades
    cvs = cvs[:max_cvs]
    
    ranked_cvs = []
    for idx, cv in enumerate(cvs, start=1):
        cv_info = {
            "id": cv[0],
            "resume_str": cv[1],
            "category": cv[2].strip().lower() if isinstance(cv[2], str) else ''
        }
        print(f"Procesando CV {idx}/{len(cvs)} (ID: {cv_info['id']}, Categoría: {cv_info['category']})...")
        score, explanation = evaluate_cv(cv_info, job_requirements, api_key)
        ranked_cvs.append((cv_info, score, explanation))
        time.sleep(1)  # Añade un retraso si es necesario
    
    # Paso 6: Ordenar y mostrar los mejores CVs
    top_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)[:5]
    print("\nLos 5 CVs más cualificados para el puesto:")
    for idx, (cv_info, score, explanation) in enumerate(top_cvs, start=1):
        print(f"\n#{idx} CV")
        print(f"ID: {cv_info['id']}")
        print(f"Categoría: {cv_info['category']}")
        print(f"Puntuación total: {score}")
        print(f"Explicación de la puntuación: {explanation}")
    
    # Paso 7: Cerrar la conexión a la base de datos
    close_db(conn)

if __name__ == "__main__":
    main()
