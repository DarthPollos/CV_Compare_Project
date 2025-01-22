import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage
import re
import difflib
import numpy as np
import json
import pickle
import time
import openai  # Para capturar RateLimitError
from database import connect_db, close_db

CATEGORY_MAPPING = {
    "software-engineering": "engineering",
    "digital-media": "media",
    "information-technology": "it",
    # Agrega más mapeos si es necesario
}

def similar(a, b):
    """Calcula la similitud entre dos cadenas."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def extract_years_of_experience(experience_str):
    """Extrae el número de años de experiencia de una cadena."""
    matches = re.findall(r'\d+', str(experience_str))
    if matches:
        return int(matches[0])
    return 0

def normalize_category(category):
    """Normaliza una categoría utilizando un mapeo predefinido."""
    if not isinstance(category, str):
        return category
    return CATEGORY_MAPPING.get(category.lower().replace(" ", "-"), category.lower())

def extract_job_requirements(job_description, api_key):
    """
    Extrae los requisitos del puesto a partir de la descripción.
    Sólo 1 intento. Si hay RateLimitError, se devuelven valores por defecto.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    prompt = (
        f"Extract detailed requirements from the following job description:\n"
        f"'''\n{job_description}\n'''\n\n"
        "Return a JSON object with the keys:\n"
        " - skills (list of strings)\n"
        " - minimum_experience (number)\n"
        " - education_level (string)\n"
        " - degrees (list of strings)\n"
        " - categories (list of strings)\n"
        "If not sure, return empty lists/strings.\n"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        requirements = json.loads(response.content)

        requirements['minimum_experience'] = extract_years_of_experience(
            requirements.get('minimum_experience', '0')
        )
        requirements['skills'] = [skill.lower() for skill in requirements.get('skills', [])]
        requirements['degrees'] = [deg.lower() for deg in requirements.get('degrees', [])]
        requirements['categories'] = [normalize_category(cat) for cat in requirements.get('categories', [])]
        return requirements

    except openai.RateLimitError as e:
        print(f"[RateLimitError] {e}. Usando valores predeterminados.")
        return {
            "skills": ["python", "java", "database management"],
            "minimum_experience": 3,
            "education_level": "bachelor's degree",
            "degrees": ["computer science", "software engineering"],
            "categories": ["engineering"]
        }

    except Exception as e:
        print(f"Error al extraer los requisitos del puesto: {e}. Usando valores predeterminados.")
        return {
            "skills": ["python", "java", "database management"],
            "minimum_experience": 3,
            "education_level": "bachelor's degree",
            "degrees": ["computer science", "software engineering"],
            "categories": ["engineering"]
        }


def extract_candidate_info(cv_text, api_key):
    """
    Extrae información del candidato desde el CV, usando gpt-3.5-turbo.
    Sólo 1 intento si hay RateLimitError => se devuelven campos vacíos.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    prompt = (
        f"Extract the candidate's information from this resume:\n"
        f"'''\n{cv_text}\n'''\n\n"
        "Return JSON with keys:\n"
        " - degrees (list of strings)\n"
        " - total_experience (number)\n"
        " - skills (list of strings)\n"
        "If not sure, return empty lists/zero.\n"
    )
    candidate_info = {
        "degrees": [],
        "total_experience": 0,
        "skills": []
    }

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)

        data['total_experience'] = extract_years_of_experience(data.get('total_experience', '0'))
        data['skills'] = [skill.lower() for skill in data.get('skills', [])]
        data['degrees'] = [deg.lower() for deg in data.get('degrees', [])]
        candidate_info = data

    except openai.RateLimitError as e:
        print(f"[RateLimitError extract_candidate_info] {e} → devolviendo info vacía.")
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        print(f"Error al extraer la información del candidato: {e}")

    return candidate_info


EMBEDDING_CACHE = {}

def compute_semantic_similarity(text1, text2, api_key):
    """
    Calcula la similitud semántica entre dos textos utilizando caché.
    Sólo 1 intento. Si hay RateLimitError => retorna 0.0.
    """
    global EMBEDDING_CACHE
    max_length = 1000  # Limitar el tamaño de los textos a 1000 caracteres
    text1 = text1[:max_length]
    text2 = text2[:max_length]

    # Generar claves únicas para los textos
    key1 = hash(text1)
    key2 = hash(text2)
    cache_key = (key1, key2)

    # Verificar caché
    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    try:
        emb1 = np.array(embeddings.embed_query(text1))
        emb2 = np.array(embeddings.embed_query(text2))
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        EMBEDDING_CACHE[cache_key] = similarity

        # (Opcional) Guardar en disco
        with open("embedding_cache.pkl", "wb") as f:
            pickle.dump(EMBEDDING_CACHE, f)

        return similarity

    except openai.RateLimitError as e:
        print(f"[RateLimitError compute_semantic_similarity] {e} → Retornando 0.0")
        return 0.0
    except Exception as e:
        print(f"Error al calcular embeddings: {e}")
        return 0.0

def evaluate_cv(cv, job_requirements, api_key):
    """Evalúa un CV basado en los requisitos del puesto."""
    resume_str = cv.get('resume_str', '')
    candidate_info = extract_candidate_info(resume_str, api_key)

    # Añade un retraso de 1s (o menos si quieres) para no saturar
    time.sleep(1)

    hard_score = 0
    explanation_parts = []

    # 1. Education
    required_degrees = job_requirements.get("degrees", [])
    candidate_degrees = candidate_info.get("degrees", [])
    if any(similar(cd, rd) > 0.8 for cd in candidate_degrees for rd in required_degrees):
        hard_score += 20
        explanation_parts.append("Coinciden títulos requeridos (+20).")

    # 2. Experience
    if candidate_info.get("total_experience", 0) >= job_requirements.get("minimum_experience", 0):
        hard_score += 20
        explanation_parts.append("Años de experiencia cumplen o superan el mínimo (+20).")

    # 3. Skills
    required_skills = job_requirements.get("skills", [])
    candidate_skills = candidate_info.get("skills", [])
    skill_points = sum(5 for skill in candidate_skills if any(similar(skill, rs) > 0.8 for rs in required_skills))
    hard_score += skill_points
    explanation_parts.append(f"Skills relevantes: +{skill_points} puntos.")

    # 4. Soft score (similitud)
    #   Compara " ".join(required_skills) con el CV
    soft_score = compute_semantic_similarity(resume_str, " ".join(required_skills), api_key) * 30
    explanation_parts.append(f"Similitud semántica => +{soft_score:.2f} (máx 30).")

    # 5. Cálculo final
    total_score = 0.7 * hard_score + 0.3 * soft_score
    explanation_parts.append(f"Puntuación final: {total_score:.2f} (Hard: {hard_score}, Soft: {soft_score:.2f}).")

    return total_score, explanation_parts

def load_dataset(file_path="Resume.csv"):
    """Carga un dataset desde un archivo CSV."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return pd.DataFrame()

def reset_database():
    """Resetea la base de datos."""
    conn, cursor = connect_db()
    try:
        cursor.execute("DROP TABLE IF EXISTS cv")
        cursor.execute(
            """
            CREATE TABLE cv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_str TEXT NOT NULL,
                category TEXT
            )
            """
        )
        conn.commit()
        print("Base de datos restablecida.")
    except Exception as e:
        print(f"Error al restablecer la base de datos: {e}")
    finally:
        close_db(conn)

def insert_cvs_from_dataset(cursor, dataset):
    """Inserta los CVs desde un dataset en la base de datos."""
    for _, row in dataset.iterrows():
        resume_str = row.get('resume', '')
        category = str(row.get('category', '')).lower()
        cursor.execute(
            "INSERT INTO cv (resume_str, category) VALUES (?, ?)", (resume_str, category)
        )

def filter_cvs_by_category(cursor, categories):
    """
    Filtra los CVs por categorías relevantes en la base de datos.
    Si 'categories' está vacío, retorna todos.
    """
    if not categories:
        print("No se especificaron categorías. Se devuelven todos los CVs.")
        cursor.execute("SELECT * FROM cv")
        return cursor.fetchall()

    query = "SELECT * FROM cv WHERE LOWER(category) IN ({})".format(
        ",".join(["?"] * len(categories))
    )
    normalized_cats = [cat.lower().replace(" ", "-") for cat in categories]
    cursor.execute(query, normalized_cats)
    return cursor.fetchall()

def filter_cvs_by_semantic_relevance(cvs, job_description, threshold, api_key):
    """
    Filtra los CVs por relevancia semántica con respecto a la descripción del puesto.
    Sólo 1 intento y 0.0 de similitud si surge RateLimitError.
    """
    relevant_cvs = []
    max_cvs_to_process = min(10, len(cvs))  # Procesa como máximo 10 CVs
    print(f"Procesando hasta {max_cvs_to_process} CVs para similitud semántica.")

    for idx, cv in enumerate(cvs[:max_cvs_to_process]):
        cv_id = cv[0]
        resume_str = cv[1]
        similarity = compute_semantic_similarity(job_description, resume_str, api_key)
        print(f"Similitud para CV ID {cv_id}: {similarity:.2f}")
        if similarity >= threshold:
            relevant_cvs.append(cv)

    print(f"Filtrados {len(relevant_cvs)} CVs relevantes por similitud >= {threshold}.")
    return relevant_cvs
