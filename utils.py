import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
import openai
import re
import difflib

def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def extract_years_of_experience(experience_str):
    """Extrae el número de años de experiencia de una cadena de texto."""
    # Buscar números en la cadena
    matches = re.findall(r'\d+', str(experience_str))
    if matches:
        # Tomar el primer número encontrado
        years = int(matches[0])
        return years
    else:
        # Si no se encuentra ningún número, devolver 0
        return 0

def extract_job_requirements(job_description, api_key):
    """Utiliza OpenAI para extraer requisitos de un puesto a partir de su descripción."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    
    # Traducir la descripción del puesto al inglés (si no lo está)
    translation_prompt = f"Please translate the following job description to English if it's not already in English:\n'''\n{job_description}\n'''"
    translation_response = llm.invoke([HumanMessage(content=translation_prompt)])
    job_description_en = translation_response.content.strip()
    
    # Mejorar el prompt para manejar descripciones detalladas
    prompt = (
        f"Based on the following detailed job description, please extract:\n"
        f"1. A list of all required skills and competencies (both technical and soft skills).\n"
        f"2. Minimum years of professional experience required.\n"
        f"3. Required education level and specific degrees or fields of study (provide a list).\n"
        f"4. Relevant job categories or fields for this position.\n"
        f"Job Description:\n'''\n{job_description_en}\n'''\n"
        "Please return the information in JSON format with the keys: "
        "'skills', 'minimum_experience', 'education_level', 'degrees', 'categories'. "
        "Ensure that 'skills', 'degrees', and 'categories' are lists, and be as comprehensive as possible."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        requirements = eval(response.content)
        # Extraer y convertir 'minimum_experience'
        experience_str = requirements.get('minimum_experience', '0')
        requirements['minimum_experience'] = extract_years_of_experience(experience_str)
        # Normalizar 'degrees', 'skills' y 'categories'
        requirements['degrees'] = [deg.strip().lower() for deg in requirements.get('degrees', [])]
        requirements['skills'] = [skill.strip().lower() for skill in requirements.get('skills', [])]
        requirements['categories'] = [cat.strip().lower() for cat in requirements.get('categories', [])]
    except Exception as e:
        print(f"Error al extraer los requisitos del puesto: {e}")
        requirements = {
            "skills": [],
            "minimum_experience": 0,
            "education_level": "",
            "degrees": [],
            "categories": []
        }
    return requirements

def extract_candidate_info(cv_text, api_key):
    """Utiliza OpenAI para extraer información relevante de un CV."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    
    prompt = (
        f"Please extract the following information from the candidate's resume:\n"
        f"1. All degrees or educational qualifications obtained (provide a list, including the field of study).\n"
        f"2. Total years of professional experience.\n"
        f"3. List of all skills and competencies (both technical and soft skills).\n"
        f"Resume:\n'''\n{cv_text}\n'''\n"
        "Return the information in JSON format with the keys: 'degrees', 'total_experience', 'skills'. "
        "Ensure that 'degrees' and 'skills' are lists, and be as comprehensive as possible."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        candidate_info = eval(response.content)
        # Extraer y convertir 'total_experience'
        experience_str = candidate_info.get('total_experience', '0')
        candidate_info['total_experience'] = extract_years_of_experience(experience_str)
        # Normalizar 'degrees' y 'skills'
        candidate_info['degrees'] = [deg.strip().lower() for deg in candidate_info.get('degrees', [])]
        candidate_info['skills'] = [skill.strip().lower() for skill in candidate_info.get('skills', [])]
    except Exception as e:
        print(f"Error al extraer la información del candidato: {e}")
        candidate_info = {
            "degrees": [],
            "total_experience": 0,
            "skills": []
        }
    return candidate_info

def evaluate_cv(cv, job_requirements, api_key):
    """Evalúa un CV en función de los requisitos del puesto."""
    score = 0
    explanation = []
    
    # Extraer información del candidato
    candidate_info = extract_candidate_info(cv['resume_str'], api_key)
    
    # Evaluar educación
    required_degrees = job_requirements.get("degrees", [])
    candidate_degrees = candidate_info.get("degrees", [])
    degree_matches = []
    for req_deg in required_degrees:
        for cand_deg in candidate_degrees:
            if similar(req_deg, cand_deg) > 0.8:
                degree_matches.append(cand_deg)
    if degree_matches:
        score += 20
        explanation.append(f"Education matches required degrees: {', '.join(degree_matches)}")
    else:
        explanation.append("Education does not match required degrees")
    
    # Evaluar experiencia
    required_experience = job_requirements.get("minimum_experience", 0)
    candidate_experience = candidate_info.get("total_experience", 0)
    if candidate_experience >= required_experience:
        score += 20
        explanation.append(f"Meets or exceeds minimum experience ({candidate_experience} years)")
    else:
        explanation.append(f"Does not meet minimum experience (has {candidate_experience} years)")
    
    # Evaluar habilidades
    required_skills = job_requirements.get("skills", [])
    candidate_skills = candidate_info.get("skills", [])
    matching_skills = []
    for req_skill in required_skills:
        for cand_skill in candidate_skills:
            if similar(req_skill, cand_skill) > 0.8:
                matching_skills.append(cand_skill)
    skill_score = len(matching_skills) * 10  # Asignar 10 puntos por habilidad coincidente
    score += skill_score
    if matching_skills:
        explanation.append(f"Matching skills: {len(matching_skills)} ({', '.join(set(matching_skills))})")
    else:
        explanation.append("No matching skills found")
    
    # Evaluación por categoría dinámica
    job_categories = job_requirements.get("categories", [])
    cv_category = cv.get('category', '').strip().lower() if isinstance(cv.get('category', ''), str) else ''
    category_match = False
    for job_cat in job_categories:
        if similar(job_cat, cv_category) > 0.8:
            category_match = True
            break
    if category_match:
        score += 10  # Asignar puntos por categoría relevante
        explanation.append(f"Category matches: {cv_category}")
    else:
        explanation.append("Category does not match")
    
    return score, "; ".join(explanation)

def load_dataset(file_path="Resume.csv"):
    """Carga el conjunto de datos desde un archivo CSV."""
    return pd.read_csv(file_path)
