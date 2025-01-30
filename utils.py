import os
import re
import json
import pandas as pd
import requests
from collections import namedtuple
from requests.exceptions import RequestException

from langchain_huggingface import HuggingFaceEmbeddings  # Importación actualizada
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def load_dataset_into_db(cursor, csv_path):
    """Carga los datos del CSV en la base de datos."""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        resume_str = str(row.get("resume", ""))
        category = str(row.get("category", "")).lower()
        cursor.execute("INSERT INTO cv (resume_str, category) VALUES (?, ?)", (resume_str, category))

def build_or_load_vector_index(cursor, rebuild=False):
    """Construye o carga un índice FAISS."""
    index_path = "faiss_index"

    if not rebuild and os.path.exists(index_path):
        print("Cargando índice FAISS existente desde disco...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docsearch = FAISS.load_local(
            folder_path=index_path,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        return docsearch

    print("Creando un nuevo índice FAISS con todos los CVs...")
    cursor.execute("SELECT id, resume_str, category FROM cv")
    rows = cursor.fetchall()
    if not rows:
        print("No hay CVs en la tabla. No se construye el índice.")
        return None

    documents = [
        Document(page_content=row[1], metadata={"id": row[0], "category": row[2]})
        for row in rows
    ]

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(documents, embedding)
    os.makedirs(index_path, exist_ok=True)
    docsearch.save_local(index_path)
    print(f"Índice guardado en carpeta: {index_path}/")
    return docsearch

def embed_and_search_in_faiss(query_text, docsearch, top_k=5):
    """Busca los documentos más similares en FAISS."""
    if not docsearch:
        return []
    results = docsearch.similarity_search_with_score(query_text, k=top_k)
    Match = namedtuple("Match", ["page_content", "metadata", "score"])
    return [Match(doc.page_content, doc.metadata, dist) for doc, dist in results]

def rerank_with_deepseek(top_docs, job_description):
    """
    Mejora en la integración con DeepSeek y mejor manejo de errores.
    """
    top_docs = top_docs[:5]  # Limitar a 5 CVs para evitar prompts largos

    # Construir información de CVs
    docs_info = []
    for i, doc in enumerate(top_docs, start=1):
        doc_id = doc.metadata.get("id")
        cat = doc.metadata.get("category", "")
        text_fragment = doc.page_content[:100].replace("\n", " ")
        docs_info.append(f"CV{i} (ID: {doc_id}, Category: {cat}): {text_fragment}...")

    # Construir prompt estructurado
    full_prompt = f"""
Eres un experto en recursos humanos con 10 años de experiencia. Analiza los siguientes CVs comparándolos con la descripción del puesto y genera un ranking en formato JSON.

Requisitos del puesto:
{job_description}

CVs a evaluar:
{chr(10).join(docs_info)}

Instrucciones:
1. Asigna un score entre 1-100 según adecuación al puesto
2. Proporciona 2-3 razones concretas por CV
3. Ordena de mayor a menor score
4. Devuelve SOLO el JSON sin comentarios

Formato requerido:
[
  {{
    "id": "ID_CV",
    "score": 0-100,
    "reasons": ["razón 1", "razón 2"]
  }},
  ...
]
"""

    print("\n🚀 Enviando consulta a DeepSeek...")
    print(f"📄 Longitud del prompt: {len(full_prompt)} caracteres")
    print(f"🔍 Vista previa del prompt:\n{full_prompt[:500]}...")

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'deepseek-llm:7b',
                'prompt': full_prompt,
                'format': 'json',
                'stream': False,
                'options': {
                    'temperature': 0.5,
                    'max_tokens': 2000,
                    'top_p': 0.9
                }
            },
            timeout=180  # Tiempo máximo de espera
        )

        # Manejo de errores HTTP
        if response.status_code != 200:
            print(f"⚠️ Error en la API (Código {response.status_code}): {response.text}")
            return []

        json_response = response.json()
        stdout = json_response.get('response', '')
        print("\n🔍 Respuesta recibida:", stdout[:200] + "...")

    except RequestException as e:
        print(f"\n🚨 Error de conexión: {str(e)}")
        print("Verifica que Ollama esté corriendo: 'ollama serve'")
        return []
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        return []

    # Procesamiento del JSON
    try:
        match = re.search(r'\[\s*{.*?}\s*\]', stdout, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
        else:
            raise ValueError("No se encontró JSON válido en la respuesta de DeepSeek")

        gpt_data = json.loads(json_str)

        if not isinstance(gpt_data, list):
            raise ValueError("El formato de respuesta no es una lista")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error procesando JSON: {str(e)}")
        print("Respuesta cruda:", stdout)
        return []

    # Construcción del resultado final
    final_output = []
    doc_map = {
        str(doc.metadata.get("id")): (doc.metadata.get("category", ""), doc.page_content[:100].replace("\n", " "))
        for doc in top_docs
    }

    for item in gpt_data:
        try:
            cv_id = str(item.get("id"))
            if cv_id not in doc_map:
                continue
                
            cat, fragment = doc_map[cv_id]
            final_output.append({
                "id": cv_id,
                "category": cat,
                "score": float(item.get("score", 0)),
                "reasons": "\n".join(item.get("reasons", [])),
                "fragment": fragment
            })
        except Exception as e:
            print(f"⚠️ Error procesando item {item}: {str(e)}")
            continue

    # Ordenar y limitar resultados
    final_output.sort(key=lambda x: x["score"], reverse=True)
    return final_output[:5]  # Devolver máximo 5 resultados
