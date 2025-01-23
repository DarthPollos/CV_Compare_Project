import os
import pandas as pd

# Usamos `langchain_community` porque las clases originales de FAISS y embeddings
# fueron movidas desde `langchain`.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document  # Clase para estructurar documentos.

def load_dataset_into_db(cursor, csv_path):
    """
    Función para cargar un archivo CSV en la tabla `cv` de la base de datos.
    Asegúrate de que la tabla tenga las columnas: `id`, `resume_str` y `category`.
    """
    df = pd.read_csv(csv_path)  # Leemos el archivo CSV.
    for _, row in df.iterrows():  # Iteramos sobre cada fila del CSV.
        resume_str = str(row.get("resume", ""))  # Obtenemos el texto del CV.
        category = str(row.get("category", "")).lower()  # Obtenemos la categoría en minúsculas.
        # Insertamos los datos en la base de datos.
        cursor.execute("INSERT INTO cv (resume_str, category) VALUES (?, ?)", (resume_str, category))

def build_or_load_vector_index(cursor, rebuild=False):
    """
    Esta función construye un índice FAISS desde los CVs en la base de datos o lo carga desde el disco.

    - `rebuild=False`: Carga el índice si ya existe.
    - `rebuild=True`: Construye un nuevo índice desde los datos en la base de datos.
    """
    index_path = "faiss_index"  # Carpeta donde se guarda el índice FAISS.

    # Si rebuild es False y el índice ya existe, lo cargamos desde el disco.
    if (not rebuild) and os.path.exists(index_path):
        print("Cargando índice FAISS existente desde disco...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Cargamos el índice FAISS.
        docsearch = FAISS.load_local(
            folder_path=index_path,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        return docsearch

    print("Creando un nuevo índice FAISS con todos los CVs...")

    # Obtenemos todos los CVs desde la tabla `cv` de la base de datos.
    cursor.execute("SELECT id, resume_str, category FROM cv")
    rows = cursor.fetchall()
    if not rows:  # Si no hay datos, mostramos un mensaje.
        print("No hay CVs en la tabla. No se construye el índice.")
        return None

    # Creamos una lista de documentos para el índice.
    documents = []
    for r in rows:
        doc_id = r[0]  # ID del CV.
        text = r[1]  # Contenido del CV.
        cat = r[2]  # Categoría del CV.
        # Creamos un documento con el contenido y la metadata.
        documents.append(
            Document(
                page_content=text,
                metadata={"id": doc_id, "category": cat}
            )
        )

    # Generamos embeddings para los documentos y construimos el índice FAISS.
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(documents, embedding)

    # Guardamos el índice en la carpeta especificada.
    os.makedirs(index_path, exist_ok=True)
    docsearch.save_local(index_path)
    print(f"Índice guardado en carpeta: {index_path}/")
    return docsearch

def embed_and_search_in_faiss(query_text, docsearch, top_k=5):
    """
    Esta función busca en el índice FAISS los documentos más similares a un texto de consulta.

    - `query_text`: Texto ingresado por el usuario.
    - `docsearch`: El índice FAISS cargado.
    - `top_k`: Número de resultados más cercanos que queremos obtener.
    """
    if not docsearch:  # Si el índice no está cargado, devolvemos una lista vacía.
        return []

    # Realizamos la búsqueda en el índice y obtenemos los resultados con sus distancias.
    results = docsearch.similarity_search_with_score(query_text, k=top_k)

    # Creamos una lista con los resultados estructurados.
    from collections import namedtuple
    Match = namedtuple("Match", ["page_content", "metadata", "score"])
    matches = []
    for (doc, dist) in results:
        matches.append(Match(doc.page_content, doc.metadata, dist))  # Guardamos el resultado.
    return matches
