import sys
import os
import time

# Importamos las funciones necesarias para conectar y cerrar la base de datos.
from database import connect_db, close_db

# Importamos las funciones principales del archivo utils.py que vamos a usar.
from utils import (
    load_dataset_into_db,  # Para cargar datos desde un CSV a la base de datos.
    build_or_load_vector_index,  # Para construir o cargar el índice FAISS.
    embed_and_search_in_faiss  # Para buscar en el índice FAISS usando embeddings.
)

def main():
    """
    Este programa tiene dos modos de uso principales:
    1. `build_index`: Genera un índice FAISS a partir de los CVs guardados en la base de datos.
    2. `query`: Permite buscar en el índice los CVs más similares a una descripción de puesto ingresada.
    
    Ejemplo de uso:
        python main.py build_index  # Para crear el índice.
        python main.py query        # Para realizar consultas en el índice.

    Las pruebas las haremos con la siguiente descripción de puesto:

        - "Desarrollador de software con experiencia en Python y Machine Learning."

        - "This is a job description for a Software Engineer. The role requires Python, 
            Java, and database management skills. A minimum of 3 years of experience is needed."

        - "Buscamos un diseñador de interiores creativo con experiencia en proyectos residenciales y comerciales. 
            Será responsable de diseñar espacios funcionales y estéticos, seleccionar materiales, mobiliario y coordinar con
            proveedores y contratistas. Se requiere manejo avanzado de AutoCAD, SketchUp, Photoshop y software de renderizado 
            (V-Ray o similar), además de habilidades en gestión de proyectos, comunicación con clientes y atención al detalle."
    """
    if len(sys.argv) < 2:  # Verificamos si el usuario ingresó el modo como argumento.
        print("Modo de uso: python main.py [build_index | query]")
        return

    mode = sys.argv[1].lower()  # Obtenemos el modo del argumento ingresado.

    # Conectamos a la base de datos para realizar operaciones.
    conn, cursor = connect_db()

    if mode == "build_index":
        # Este modo crea un índice FAISS a partir de los datos en la base de datos.
        # Si es necesario, también podemos cargar un CSV antes de construir el índice.
        # load_dataset_into_db(cursor, "Resume.csv")  # Descomentar si queremos cargar un archivo CSV.
        # conn.commit()  # Guardamos los cambios en la base de datos.

        # Creamos el índice FAISS desde los datos de la tabla `cv`.
        build_or_load_vector_index(cursor, rebuild=True)
        print("Índice FAISS construido y guardado con éxito.")

    elif mode == "query":
        # En este modo, el programa permite al usuario buscar CVs similares.
        print("Por favor, ingresa la descripción del puesto. Cuando termines, presiona Enter dos veces:")
        lines = []
        while True:
            line = input()
            if line.strip() == "":  # Si la línea está vacía, terminamos de capturar el texto.
                break
            lines.append(line)

        # Unimos las líneas ingresadas en un solo texto.
        job_description = "\n".join(lines).strip()

        if not job_description:  # Si no se ingresó nada, mostramos un mensaje y salimos.
            print("No se ingresó ninguna descripción.")
            close_db(conn)
            return

        # Intentamos cargar el índice FAISS desde el disco.
        docsearch = build_or_load_vector_index(cursor, rebuild=False)
        if not docsearch:  # Si el índice no existe, mostramos un error.
            print("No se pudo cargar ni crear el índice FAISS.")
            close_db(conn)
            return

        # Realizamos la búsqueda en el índice.
        top_matches = embed_and_search_in_faiss(job_description, docsearch, top_k=5)

        # Mostramos los resultados de la búsqueda.
        print(f"\nSe han recuperado {len(top_matches)} CVs más similares:\n")
        for i, match in enumerate(top_matches, start=1):
            metadata = match.metadata  # Metadata del CV.
            doc_text = match.page_content  # Contenido del CV.
            score = match.score  # Puntaje de similitud (distancia).
            cv_id = metadata.get("id")  # ID del CV.
            cv_cat = metadata.get("category", "")  # Categoría del CV.
            print(f"==> {i}) CV ID: {cv_id}, Categoría: {cv_cat}, Distancia: {score:.4f}")
            print(f"    Fragmento del CV:\n{doc_text[:200]}...\n")  # Mostramos un fragmento del CV.

    else:
        # Si el modo ingresado no es válido, mostramos un mensaje de error.
        print("Opción desconocida. Usa: build_index ó query")

    # Cerramos la conexión con la base de datos al final.
    close_db(conn)

# Ejecutamos el programa llamando a la función main.
if __name__ == "__main__":
    main()
