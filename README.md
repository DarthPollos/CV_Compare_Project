# CV_Compare_Project

Este proyecto tiene como objetivo comparar currículums (CVs) en una base de datos y encontrar los más relevantes para una descripción de puesto proporcionada por el usuario. Utiliza técnicas avanzadas como embeddings y el motor de búsqueda FAISS para medir la similitud entre los textos.

¿Qué hace este proyecto?

Construcción del índice FAISS:
Convierte los CVs almacenados en una base de datos en vectores de texto (usando HuggingFaceEmbeddings).
Crea un índice FAISS para hacer búsquedas rápidas y eficientes.
Búsqueda de CVs relevantes:
Acepta una descripción de puesto ingresada por el usuario.
Encuentra los CVs más relevantes basados en la similitud semántica.
Nueva funcionalidad: RAG:
La rama rag_integration implementa una técnica mejorada llamada Retrieval-Augmented Generation (RAG), que combina recuperación de datos y generación de texto.
Estructura del proyecto
main.py:
Archivo principal para interactuar con el proyecto.
Permite crear el índice (build_index) o realizar consultas (query).
utils.py:
Contiene funciones auxiliares para manejar la base de datos, construir/cargar el índice FAISS y realizar búsquedas.
Base de datos:
Almacena los CVs en una tabla con las columnas: id, resume_str (texto del CV) y category (categoría del CV).
Requisitos
Antes de usar el proyecto, asegúrate de tener instalados los siguientes paquetes:

Python 3.8+
FAISS: Para construir y gestionar el índice de búsqueda.
LangChain Community: Para embeddings y manejo de índices.
Pandas: Para manipular datos del CSV.
SQLite3: Base de datos para almacenar los CVs.

Instala las dependencias con:
pip install langchain-community pandas faiss-cpu


Estructura de la base de datos
La tabla cv debe contener las siguientes columnas:

id: Identificador único del CV.
resume_str: Texto del CV (el contenido completo).
category: Categoría o sector del CV (ej. "Diseño", "Tecnología").


Para usar el proyecto:

Crear el índice FAISS:

Asegúrate de que la base de datos esté configurada y contenga CVs en la tabla cv.
Ejecuta el siguiente comando:
python main.py build_index
Esto generará un índice en la carpeta faiss_index/.

Buscar CVs relevantes:

Ingresa la descripción del puesto cuando se te solicite, después de este comando:
python main.py query
[Solicita descripción]







