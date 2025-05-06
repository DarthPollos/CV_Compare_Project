import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"  # Modelo actualizado

# Configuración de FAISS
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración de la base de datos
DB_CONFIG = {
    "db_name": "cv_database.db",
    "check_same_thread": False
}

def get_db_path():
    return os.path.abspath(DB_CONFIG["db_name"])