import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    """Obtiene la clave de API de OpenAI desde las variables de entorno."""
    return os.getenv("OPENAI_API_KEY")
