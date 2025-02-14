import os
from dotenv import load_dotenv

# Cargar variables desde key.env
dotenv_path = os.path.join(os.path.dirname(__file__), "key.env")
load_dotenv(dotenv_path)

# Obtener API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ No se encontró la clave OPENAI_API_KEY en key.env")
