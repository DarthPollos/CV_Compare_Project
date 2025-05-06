import gradio as gr
import asyncio
import json
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from chat_agent import get_candidate_data  # Función asíncrona que genera la respuesta del agente
from send_email import send_email_sync

# Variables globales para almacenar los candidatos y la descripción del puesto
JOB_DESCRIPTION = ""
CANDIDATES_FILE = "candidatos.json"

# Función para manejar la consulta del usuario y devolver la respuesta del agente
async def handle_user_query(query: str) -> str:
    logging.info("🔍 handle_user_query – query type: %s | content: %r", type(query), query)
    global JOB_DESCRIPTION

    text = query.strip().lower()

    # 1) Saludos / small‑talk
    saludos = {"hola", "buenos días", "buenas tardes", "buenas noches"}
    if text in saludos or "¿cómo estás" in text or "como estas" in text:
        return "¡Hola! Estoy aquí para ayudarte con preguntas **sobre los candidatos** finalistas. 😊"

    # 2) Carga y validación de candidatos
    try:
        with open(CANDIDATES_FILE, "r") as f:
            candidatos = json.load(f)
        if not isinstance(candidatos, list) or not candidatos:
            return "⚠️ No hay candidatos guardados o el archivo está vacío."
        candidatos_validos = [
            c for c in candidatos 
            if isinstance(c, dict) and "ID" in c and "Nombre" in c and "Descripción" in c
        ]
        if not candidatos_validos:
            return "⚠️ No se encontraron candidatos válidos."
    except (FileNotFoundError, json.JSONDecodeError):
        return "⚠️ Error: No se encontraron candidatos guardados."

    # 3) Si es comando de correo (aún lo tienes, opcional quitarlo)
    if "envia un correo" in text:
        return process_user_input(query, candidatos_validos[0])

    # 4) Detección de consulta relevante sobre candidatos
    palabras_clave = (
        "candidato", "candidatos",
        "idioma", "idiomas",
        "experiencia",
        "habilidad", "habilidades",
        "quién", "quien",
        "descripción", "descripci"
    )
    if any(p in text for p in palabras_clave):
        return await get_candidate_data(query, candidatos_validos, JOB_DESCRIPTION)

    # 5) Respuesta por defecto si no es small‑talk ni consulta de candidatos
    return (
        "Lo siento, solo puedo responder consultas **sobre los candidatos** finalistas. "
        "Por ejemplo: “¿Quién habla inglés?” o “¿Qué habilidades tiene María López?”."
    )


def handle_email_command(candidate_data: dict) -> str:
    # Se elimina el parámetro "command" ya que no se utiliza
    if "Correo" not in candidate_data or candidate_data["Correo"] == "No disponible":
        logging.warning("El candidato no tiene un correo registrado.")
        return "El candidato no tiene un correo registrado."
    recipient_email = candidate_data["Correo"]
    subject = "Proceso de Selección - Información Actualizada"
    body = (
        f"Hola {candidate_data.get('Nombre', 'Candidato')},\n\n"
        "Te informamos que has sido seleccionado para avanzar al siguiente nivel del proceso de selección. "
        "Por favor, confirma tus datos de contacto y tu disponibilidad para la siguiente fase.\n\n"
        "Saludos cordiales,\nEquipo de Reclutamiento"
    )
    send_email_sync(recipient_email, subject, body)
    return f"Correo enviado a {recipient_email}"

# Wrapper sincrónico para que Gradio llame a la función asíncrona
def sync_handle_user_query(query):
    return asyncio.run(handle_user_query(query))

# Función auxiliar para aplanar la respuesta si es una tupla
def flatten_response(resp):
    """
    Aplana la respuesta para asegurarse de que si se retorna un tuple se extraiga el primer elemento.
    """
    while isinstance(resp, tuple):
        resp = resp[0]
    return resp


# Función auxiliar para normalizar el historial (asegurar que cada elemento sea un diccionario con "content")
def normalize_history(history):
    normalized = []
    for item in history:
        if isinstance(item, dict) and "content" in item:
            normalized.append(item)
        else:
            normalized.append({"role": "assistant", "content": str(item)})
    return normalized

# Nueva función de chat que usaremos en gr.ChatInterface.
# Recibe el mensaje del usuario y el historial, y devuelve la respuesta del agente junto al historial actualizado.
def chat_fn(message, history):
    logging.info("🔍 chat_fn – message type: %s | content: %r", type(message), message)
    if history is None:
        history = []
    else:
        history = normalize_history(history)
    
    logging.info("Mensaje recibido: %s", message)
    
    # Obtenemos y aplanamos la respuesta del agente
    response = asyncio.run(handle_user_query(message))
    response = flatten_response(response)
    # Forzamos que sea un string
    if not isinstance(response, str):
        response = str(response)
    logging.info("Respuesta final: %s", response)
    
    # Actualizamos el historial de mensajes
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    logging.info("Historial actualizado: %s", history)
    return history[-1:]  # Devuelve solo el historial actualizado


# Función para generar la interfaz de chat usando gr.ChatInterface
def chat_interface():
    chat_ui = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        title="Agente de Reclutamiento",
        description="Interactúa con el agente de reclutamiento para obtener información sobre candidatos."
    )
    return chat_ui

if __name__ == "__main__":
    chat_ui = chat_interface()
    chat_ui.launch(server_name="0.0.0.0", server_port=7862)
