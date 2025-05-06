import os
import asyncio
import aiosmtplib
from email.message import EmailMessage
import logging
import json
import time
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
from utils import generar_respuesta


# Importar gradio para la interfaz gráfica
import gradio as gr

load_dotenv("key.env", override=True)

# Configuración básica de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables globales
CANDIDATES_FILE = "candidatos.json"
# Diccionario global para almacenar candidatos; clave: ID del candidato (como string)
CANDIDATES_DICT: Dict[str, Dict[str, Any]] = {}

# Carga automática de los candidatos desde candidatos.json
if os.path.exists(CANDIDATES_FILE):
    try:
        with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
            candidatos_lista = json.load(f)
        for candidate in candidatos_lista:
            # Asegurarse de convertir el ID a string y limpiarlo
            candidate_id = str(candidate.get("ID", "")).strip()
            if candidate_id:
                CANDIDATES_DICT[candidate_id] = candidate
        logging.info("Candidatos cargados de %s: %s", CANDIDATES_FILE, list(CANDIDATES_DICT.keys()))
    except Exception as e:
        logging.error("Error al cargar candidatos desde %s: %s", CANDIDATES_FILE, e)
else:
    logging.info("No se encontró %s. Asegúrate de que se guarden los candidatos tras una búsqueda.", CANDIDATES_FILE)



#############################################
# Funciones de envío de correo
#############################################
async def send_email(recipient_email: str, subject: str, body: str) -> bool:
    logging.debug("send_email() iniciado para %s con asunto: %s", recipient_email, subject)
    message = EmailMessage()
    message["From"] = os.getenv("SMTP_FROM_EMAIL")
    if not message["From"]:
        logging.error("La variable SMTP_FROM_EMAIL no está definida.")
        return False
    message["To"] = recipient_email
    message["Subject"] = subject
    message.set_content(body)
    
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port_str = os.getenv("SMTP_PORT", "587")
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        logging.error("El valor de SMTP_PORT (%s) no es válido. Usando 587 por defecto.", smtp_port_str)
        smtp_port = 587
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    if not smtp_host or not smtp_username or not smtp_password:
        logging.error("Faltan variables de entorno para SMTP.")
        return False
    try:
        logging.debug("Llamando a aiosmtplib.send para %s", recipient_email)
        await aiosmtplib.send(
            message,
            hostname=smtp_host,
            port=smtp_port,
            username=smtp_username,
            password=smtp_password,
            start_tls=True
        )
        logging.info("✅ Correo enviado exitosamente a %s", recipient_email)
        return True
    except Exception as e:
        logging.error("❌ Error al enviar correo a %s: %s", recipient_email, e)
        return False

def send_email_sync(recipient_email: str, subject: str, body: str) -> bool:
    logging.debug("send_email_sync() iniciado para %s", recipient_email)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        result = loop.run_until_complete(send_email(recipient_email, subject, body))
        return result
    else:
        return asyncio.run(send_email(recipient_email, subject, body))

#############################################
# Clase para almacenar solicitudes de correo
#############################################
class EmailRequestStore:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EmailRequestStore()
        return cls._instance
    
    def __init__(self):
        self.requests = {}
    
    def add_request(self, request_id, email_data):
        logging.debug("Añadiendo solicitud %s con datos: %s", request_id, email_data)
        self.requests[request_id] = email_data
        
    def update_status(self, request_id, status):
        logging.debug("Actualizando estado de %s a %s", request_id, status)
        if request_id in self.requests:
            self.requests[request_id]["status"] = status

email_store = EmailRequestStore.get_instance()

#############################################
# Integración con HumanLayer para aprobación
#############################################
from humanlayer import HumanLayer  
HUMANLAYER_API_KEY = os.getenv("HUMANLAYER_API_KEY")
if not HUMANLAYER_API_KEY:
    logging.error("La clave HUMANLAYER_API_KEY no está definida en key.env")
    raise ValueError("La clave HUMANLAYER_API_KEY es requerida para continuar.")
hl = HumanLayer.cloud(api_key=HUMANLAYER_API_KEY, verbose=True)
logging.debug("HumanLayer inicializado con API key.")

@hl.require_approval()
def send_email_with_approval(recipient_email: str, subject: str, body: str) -> Dict[str, Any]:
    logging.debug("send_email_with_approval() llamado para %s", recipient_email)
    request_id = str(uuid.uuid4())
    email_data = {
        "recipient_email": recipient_email,
        "subject": subject,
        "body": body,
        "status": "pending",
        "timestamp": time.time()
    }
    email_store.add_request(request_id, email_data)
    success = send_email_sync(recipient_email, subject, body)
    if success:
        email_store.update_status(request_id, "sent")
        logging.info("✅ Correo enviado a %s tras aprobación. ID: %s", recipient_email, request_id)
        return {"success": True, "message": f"Correo enviado a {recipient_email}", "request_id": request_id}
    else:
        email_store.update_status(request_id, "failed")
        logging.error("❌ Error al enviar correo a %s tras aprobación. ID: %s", recipient_email, request_id)
        return {"success": False, "message": f"Error al enviar correo a {recipient_email}", "request_id": request_id}

#############################################
# Funciones para la interfaz Gradio (Vista previa, envío de correo, estado)
#############################################
def parse_email_intent(query: str):
    logging.debug("parse_email_intent() llamado con query: %s", query)
    if "entrevista" in query.lower():
        subject = "Invitación a entrevista"
        body_template = (
            "Hola {name},\n\n"
            "Te invitamos a una entrevista el próximo miércoles a las 10am.\n"
            "Por favor, confirma tu disponibilidad.\n\n"
            "Saludos,\nEquipo de Reclutamiento"
        )
    elif "descartar" in query.lower() or "no vamos a seguir" in query.lower():
        subject = "Notificación de Proceso de Selección"
        body_template = (
            "Hola {name},\n\n"
            "Te agradecemos el tiempo y la dedicación, pero en esta ocasión no continuaremos con el proceso de selección.\n\n"
            "Saludos,\nEquipo de Reclutamiento"
        )
    else:
        subject = "Proceso de Selección - Información Actualizada"
        body_template = (
            "Hola {name},\n\n"
            "Queríamos compartirte información sobre el proceso de selección.\n\n"
            "Saludos,\nEquipo de Reclutamiento"
        )
    return subject, body_template


async def generate_candidate_email(candidate: dict, query: str) -> str:
    """
    Usa inteligencia artificial para generar la redacción de un correo para un candidato,
    basándose en la consulta (query) de reclutamiento y los datos del candidato.
    """
    prompt = f"""
Eres un experto en redacción profesional de correos de reclutamiento.
Redacta un correo dirigido a {candidate["Nombre"]} ({candidate["Correo"]}) para el siguiente propósito: "{query}".
Datos relevantes del candidato:
- Nombre: {candidate["Nombre"]}
- Correo: {candidate["Correo"]}
- Teléfono: {candidate.get("Teléfono", "No disponible")}
- Descripción: {candidate.get("Descripción", "No disponible")}
Asegúrate de que el email sea profesional, claro y adaptado a la solicitud. 
Si el query menciona una fecha específica o solicita feedback, incorpóralo de forma natural.
    """
    email_text = await generar_respuesta(prompt)
    return email_text.strip()


def preview_email(candidate_id: str, query: str) -> str:
    """
    Devuelve una vista previa del correo generado automáticamente basado en el ID
    del candidato y la consulta.
    """
    candidate = CANDIDATES_DICT.get(candidate_id)
    if not candidate:
        logging.error("Candidato no encontrado: %s", candidate_id)
        return "⚠️ No se encontró un candidato con ese ID."

    try:
        generated_email = asyncio.run(generate_candidate_email(candidate, query))
        return generated_email
    except Exception as e:
        logging.error("Error al generar vista previa: %s", e)
        return f"Error: {e}"


def send_email_now(candidate_id: str, query: str) -> str:
    """
    Envía el correo al candidato identificado por su ID utilizando
    inteligencia artificial para generar la redacción del correo.
    """
    candidate = CANDIDATES_DICT.get(candidate_id)
    if not candidate:
        logging.error("Candidato no encontrado: %s", candidate_id)
        return "⚠️ No se encontró un candidato con ese ID."

    try:
        generated_email = asyncio.run(generate_candidate_email(candidate, query))
    except Exception as e:
        logging.error("Error al generar redacción del correo: %s", e)
        return f"Error: {e}"

    subject = "Proceso de Selección - Información Actualizada"
    body = generated_email
    result = send_email_with_approval(candidate["Correo"], subject, body)
    return json.dumps(result, indent=2)



def check_recent_requests() -> str:
    if not email_store.requests:
        return "No hay solicitudes de correo registradas."
    result = "Solicitudes de correo recientes:\n\n"
    for req_id, data in email_store.requests.items():
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))
        result += (
            f"ID: {req_id}\nDestinatario: {data['recipient_email']}\nAsunto: {data['subject']}\n"
            f"Estado: {data['status']}\nFecha: {timestamp}\n{'-'*50}\n"
        )
    return result

#############################################
# Interfaz Gradio para correo (Vista previa, Envío, Estado)
#############################################
iface_preview = gr.Interface(
    fn=preview_email,
    inputs=[
        gr.Dropdown(
            label="Selecciona el candidato",
            choices=list(CANDIDATES_DICT.keys()),
            value=list(CANDIDATES_DICT.keys())[0] if CANDIDATES_DICT else None
        ),
        gr.Textbox(label="Query (ej. 'envia un correo para entrevista')", 
                   value="envia un correo para entrevista", lines=1)
    ],
    outputs="text",
    title="Vista previa del correo"
)

iface_send = gr.Interface(
    fn=send_email_now,
    inputs=[
        gr.Dropdown(
            label="Selecciona el candidato",
            choices=list(CANDIDATES_DICT.keys()),
            value=list(CANDIDATES_DICT.keys())[0] if CANDIDATES_DICT else None
        ),
        gr.Textbox(label="Query (ej. 'envia un correo para entrevista')", 
                   value="envia un correo para entrevista", lines=1)
    ],
    outputs="text",
    title="Enviar correo (con aprobación)"
)

iface_status = gr.Interface(
    fn=check_recent_requests,
    inputs=[],
    outputs="text",
    title="Ver solicitudes recientes"
)

demo = gr.TabbedInterface(
    [iface_preview, iface_send, iface_status],
    ["Vista previa", "Enviar correo", "Solicitudes recientes"]
)

#############################################
# Lanzamiento de la App Gradio
#############################################
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
