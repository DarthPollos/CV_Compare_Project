import asyncio
import logging
from typing import List, Dict, Any
from utils import generar_respuesta
from send_email import send_email_sync
import json

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def get_candidate_data(query: str, candidates: List[Dict[str, Any]], job_description: str = "") -> str:
    logging.info("🔍 get_candidate_data – recibido query tipo %s: %r", type(query), query)
    if not candidates:
        return "⚠️ No hay candidatos seleccionados aún."
    lineas = []
    # Dentro de la función (por ejemplo, en get_candidate_data o en el bloque de construcción del prompt en rerank)
    for c in candidates:
        id_candidato = str(c.get('ID', 'N/A')).strip()
        nombre = c.get('Nombre', 'Sin Nombre')
        descripcion = c.get('Descripción', 'No disponible')
        correo = c.get('Correo', 'No disponible')
        telefono = c.get('Teléfono', 'No disponible')
    
    # Procesar Idiomas
        idiomas_raw = c.get('Idiomas') or "No disponible"
        if not idiomas_raw.strip():
            idiomas_raw = "No disponible"
        try:
            # Intentamos interpretar como JSON si es posible
            idiomas_dict = json.loads(idiomas_raw) if idiomas_raw != "No disponible" else {}
            idiomas_formateado = ", ".join(f"{idioma} ({nivel})" for idioma, nivel in idiomas_dict.items())
            if not idiomas_formateado:
                idiomas_formateado = "No disponible"
        except Exception:
            idiomas_formateado = idiomas_raw

        # Procesar Habilidades
        habilidades_raw = c.get('Habilidades') or "No disponible"
        if not habilidades_raw.strip():
            habilidades_raw = "No disponible"
        try:
            # Intentamos interpretar como JSON (ejemplo: '{"Python": "Avanzado", "SQL": "Intermedio"}')
            habilidades_dict = json.loads(habilidades_raw) if habilidades_raw != "No disponible" else {}
            habilidades_formateado = ", ".join(f"{habilidad} ({nivel})" for habilidad, nivel in habilidades_dict.items())
            if not habilidades_formateado:
                habilidades_formateado = "No disponible"
        except Exception:
            habilidades_formateado = habilidades_raw

        justificacion = c.get('Justificación', 'No proporcionada')

        linea = (
            f"ID: {id_candidato} | Nombre: {nombre}.\n"
            f"   📜 Descripción: {descripcion}\n"
            f"   🗣️ Idiomas: {idiomas_formateado}\n"  # Información de idiomas
            f"   💡 Habilidades: {habilidades_formateado}\n"  # Información de habilidades
            f"   📧 Correo: {correo}\n"
            f"   📞 Teléfono: {telefono}\n"
            f"   ✅ Justificación: {justificacion}\n"
        )
        lineas.append(linea)


    resumen_candidatos = "\n".join(lineas)
    if not resumen_candidatos:
        return "⚠️ No se encontró información de los candidatos. Revisa si la búsqueda se ejecutó correctamente."
    logging.info("=== Prompt al LLM ===")
    logging.info(resumen_candidatos)
    prompt = f"""
    Responde **únicamente** a la siguiente pregunta, sin mencionar ni tener en cuenta preguntas anteriores. 
    Responde de forma clara y directa.
    Eres un asistente experto en reclutamiento.
    Tienes la siguiente descripción de puesto: {job_description}

    Estos son los candidatos finalistas seleccionados:

    {resumen_candidatos}

    El usuario pregunta:
    "{query}"

    Responde de forma clara y directa. Si la pregunta requiere información sobre un candidato, proporciónala.
    Si la pregunta es general, responde con base en los datos disponibles.
    """
    try:
        respuesta = await generar_respuesta(prompt)
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {e}")
        return "⚠️ Ocurrió un error al procesar la solicitud. Inténtalo de nuevo."
    logging.info("=== Respuesta de la API ===")
    logging.info(respuesta)
    return respuesta

def handle_email_command(candidate_data: dict, command: str) -> str:
    # Verifica que el candidato tenga correo
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

def process_user_input_multiple(query: str, candidates: List[Dict[str, Any]]) -> str:
    logging.info(f"Query recibida: {query}")
    if "envia un correo" in query.lower():
        # Usamos la función parse_email_intent para obtener el asunto y cuerpo
        subject, body_template = parse_email_intent(query)
        result = send_bulk_emails(candidates, subject, body_template)
        return result
    else:
        return asyncio.run(get_candidate_data(query, candidates, "Ejemplo de puesto"))


# Bloque de prueba
if __name__ == "__main__":
    candidates_list = [
        {
            "ID": "123",
            "Nombre": "Juan Pérez",
            "Correo": "juan.perez@example.com",
            "Teléfono": "123456789",
            "Descripción": "Experiencia en desarrollo de software.",
            "Justificación": "Gran experiencia en Python."
        },
        {
            "ID": "456",
            "Nombre": "María López",
            "Correo": "maria.lopez@example.com",
            "Teléfono": "987654321",
            "Descripción": "Especialista en marketing digital.",
            "Justificación": "Buen manejo de redes sociales."
        }
    ]
    user_query = "Envía un correo a todos los candidatos para la entrevista el viernes a las 3pm."
    respuesta = process_user_input_multiple(user_query, candidates_list)
    print("=== Respuesta ===")
    print(respuesta)

