import gradio as gr
import asyncio
import json
from search_ui import buscar_cvs
from interface_chat import chat_interface
from send_email import preview_email, send_email_now


CANDIDATES_FILE = "candidatos.json"
CANDIDATES_DICT = {}  # Diccionario global para almacenar los datos completos de los candidatos

async def iniciar_busqueda(descripcion_puesto, option_toggle):
    """
    Función que ejecuta la búsqueda de CVs cuando el usuario lo inicie desde Gradio.
    """
    global JOB_DESCRIPTION, CANDIDATES_DICT
    JOB_DESCRIPTION = descripcion_puesto  # Guardamos la descripción del puesto

    print("🔍 Buscando y rankeando candidatos...")
    candidatos_seleccionados = await buscar_cvs(descripcion_puesto, option_toggle)

    # Guardamos los datos completos con IDs como strings
    CANDIDATES_DICT = {str(c["ID"]).strip(): c for c in candidatos_seleccionados}

    print("\n=== Candidatos completos guardados en memoria (CANDIDATES_DICT) ===")
    print(json.dumps(CANDIDATES_DICT, indent=2, ensure_ascii=False))

    # Filtrar y mostrar solo datos necesarios para el ranking,
    # pero incluyendo campos de contacto para que el agente pueda usarlos
    candidatos_filtrados = [
        {
            "ID": str(c["ID"]).strip(),
            "Nombre": c["Nombre"],
            "Descripción": c["Descripción"],
            "Justificación": c.get("Justificación", "No proporcionada"),
            "Correo": c.get("Correo", "No disponible"),
            "Teléfono": c.get("Teléfono", "No disponible"),
            "Idiomas": c.get("Idiomas", "No disponible"),
            "Habilidades": c.get("Habilidades", "No disponible")
        }
        for c in candidatos_seleccionados
    ]

    if not candidatos_filtrados:
        print("❌ Error: No se encontraron candidatos válidos después del filtrado.")
        return "❌ No se encontraron candidatos válidos. Intenta nuevamente."

    # Guardamos los candidatos completos en JSON para que el agente tenga acceso a toda la información
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(candidatos_seleccionados, f, indent=2, ensure_ascii=False)

    print("✅ Candidatos guardados en 'candidatos.json'")

    # Devolvemos una versión en texto estructurado para Gradio, mostrando también correo y teléfono
    resultado_legible = "🔝 Ranking de Candidatos:\n\n"
    for i, candidato in enumerate(candidatos_filtrados, start=1):
        resultado_legible += (
            f"{i}. {candidato['Nombre']}\n"
            f"   - 🆔 ID: {candidato['ID']}\n"
            f"   - 📜 Descripción: {candidato['Descripción']}\n"
            f"   - ✅ Justificación: {candidato['Justificación']}\n"
            f"   - 📧 Correo: {candidato['Correo']}\n"
            f"   - 📞 Teléfono: {candidato['Teléfono']}\n\n"
        )

    return resultado_legible

def sync_iniciar_busqueda(descripcion_puesto, option_toggle) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Ejecuta la búsqueda y recibe solo el texto preparado para Gradio
        resultado = loop.run_until_complete(iniciar_busqueda(descripcion_puesto, option_toggle))
        return resultado
    except Exception as e:
        print(f"❌ Error en sync_iniciar_busqueda: {e}")
        return f"❌ Error al procesar la búsqueda: {str(e)}"
    finally:
        loop.close()


def principal_interface():
    """
    Interfaz principal que permite realizar la búsqueda y acceder al agente de reclutamiento.
    """
    with gr.Blocks() as ui:
       
        gr.Markdown("## 📑 Búsqueda de Candidatos")

        with gr.Tabs():
            with gr.Tab("🔍 Buscar Candidatos"):
                gr.Markdown("### 🏆 Encuentra a los mejores candidatos con IA")
                descripcion_puesto = gr.Textbox(
                    label="Descripción del puesto",
                    placeholder="Ejemplo: Desarrollador Python con experiencia en IA"
                )
                option_toggle = gr.Radio(
                    choices=["🔍 Solo RAG", "🤖 RAG + LLM (IA Avanzada)"],
                    label="Modo de búsqueda",
                    value="🤖 RAG + LLM (IA Avanzada)"
                )
                search_button = gr.Button("🔎 Iniciar Búsqueda")
                resultado_output = gr.Textbox(label="Candidatos Encontrados", lines=10)

                search_button.click(
                    fn=sync_iniciar_busqueda,
                    inputs=[descripcion_puesto, option_toggle],
                    outputs=[resultado_output]
                )


            with gr.Tab("🤖 Agente de Reclutamiento"):
                gr.Markdown("### 🗨️ Interactúa con el Agente sobre los candidatos")
                chat_ui = chat_interface()

            with gr.Tab("✉️ Enviar Correo"):
                gr.Markdown("### 📧 Automatización del envío de correos a candidatos")
                # Usamos un Textbox para introducir el nombre del candidato manualmente
                candidate_input = gr.Textbox(
                    label="ID del candidato",
                    placeholder="Ejemplo: 123"
                )
                query_email = gr.Textbox(
                    label="Drescribe el correo que quieres enviar", 
                    value="envia un correo para entrevista", 
                    lines=1
                )
                email_preview = gr.Textbox(label="Vista previa del correo", lines=10)
                preview_btn = gr.Button("Mostrar vista previa")
                send_btn = gr.Button("Enviar correo ahora")
                
                preview_btn.click(fn=preview_email, inputs=[candidate_input, query_email], outputs=[email_preview])
                send_btn.click(fn=send_email_now, inputs=[candidate_input, query_email], outputs=[email_preview])


    return ui

if __name__ == "__main__":
    ui = principal_interface()
    ui.launch(server_name="0.0.0.0", server_port=7861)
