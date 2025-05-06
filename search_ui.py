import os
import requests
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from collections import namedtuple
from database import connect_db, close_db
from utils import (
    build_or_load_vector_index,
    embed_and_search_in_faiss,
    rerank as rerank_con_deepseek,
    buscar_cvs,
    mostrar_resultados_texto
)
import gradio as gr
import asyncio

#Función para mostrar resultados formateados como JSON
async def buscar_cvs_con_distancia(job_description, option_toggle):
    if not job_description:
        return "❌ Por favor, ingresa una descripción del puesto."
    
    resultados = await buscar_cvs(job_description, option_toggle)
    
    #Si el resultado es un string, lo devolvemos directamente
    if isinstance(resultados, str):
        return resultados

    #Si es una lista, la formateamos
    if isinstance(resultados, list):
        if option_toggle == "🔍 Solo RAG":
            resultados_legibles = []
            for match in resultados:
                #Asumimos que ya vienen formateados los datos principales
                resultado = {
                    "Nombre": match.get('Nombre', 'Sin Nombre'),
                    "ID": match.get('ID', 'Desconocido'),
                    "Distancia": match.get('Distancia', 0.0),
                    "Descripción": match.get('Descripción', 'Sin Contenido')
                }
                resultados_legibles.append(resultado)
            return mostrar_resultados_texto(resultados_legibles)
        elif option_toggle == "🤖 RAG + LLM (IA Avanzada)":
            return mostrar_resultados_texto(resultados)
    else:
        return "❌ Error interno: el formato de resultados no es válido."

#Función de envoltura sincrónica para Gradio
def sync_buscar_cvs(job_description, option_toggle):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("⚠️ Advertencia: Ya hay un bucle de eventos en ejecución.")
            # Esperar el resultado en vez de devolver una tarea
            return asyncio.run_coroutine_threadsafe(buscar_cvs_con_distancia(job_description, option_toggle), loop).result()
        else:
            return loop.run_until_complete(buscar_cvs_con_distancia(job_description, option_toggle))
    except RuntimeError as e:
        print(f"🔄 Creando un nuevo bucle de eventos: {str(e)}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(buscar_cvs_con_distancia(job_description, option_toggle))

#Interfaz con Gradio
def search_interface():
    with gr.Blocks(elem_id="search-container") as search_page:
        gr.Markdown("🔎 **Búsqueda de CVs con IA**", elem_id="search-title")

        job_input = gr.Textbox(
            label="Descripción del Puesto", 
            elem_id="job-input",
            placeholder="Ej. Desarrollador Python con experiencia en Machine Learning"
        )
        option_toggle = gr.Radio(
            ["🔍 Solo RAG", "🤖 RAG + LLM (IA Avanzada)"], 
            label="Método de búsqueda", 
            elem_id="search-toggle"
        )
        search_button = gr.Button("Buscar", elem_id="search-button")
        result_output = gr.Textbox(
            label="Resultados", 
            elem_id="result-output", 
            lines=20,  # Número de líneas visibles
            placeholder="Aquí aparecerán los resultados..."
        )  

        #Llamada a la función sincronizada
        search_button.click(fn=sync_buscar_cvs, inputs=[job_input, option_toggle], outputs=result_output)

    return search_page


if __name__ == "__main__":
    interfaz = search_interface()
    interfaz.launch(server_name="0.0.0.0", server_port=7860)
