import os
import asyncio
import aiohttp
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from collections import namedtuple
from database import connect_db, close_db
from dotenv import load_dotenv
load_dotenv("key.env", override=True)
import re

# Configuración global
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
#OPENAI_MODEL = "gpt-4"


# =============================================================================
# Función para verificar la base de datos.
# =============================================================================
def verificar_base_datos():
    db_path = os.path.abspath('cv_database.db')
    existe = os.path.exists(db_path)
    print(f"=== Verificación de Base de Datos ===")
    print(f"Ruta: {db_path}")
    print(f"¿Existe?: {existe}")
    print("======================================")
    return existe

# =============================================================================
# Función para construir o cargar el índice FAISS.
# =============================================================================
def build_or_load_vector_index(conn, cursor, rebuild=True, batch_size=10):
    if rebuild or not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        print("🔨 Creando nuevo índice FAISS...")
        # Ajustamos la consulta para incluir la columna 'habilidades'
        cursor.execute("""
            SELECT id, nombre, COALESCE(resumen, ''), email, telefono,
            COALESCE(idiomas, ''), COALESCE(habilidades, ''),
            COALESCE(experiencia, ''), COALESCE(ubicacion, ''), COALESCE(educacion, '')
            FROM cv
        """)


        documentos = []
        rows = cursor.fetchall()
        for cv_id, nombre, resumen, email, telefono, idiomas, habilidades, experiencia, ubicacion, educacion in rows:
            if resumen.strip() == "":
                print(f"⚠️ El resumen está vacío para el CV de {nombre} (ID: {cv_id})")
            
            page_content = f"""
                RESUMEN: {resumen.strip()}
                IDIOMAS: {idiomas.strip()}
                HABILIDADES: {habilidades.strip()}
                EXPERIENCIA: {experiencia.strip()}
                UBICACIÓN: {ubicacion.strip()}
                EDUCACIÓN: {educacion.strip()}
                """
            metadata = {
                "id": cv_id,
                "name": nombre.strip(),
                "email": email.strip() if email else "",
                "telefono": telefono.strip() if telefono else "",
                "idiomas": idiomas.strip() if idiomas else "No disponible",
                "habilidades": habilidades.strip() if habilidades else "No disponible",
                "experiencia": experiencia.strip() if experiencia else "No disponible",
                "ubicacion": ubicacion.strip() if ubicacion else "No disponible",
                "educacion": educacion.strip() if educacion else "No disponible"
            }

                
            # Crear un objeto Document por cada fila
            doc = Document(page_content=page_content, metadata=metadata)
            documentos.append(doc)

        print(f"Cantidad de documentos procesados: {len(documentos)}")
        if not documentos:
            print("⚠️ No hay documentos válidos para crear el índice.")
            return None

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        indice = None

        # Procesar documentos por lotes (batch_size)
        for i in range(0, len(documentos), batch_size):
            batch = documentos[i : i + batch_size]
            print(f"🔄 Procesando batch {i} a {i + len(batch)}...")
            if indice is None:
                indice = FAISS.from_documents(batch, embeddings)
            else:
                indice.add_documents(batch)

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        indice.save_local(FAISS_INDEX_PATH)
        print(f"Total documentos en el índice: {indice.index.ntotal}")
        return indice

    else:
        print("♻️ Cargando índice existente...")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True
        )

# =============================================================================
# Función para realizar búsqueda semántica en FAISS.
# =============================================================================
def embed_and_search_in_faiss(query_text, docsearch, top_k=40):
    resultados_legibles = []
    try:
        resultados = docsearch.similarity_search_with_score(query_text, k=top_k)
        print(f"Número de resultados devueltos por similarity_search_with_score: {len(resultados)}")
        for doc, dist in resultados:
            resultado = {
                "Nombre": doc.metadata.get('name', 'Sin Nombre').title(),
                "ID": str(doc.metadata.get('id', 'Desconocido')),
                "Distancia": round(float(dist), 2),
                "Descripción": doc.page_content[:100] + "...",
                "Correo": doc.metadata.get('email', ""),
                "Teléfono": doc.metadata.get('telefono', ""),
                "Idiomas": doc.metadata.get('idiomas', "No disponible"),
                "Habilidades": doc.metadata.get('habilidades', "No disponible")
            }
            resultados_legibles.append(resultado)
        print(f"🔎 Resultado FAISS (formateado): {resultado}")
        print("\n=== Resultado Final embed_and_search_in_faiss (formateado) ===")
        print(resultados_legibles)
        print("Tipo de datos:", type(resultados_legibles))
        print("==========================================\n")
        return resultados_legibles
    except Exception as e:
        print(f"Error en FAISS: {str(e)}")
        return resultados_legibles

# =============================================================================
# Función de reordenamiento para RAG + LLM.
# Toma los 20 resultados de FAISS y usa el LLM para reordenarlos.
# =============================================================================
async def rerank(candidatos, descripcion_puesto):
    def limpiar_variables_globales():
        global ranking_final, resultados_formateados
        ranking_final = []
        resultados_formateados = []
    limpiar_variables_globales()

    # Construir la lista de CVs con el formato adecuado
    resumenes = []
    valid_ids = ", ".join(str(c['ID']) for c in candidatos)
    
    for idx, c in enumerate(candidatos, start=1):
        # Obtenemos idiomas y habilidades de la metadata
        idiomas = c.get('Idiomas', 'No disponible')
        habilidades = c.get('Habilidades', 'No disponible')
        
        resumen = (
            f"{idx}. ID: {c['ID']}, "
            f"Nombre: {c['Nombre']}\n"
            f"Descripción: {c['Descripción']}\n"
            f"Idiomas: {idiomas}\n"
            f"Habilidades: {habilidades}\n"
            f"Experiencia: {c.get('experiencia', 'No disponible')}\n"
            f"Ubicación: {c.get('ubicacion', 'No disponible')}\n"
            f"Educación: {c.get('educacion', 'No disponible')}\n"
            f"Correo: {c.get('Correo', '')}\n"
            f"Teléfono: {c.get('Teléfono', '')}\n"
        )

        resumenes.append(resumen)
    
    resumen_texto = "\n".join(resumenes)

    prompt = f"""
Eres un experto(a) en reclutamiento y selección de personal para todo tipo de roles.

Puesto: {descripcion_puesto}

Utiliza **exclusivamente** los siguientes candidatos, manteniendo intactos sus IDs. Si alguno no 
cumple los criterios de la descripcion no lo añadas en el ranking.
Los únicos IDs válidos son: {valid_ids}.

{resumen_texto}

Criterios clave de evaluación (ejemplos):
- Años de experiencia relevantes.
- Competencias técnicas y/o especializadas.
- Habilidades blandas o de liderazgo (si aplican).
- Idiomas (si son necesarios).
- Ubicación y disponibilidad geográfica (si corresponde).

Se presentan 20 CVs resumidos.
Objetivo:
Selecciona únicamente a los 5 candidatos que mejor cumplan los criterios anteriores.
Ordénalos del 1 al 5 en un ranking y **no modifiques los IDs**; utiliza exactamente los que se han proporcionado.
Justifica brevemente tu elección para cada candidato, mencionando años de experiencia, habilidades, idiomas, etc.

**Devuelve la respuesta en formato JSON**, con la siguiente estructura:
[
    {{"ID": "151", "Justificación": "Texto de justificación"}},
    {{...}},
    ...

]
    """
    
    print("=== Prompt enviado al LLM ===")
    print(prompt)
    ranking_text = await generar_respuesta(prompt)

    # Limpiar la respuesta para remover delimitadores de código si existen
    ranking_text = ranking_text.strip()
    if ranking_text.startswith("```"):
        # Remover la primera línea (delimitador) y la última línea si es también un delimitador
        ranking_text = "\n".join(ranking_text.splitlines()[1:])
        if ranking_text.endswith("```"):
            ranking_text = "\n".join(ranking_text.splitlines()[:-1])
    ranking_text = ranking_text.strip()

    # Procesar la respuesta JSON recibida
    try:
        ranking_json = json.loads(ranking_text)
    except json.JSONDecodeError as e:
        print("❌ Error al decodificar JSON:", e)
        return [{"Error": "No se pudo procesar la respuesta del LLM. Verifica el formato JSON."}]
    
    # Asociar cada entrada JSON al candidato correspondiente
    ranking_final = []
    for item in ranking_json:
        current_id = str(item.get("ID", "")).strip()
        justificacion = item.get("Justificación", "").strip()
        candidato = next((c for c in candidatos if str(c["ID"]).strip() == current_id), None)
        if candidato:
            candidato["Justificación"] = justificacion
            ranking_final.append(candidato)
        else:
            print(f"⚠️ No se encontró candidato con ID: {current_id}")
    
    if not ranking_final:
        print("⚠️ No se encontraron coincidencias de IDs.")
        return [{"Error": "No se encontraron coincidencias. Revisa el formato de los IDs o la lógica de matching."}]
    
    # Formatear los resultados en un formato estructurado para la interfaz gráfica
    resultados_formateados = []
    for idx, candidato in enumerate(ranking_final, start=1):
        resultados_formateados.append({
            "Posición": idx,
            "Nombre": candidato.get('Nombre', 'Sin Nombre'),
            "ID": candidato.get('ID', 'Desconocido'),
            "Descripción": candidato.get('Descripción', 'Sin Contenido'),
            "Justificación": candidato.get('Justificación', 'Sin Justificación'),
            "Correo": candidato.get('Correo', 'No disponible'),
            "Teléfono": candidato.get('Teléfono', 'No disponible')
        })
    
    return resultados_formateados



# =============================================================================
# Función principal de búsqueda.
# =============================================================================
async def buscar_cvs(descripcion_puesto, option_toggle):
    def limpiar_variables_globales():
        global ranking_final, resultados_formateados
        ranking_final = []
        resultados_formateados = []

    limpiar_variables_globales()
    verificar_base_datos()
    resultados = []
    try:
        conn, cursor = connect_db(db_name="cv_database.db")
        if not conn:
            print("❌ Error: No se pudo conectar a la base de datos.")
            return []
        indice = build_or_load_vector_index(conn, cursor)
        if not indice:
            print("⚠️ Advertencia: No se pudo construir/cargar el índice FAISS.")
            return []
        candidatos = embed_and_search_in_faiss(descripcion_puesto, indice, top_k=40)
        print(f"🔍 Se encontraron {len(candidatos)} candidatos con FAISS.")
        if not candidatos:
            print("⚠️ Advertencia: No se encontraron candidatos en la búsqueda semántica.")
            return []
        print(f"Valor de option_toggle: '{option_toggle}'")
        if option_toggle == "🤖 RAG + LLM (IA Avanzada)":
            print("🔄 Seleccionando y rankeando los mejores candidatos con el LLM...")
            ranking = await rerank(candidatos, descripcion_puesto)
            resultados = ranking
        else:
            print("✅ Resultados obtenidos con Solo RAG.")
            resultados = candidatos or []
    except Exception as e:
        print(f"❌ Error crítico en buscar_cvs: {str(e)}")
        resultados = []
    finally:
        if conn:
            close_db(conn)

    print("\n=== Resultado Final buscar_cvs ===")
    print(resultados)
    print("Tipo de datos:", type(resultados))
    print("==========================================\n")
    return resultados

# =============================================================================
# Función para mostrar resultados en formato de texto.
# =============================================================================
def mostrar_resultados_texto(resultados):
    if isinstance(resultados, list):
        if len(resultados) == 1 and "Error" in resultados[0]:
            return resultados[0]["Error"]
        texto = ""
        for match in resultados:
            texto += (
                f"Posición: {match.get('Posición', 'N/A')}\n"
                f"Nombre: {match.get('Nombre', 'Sin Nombre')}\n"
                f"ID: {match.get('ID', 'Desconocido')}\n"
                f"Descripción: {match.get('Descripción', 'Sin Contenido')}\n"
                f"Justificación: {match.get('Justificación', 'Sin Justificación')}\n"
                "-------------------------\n"
            )
        return texto
    elif isinstance(resultados, str):
        return resultados
    else:
        return "Error interno: el formato de resultados no es válido."

# =============================================================================
# Función para generar respuesta de OpenAI de forma asíncrona.
# =============================================================================
async def generar_respuesta(prompt):
    if not OPENAI_API_KEY:
        raise ValueError("❌ Error: La API Key de OpenAI no está configurada.")
    api_key = OPENAI_API_KEY.strip()

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un experto en selección de personal."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.5
    }
    print("=== Solicitud a la API ===")
    print(json.dumps(payload, indent=4))

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response_text = await response.text()
            if response.status == 200:
                response_json = await response.json()
                content = response_json['choices'][0]['message']['content']
                print("=== Respuesta de la API ===")
                print(json.dumps(response_json, indent=4))
                return content.strip()
            else:
                try:
                    error_json = await response.json()
                except Exception:
                    error_json = {"error": response_text}
                print(f"❌ Error en OpenAI: {json.dumps(error_json, indent=4)}")
                return f"Error en la API: {json.dumps(error_json, indent=4)}"

# =============================================================================
# Bloque principal (para pruebas locales)
# =============================================================================
async def main():
    modo = "🤖 RAG + LLM (IA Avanzada)"
    resultados = await buscar_cvs(
        descripcion_puesto="Desarrollador Python con experiencia en machine learning",
        option_toggle=modo
    )

    if isinstance(resultados, list):
        for res in resultados:
            print(f"ID: {res.get('ID', 'Desconocido')} | Justificación: {res.get('Justificación', 'N/A')}")
    else:
        print("Ranking obtenido:")
        print(resultados)

if __name__ == "__main__":
    asyncio.run(main())
