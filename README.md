# 🤖 Sistema de Selección de Candidatos con RAG, FAISS, LLM y Envío Inteligente de Correos

Este proyecto implementa una plataforma inteligente para la **evaluación y contacto automatizado de candidatos**. Usa recuperación aumentada con generación (RAG), embeddings semánticos y modelos LLM para analizar currículums, rankear candidatos y facilitar la comunicación mediante un agente conversacional y envíos de correos personalizados.

## 🧠 ¿Qué hace este sistema?

* 🔍 Procesa currículums en texto estructurado y los guarda en SQLite.
* 📌 Genera embeddings (Hugging Face) y los indexa en FAISS.
* 🧠 Busca y **rankea automáticamente** candidatos con GPT-4.1-nano o Llama 3.
* 🤖 Incorpora un **agente de IA** para consultas sobre los candidatos finalistas.
* 📤 Genera y envía **correos profesionales personalizados** (con validación vía HumanLayer).
* 🖥️ Cuenta con una **interfaz gráfica multicomponente** mediante Gradio.

## 📂 Estructura del proyecto

```
.
├── Base_datos_final.txt         # Fuente inicial de CVs en texto plano
├── cv_database.db               # Base de datos SQLite con la información procesada
├── faiss_index/                 # Carpeta con el índice vectorial FAISS
├── candidatos.json              # Lista de candidatos seleccionados
├── utils.py                     # Funciones de embeddings, búsqueda, ranking y LLM
├── search_ui.py                 # Lógica de búsqueda y ranking (FAISS + GPT)
├── send_email.py                # Sistema de generación y envío de correos
├── interface_chat.py            # Agente conversacional (Q&A sobre los candidatos)
├── main.py                      # Interfaz Gradio principal
├── chat_agent.py                # Agente de IA para interacción libre con múltiples candidatos
├── load_txt_to_db.py            # Script de carga inicial de CVs a SQLite
├── config.py, .env              # Variables de entorno (claves, correo, modelos)
```

## 🧰 Requisitos

* Python 3.9+
* `.env` con variables de configuración:

```env
OPENAI_API_KEY=tu_api_key_openai
OPENROUTER_API_KEY=tu_api_key_openrouter
SMTP_FROM_EMAIL=remitente@dominio.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=usuario@gmail.com
SMTP_PASSWORD=tu_contraseña
HUMANLAYER_API_KEY=tu_api_key_humanlayer
```

Instala dependencias:

```bash
pip install -r paquetes.txt
```

## 🚀 Cómo funciona

### Fase 1: Carga y procesamiento de CVs

```bash
python load_txt_to_db.py
```

Los datos del archivo `Base_datos_final.txt` se limpian y almacenan en SQLite.

### Fase 2: Búsqueda y ranking inteligente

Ejecuta la app:

```bash
python main.py
```

Abre [http://localhost:7861](http://localhost:7861) y accede a:

* **Buscar candidatos**: introduce la descripción del puesto.
* **Agente de reclutamiento**: chatea con la IA para preguntar por idiomas, experiencia, habilidades, etc.
* **Enviar correos**: genera correos profesionales, que serán validados manualmente por HumanLayer antes del envío.

### Fase 3: Interacción avanzada con el agente de IA

El agente:

* Procesa consultas como:
  *“¿Qué idiomas hablan los mejores candidatos?”*
  *“¿Cuáles tienen experiencia en Python y están en Madrid?”*

* Si el usuario escribe:
  *“Envía un correo para la entrevista del viernes a las 10h”*
  → se redacta y lanza automáticamente un correo por candidato con HumanLayer para validación.

## 🤖 Modelos utilizados

* **Embeddings**: `distiluse-base-multilingual-cased-v2`
* **LLM**:

  * `gpt-4.1-nano` (OpenAI API)
  * Opción local: `Llama 3.1` (integración experimental)
* **RAG**: Vector search + rerank con LLM
* **Correo inteligente**: `aiosmtplib` + HumanLayer + `EmailMessage`

## 🔐 Seguridad

* El archivo `.env` **nunca debe subirse a Git**.
* Todos los correos pasan por aprobación vía HumanLayer.
* Se pueden consultar y auditar las solicitudes recientes desde la interfaz.

## 🧪 Próximas mejoras

* Extracción automática de datos desde PDFs o LinkedIn.
* Soporte para múltiples perfiles de búsqueda simultáneamente.
* Métricas de comparación entre candidatos con visualización gráfica.

## 📄 Licencia

MIT License. Uso libre para proyectos académicos, personales o de investigación.

