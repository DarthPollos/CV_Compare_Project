import sqlite3
import re

# Conectar a la base de datos
conn = sqlite3.connect("cv_database.db")
cursor = conn.cursor()

# Crear tabla (incluyendo la columna "ubicacion")
cursor.execute("DROP TABLE IF EXISTS cv")
cursor.execute("""
    CREATE TABLE cv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT,
        email TEXT,
        telefono TEXT,
        educacion TEXT,
        experiencia TEXT,
        habilidades TEXT,
        idiomas TEXT,
        resumen TEXT,
        ubicacion TEXT
    )
""")

# Leer archivo TXT
def read_txt_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# Preprocesar texto (limpiar saltos de línea y espacios extra)
def preprocess_text(text):
    text = text.replace("\r", "\n").replace("•", "-")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Dividir perfiles en base al patrón de "ID: <número> Nombre:"
def split_profiles(text):
    return re.split(r"(?=ID:\s+\d+\s+Nombre:)", text)

# Normalizar texto (convertir a minúsculas y eliminar espacios adicionales)
def normalize_text(value):
    if value:
        return value.strip().lower()
    return "no especificado"

# Extraer datos usando expresiones regulares (incluyendo ubicación)
def extract_data(profile):
    nombre = re.search(r"Nombre:\s+(.*?)\s+Email:", profile)
    email = re.search(r"Email:\s+(.*?)\s+Teléfono:", profile)
    telefono = re.search(r"Teléfono:\s+(.*?)\s+Educación:", profile)
    educacion = re.search(r"Educación:\s+(.*?)\s+Experiencia:", profile, re.DOTALL)
    experiencia = re.search(r"Experiencia:\s+(.*?)\s+Habilidades:", profile, re.DOTALL)
    habilidades = re.search(r"Habilidades:\s+(.*?)\s+Idiomas:", profile)
    idiomas = re.search(r"Idiomas:\s+(.*?)\s+Resumen:", profile)
    # Capturar "Resumen:" hasta "Ubicación:" o, si no hay etiqueta posterior, hasta el final
    resumen = re.search(r"Resumen:\s+(.*?)(?:\s+Ubicación:|$)", profile, re.DOTALL)
    # Capturar "Ubicación:" y todo lo que venga después hasta el final
    ubicacion = re.search(r"Ubicación:\s+(.*)", profile, re.DOTALL)

    return {
        "nombre": normalize_text(nombre.group(1)) if nombre else "no especificado",
        "email": normalize_text(email.group(1)) if email else "no especificado",
        "telefono": normalize_text(telefono.group(1)) if telefono else "no especificado",
        "educacion": normalize_text(educacion.group(1)) if educacion else "no especificado",
        "experiencia": normalize_text(experiencia.group(1)) if experiencia else "no especificado",
        "habilidades": normalize_text(habilidades.group(1)) if habilidades else "no especificado",
        "idiomas": normalize_text(idiomas.group(1)) if idiomas else "no especificado",
        "resumen": normalize_text(resumen.group(1)) if resumen else "no especificado",
        "ubicacion": normalize_text(ubicacion.group(1)) if ubicacion else "no especificado"
    }

# Verificar duplicados usando múltiples campos
def is_duplicate(cursor, data):
    cursor.execute("""
        SELECT COUNT(*) FROM cv WHERE 
        nombre=? AND email=? AND telefono=? AND educacion=? AND experiencia=?
    """, (data['nombre'], data['email'], data['telefono'], 
          data['educacion'], data['experiencia']))
    return cursor.fetchone()[0] > 0

# Procesar archivo TXT
txt_content = read_txt_file("Base_datos_final.txt")
profiles = split_profiles(txt_content)

# Insertar datos en la base de datos
for profile in profiles:
    profile = preprocess_text(profile)
    data = extract_data(profile)

    # Validar datos antes de insertar
    if data['nombre'] != "no especificado" and data['email'] != "no especificado":
        if not is_duplicate(cursor, data):
            cursor.execute("""
                INSERT INTO cv (nombre, email, telefono, educacion, experiencia, habilidades, idiomas, resumen, ubicacion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['nombre'],
                data['email'],
                data['telefono'],
                data['educacion'],
                data['experiencia'],
                data['habilidades'],
                data['idiomas'],
                data['resumen'],
                data['ubicacion']
            ))

# Confirmar cambios en la base de datos
conn.commit()
print("✅ Base de datos creada y registros insertados exitosamente.")

# Cerrar conexión
conn.close()
