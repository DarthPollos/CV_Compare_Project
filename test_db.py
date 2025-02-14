import sqlite3
import pandas as pd

# Crear una nueva base de datos de prueba
conn = sqlite3.connect("test_cv_database.db")
cursor = conn.cursor()

# Crear la tabla de CVs
cursor.execute('''
    CREATE TABLE IF NOT EXISTS cv (
        id INTEGER PRIMARY KEY,
        nombre TEXT,
        titulo TEXT,
        experiencia INTEGER,
        habilidades TEXT,
        tecnologias TEXT,
        ultimo_puesto TEXT,
        educacion TEXT,
        resumen TEXT
    )
''')

# Cargar el CSV
df = pd.read_csv("candidatos.csv")

# Insertar los datos en la base de datos de prueba
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO cv (id, nombre, titulo, experiencia, habilidades, tecnologias, ultimo_puesto, educacion, resumen) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["id"], row["nombre"], row["titulo"], row["experiencia"], 
        row["habilidades"], row["tecnologias"], row["ultimo_puesto"], 
        row["educacion"], row["resumen"]
    ))

conn.commit()
print("✅ Datos cargados correctamente en la base de datos de prueba.")

# Verificar la inserción
cursor.execute("SELECT * FROM cv LIMIT 5")
print("\n🔍 Ejemplo de datos en la BD de prueba:")
for row in cursor.fetchall():
    print(row)

conn.close()
