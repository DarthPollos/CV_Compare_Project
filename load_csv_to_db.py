import sqlite3
import pandas as pd
from database import connect_db

# Conectar a la base de datos
conn, cursor = connect_db()

# Crear la tabla si no existe
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
df.columns = df.columns.str.lower()  # 🔹 Convertir nombres de columnas a minúsculas

# Insertar los datos en la base de datos
for _, row in df.iterrows():
    cursor.execute("""
        INSERT OR IGNORE INTO cv (id, nombre, titulo, experiencia, habilidades, tecnologias, ultimo_puesto, educacion, resumen) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["id"], row["nombre"], row["título"], row["experiencia"], 
        row["habilidades"], row["tecnologías"], row["último puesto"], 
        row["educación"], row["resumen"]
    ))

conn.commit()
print("✅ Datos cargados correctamente en la base de datos.")

# Verificar la inserción
cursor.execute("SELECT * FROM cv LIMIT 5")
print("\n🔍 Ejemplo de registros en la BD:")
for row in cursor.fetchall():
    print(row)

conn.close()
