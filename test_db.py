import sqlite3
import os

DB_PATH = "cv_database.db"

def test_database_connection():
    print("=== Test de Conexión a la Base de Datos ===")
    if not os.path.exists(DB_PATH):
        print(f"❌ La base de datos no existe en la ruta: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print("✅ Conexión exitosa a la base de datos.")
    conn.close()

def test_table_structure():
    print("\n=== Test de Estructura de la Tabla ===")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(cv);")
    columns = cursor.fetchall()
    print("Columnas de la tabla 'cv':")
    for col in columns:
        print(col)
    conn.close()

def test_data_retrieval():
    print("\n=== Test de Recuperación de Datos ===")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, COALESCE(resumen, '') AS resumen FROM cv")
    rows = cursor.fetchall()
    print(f"Número de filas recuperadas: {len(rows)}")

    for i, row in enumerate(rows):
        print(f"Fila {i}: Longitud -> {len(row)}, Datos -> {row}")
        if len(row) != 3:
            print(f"❌ Fila {i} no tiene 3 columnas.")
        else:
            print(f"✅ Fila {i} tiene 3 columnas.")
        
        for j, value in enumerate(row):
            if value is None or value == '':
                print(f"⚠️  Advertencia: Columna {j} en Fila {i} es nula o vacía")
    conn.close()

if __name__ == "__main__":
    test_database_connection()
    test_table_structure()
    test_data_retrieval()
