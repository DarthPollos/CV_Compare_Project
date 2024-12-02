import sqlite3
import pandas as pd

def connect_db(db_name="cv_database.db"):
    """Conecta a la base de datos SQLite y crea la tabla 'cv' si no existe."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cv (
            id TEXT PRIMARY KEY,
            resume_str TEXT,
            category TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def insert_cvs_from_dataset(cursor, dataset):
    """Inserta los CVs del dataset en la base de datos, omitiendo duplicados."""
    for _, row in dataset.iterrows():
        cursor.execute(
            "INSERT OR IGNORE INTO cv (id, resume_str, category) VALUES (?, ?, ?)",
            (row["ID"], row["Resume_str"], row["Category"])
        )

def get_all_cvs(cursor):
    """Recupera todos los CVs almacenados en la base de datos."""
    cursor.execute("SELECT id, resume_str, category FROM cv")
    return cursor.fetchall()

def reset_database(db_name="cv_database.db"):
    """Resetea la base de datos eliminando y recreando la tabla 'cv'."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS cv")
    conn.commit()
    conn.close()
    connect_db(db_name)

def close_db(conn):
    """Cierra la conexión a la base de datos."""
    conn.close()
