import sys
import os
import time

from database import connect_db, close_db
from utils import (
    load_dataset_into_db,
    build_or_load_vector_index,
    embed_and_search_in_faiss,
    rerank_with_deepseek
)

# Evitar warnings de tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """
    Modos de uso:
      1) build_index
         - Construye el índice FAISS a partir de la tabla `cv` en tu DB.

      2) query
         - Pide por consola la descripción del puesto y busca los 5 CVs más similares (RAG).
         - Solo muestra resultados, sin DeepSeek-r1.

      3) rag_deepseek
         - Recupera los 20 CVs más relevantes (RAG).
         - Llama a DeepSeek-r1 para refinar/rerank y muestra un ranking final.
    """

    if len(sys.argv) < 2:
        print("Uso: python main.py [build_index | query | rag_deepseek]")
        return

    mode = sys.argv[1].lower()

    # Conexión DB
    conn, cursor = connect_db()

    if mode == "build_index":
        # Construir índice
        build_or_load_vector_index(cursor, rebuild=True)
        print("Índice FAISS construido y guardado con éxito.")

    elif mode == "query":
        # 1. Leer descripción
        print("Por favor, ingresa la descripción del puesto. Enter en blanco para finalizar:")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        job_description = "\n".join(lines).strip()

        if not job_description:
            print("No se ingresó ninguna descripción.")
            close_db(conn)
            return

        # 2. Cargar índice
        docsearch = build_or_load_vector_index(cursor, rebuild=False)
        if not docsearch:
            print("No se pudo cargar o crear el índice.")
            close_db(conn)
            return

        # 3. Recuperar top 5
        top_matches = embed_and_search_in_faiss(job_description, docsearch, top_k=5)
        print(f"\nSe han recuperado {len(top_matches)} CVs más similares:\n")

        # 4. Imprimir con estilo
        for rank, match in enumerate(top_matches, start=1):
            md = match.metadata
            text = match.page_content
            sc = match.score
            cv_id = md.get("id")
            cv_cat = md.get("category", "desconocida").upper()
            snippet = text[:150].replace("\n", " ")

            print(f"========= RANK #{rank} =========")
            print(f"CV ID:         {cv_id}")
            print(f"Categoría:     {cv_cat}")
            print(f"Distancia:     {sc:.4f}")
            print(f"----- FRAGMENTO CV -----")
            print(snippet + "...")
            print("================================\n")

    elif mode == "rag_deepseek":
        # 1. Leer descripción
        print("Por favor, ingresa la descripción del puesto. Enter en blanco para finalizar:")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        job_description = "\n".join(lines).strip()

        if not job_description:
            print("No se ingresó ninguna descripción.")
            close_db(conn)
            return

        # 2. Cargar índice
        print("🔹 Cargando/creando índice FAISS...")
        docsearch = build_or_load_vector_index(cursor, rebuild=False)
        if not docsearch:
            print("❌ No se pudo cargar ni crear el índice FAISS.")
            close_db(conn)
            return

        # 3. Recuperar top 20
        print("🔹 Recuperando los 20 CVs más relevantes...")
        top_20 = embed_and_search_in_faiss(job_description, docsearch, top_k=20)
        if not top_20:
            print("❌ No se encontraron CVs relevantes.")
            close_db(conn)
            return

        print(f"\n✅ Se han recuperado {len(top_20)} CVs en la fase RAG.")
        print("🔹 Enviando CVs a DeepSeek-r1 para refinado...")

        # 4. Re-rank con deepseek
        final_rank = rerank_with_deepseek(top_20, job_description)
        print("✅ DeepSeek-r1 completado.")

        # 5. Mostrar ranking final
        print("\n=== Ranking final refinado con DeepSeek-r1 ===")
        if not final_rank:
            print("No se obtuvo un ranking final. Revisa la salida anterior.")
        else:
            for pos, item in enumerate(final_rank, start=1):
                cv_id = item["id"]
                cat = item["category"].upper()
                score = item["score"]
                reasons = item["reasons"]
                snippet = item["fragment"]

                print(f"\n--- CV RANK #{pos} ---")
                print(f"ID:     {cv_id}")
                print(f"CAT:    {cat}")
                print(f"SCORE:  {score:.2f}")
                print(f"RAZONES:\n{reasons}")
                print(f"--- FRAGMENTO ---\n{snippet}...")
                print("------------------------")

    else:
        print("Opción desconocida. Usa: build_index, query o rag_deepseek.")

    close_db(conn)


if __name__ == "__main__":
    main()
