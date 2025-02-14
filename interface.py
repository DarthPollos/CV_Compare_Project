import gradio as gr
from search_ui import search_interface

def main_interface():
    # ✅ CSS mejorado
    custom_css = """
    /* ====== ESTILOS GLOBALES ====== */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: "Arial", sans-serif;
    }

 #welcome-container {
    width: 100%;
    height: 100vh;
    background: url("./background.png") no-repeat center center/cover;
    background-color: #04244d; /* Color de respaldo si la imagen no carga */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    transition: opacity 0.5s ease-in-out, visibility 0.5s ease-in-out;
}


    #welcome-content {
        max-width: 600px;
        padding: 30px;
        background: rgba(0, 0, 0, 0.7);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    #welcome-content h1 {
        font-size: 3em;
        color: white;
    }

    #welcome-content p {
        font-size: 1.5em;
        color: #d0d0d0;
    }

    /* ====== PESTAÑA BÚSQUEDA ====== */
    #search-container {
        width: 100%;
        min-height: 100vh;
        background-color: #04244d !important;
        color: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start; /* Ajusta según tu gusto */
        padding: 20px;
        transition: opacity 0.5s ease-in-out, visibility 0.5s ease-in-out;
    }

    #search-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }

    #job-input, #search-toggle, #search-button, #result-output, #cancel-button {
        width: 80%;
        margin: 15px auto;
        padding: 12px;
        border-radius: 8px;
        font-size: 1.2em;
    }

    #job-input {
    background: #1c3b5a;
    color: white; /* Para que el texto sea visible */
    border: 2px solid #ff8c00;
    padding: 10px;
    border-radius: 8px;
}


    #search-toggle {
        display: flex;
        justify-content: center;
        gap: 15px;
    }

    #search-button {
        background-color: #ff8c00;
        color: white;
        font-size: 1.3em;
        padding: 15px 30px;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 2px 2px 10px rgba(255, 140, 0, 0.5);
    }

    #search-button:hover {
        background-color: #e07b00;
        box-shadow: 2px 2px 15px rgba(255, 140, 0, 0.8);
    }

    #cancel-button {
        background-color: #d9534f !important; /* Rojo */
        color: white !important;
        font-size: 1.2em;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
        margin-top: 20px;
        box-shadow: 2px 2px 10px rgba(217, 83, 79, 0.5);
    }

    #cancel-button:hover {
        background-color: #c9302c;
        box-shadow: 2px 2px 15px rgba(217, 83, 79, 0.8);
    }

    #result-output {
        background: #1c3b5a;
        color: white;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.2em;
        max-height: 400px;
        overflow-y: auto;
        border: 2px solid #ff8c00;
    }
    """

    # ✅ Función para cambiar a la pestaña de búsqueda
    def switch_to_search():
        return gr.Tabs.update(selected=1)

    # ✅ Función para volver a la pestaña de bienvenida (Cancelar búsqueda)
    def cancel_search():
        return gr.Tabs.update(selected=0)

    with gr.Blocks(css=custom_css) as interface:
        with gr.Tabs() as main_tabs:
            # ---- Pestaña 0: Bienvenida ----
            with gr.Tab("Bienvenida"):
                with gr.Column(elem_id="welcome-container"):
                    gr.HTML("""
                        <div id="welcome-content">
                            <h1>Bienvenido a <span style="color: #57d2ff;">HireLens</span></h1>
                            <p>Encuentra los candidatos ideales con inteligencia artificial.</p>
                        </div>
                    """)

            # ---- Pestaña 1: Búsqueda ----
            with gr.Tab("Búsqueda"):
                with gr.Column(elem_id="search-container"):
                    search_interface()
                    cancel_button = gr.Button("Cancelar búsqueda", elem_id="cancel-button")

        # ✅ Callback para "Cancelar búsqueda"
        cancel_button.click(
            fn=cancel_search,  # Función Python que cambia la pestaña
            inputs=[],
            outputs=main_tabs     # Actualiza el objeto Tabs
        )

    return interface
