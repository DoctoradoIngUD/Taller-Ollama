import json
from pathlib import Path
import chromadb
import ollama

def cargar_jsons(ruta_carpeta):
    documentos = []
    for archivo in Path(ruta_carpeta).glob("*.json"):
        with open(archivo, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convertimos cada entrada del JSON en texto
            if isinstance(data, list):
                for item in data:
                    documentos.append(str(item))
            elif isinstance(data, dict):
                documentos.append(str(data))
    return documentos

docs = cargar_jsons("./datasets")  # Carpeta con tus 4 JSON
print("Ejemplo documento:", docs[0][:200])


# Inicializar Chroma
chroma_client = chromadb.Client()
coleccion = chroma_client.create_collection("yahellball")

# Crear embeddings y guardarlos
for i, doc in enumerate(docs):
    emb = ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"]
    coleccion.add(documents=[doc], embeddings=[emb], ids=[str(i)])

def consultar(pregunta):
    # Embedding de la pregunta
    emb = ollama.embeddings(model="nomic-embed-text", prompt=pregunta)["embedding"]
    
    # Buscar en Chroma
    resultados = coleccion.query(query_embeddings=[emb], n_results=3)
    contexto = "\n".join(resultados["documents"][0])
    
    # Pasar contexto al modelo
    prompt = f"""
    Usa la siguiente información para responder de manera clara:

    {contexto}

    Pregunta: {pregunta}
    """
    respuesta = ollama.generate(model="llama3", prompt=prompt)
    return respuesta["response"]

print("CON DATASET (RAG):", consultar("¿Quién fue el mejor jugador en 1923 en yahellball?"))

resp_normal = ollama.generate(model="llama3", prompt="¿Quién fue el mejor jugador en 1923 en yahellball?")
print("MODELO SIN RAG:", resp_normal["response"])
