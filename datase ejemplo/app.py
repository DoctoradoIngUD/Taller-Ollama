from flask import Flask, request, jsonify, render_template
import ollama
import ConvertirAFrases as caf
import chromadb

app = Flask(__name__)


docs = caf.cargar_jsons("./datasets")
print("Ejemplo documento:", docs[0][:200])

chroma_client = chromadb.Client()
coleccion = chroma_client.get_or_create_collection("yahellball")


if coleccion.count() == 0:
    for i, doc in enumerate(docs):
        emb = ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"]
        coleccion.add(documents=[doc], embeddings=[emb], ids=[str(i)])

#RAG
def consultar(pregunta):
    emb = ollama.embeddings(model="nomic-embed-text", prompt=pregunta)["embedding"]
    resultados = coleccion.query(query_embeddings=[emb], n_results=3)
    contexto = "\n".join(resultados["documents"][0])
    prompt = f"""
    Usa la siguiente información para responder de manera clara:

    {contexto}

    Pregunta: {pregunta}
    """
    respuesta = ollama.generate(model="llama3", prompt=prompt)
    return respuesta["response"]

#WEB
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        print("Datos recibidos:", data)
        question = data.get("question", "")
        answer = consultar(question)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Error en /ask:", e)
        return jsonify({"answer": "Error interno"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
