import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDINGS_FILE = "embeddings.json"

with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
texts = [item["text"] for item in data]
embeddings = np.array([item["embedding"] for item in data])

conversations = {}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar(query, top_k=3):
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = np.array(response.data[0].embedding)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [texts[i] for i in top_indices]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    relevant_docs = search_similar(user_message)
    context = "\n\n".join(relevant_docs)
    
    conversations[session_id].append({"role": "user", "content": user_message})
    
    messages = [
        {"role": "system", "content": f"Voce e um assistente de atendimento. Use este contexto:\n\n{context}"},
        *conversations[session_id]
    ]
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    assistant_message = response.choices[0].message.content
    
    conversations[session_id].append({"role": "assistant", "content": assistant_message})
    
    return jsonify({"response": assistant_message})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "chunks": len(texts)})

if __name__ == '__main__':
    print(f"API iniciada! Base: {len(texts)} chunks")
    app.run(debug=True, port=5000)