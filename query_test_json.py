# query_test_json.py
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDINGS_FILE = "embeddings.json"

# Carrega embeddings
print("Carregando base de conhecimento...")
with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item["text"] for item in data]
embeddings = np.array([item["embedding"] for item in data])

# Função de busca por similaridade
def similarity_search(query, k=4):
    # Gera embedding da query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array(response.data[0].embedding)
    
    # Calcula similaridade (cosine)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Top k resultados
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": texts[idx],
            "score": float(similarities[idx])
        })
    
    return results

# Teste
query = "Como fazer a instalação?"
print(f"\n🔍 Buscando: '{query}'\n")

results = similarity_search(query, k=3)

for i, result in enumerate(results, 1):
    print(f"--- Resultado {i} (score: {result['score']:.4f}) ---")
    print(result['text'][:500])
    print()
