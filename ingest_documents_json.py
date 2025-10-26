# ingest_documents_json.py
import os
import json
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")

client = OpenAI(api_key=OPENAI_API_KEY)

PDF_PATH = "meu_documento.pdf"  # Ajuste para o caminho do seu PDF
OUTPUT_JSON = "embeddings.json"

# Carrega PDF
print("Carregando PDF...")
reader = PdfReader(PDF_PATH)
pages = [p.extract_text() or "" for p in reader.pages]
full_text = "\n\n".join(pages)

# Split em chunks
print("Dividindo em chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(full_text)

print(f"Total de chunks: {len(chunks)}")

# Gera embeddings
print("Gerando embeddings...")
embeddings_data = []

for i, chunk in enumerate(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )
    embedding = response.data[0].embedding
    
    embeddings_data.append({
        "chunk_id": i,
        "text": chunk,
        "embedding": embedding
    })
    
    if (i + 1) % 10 == 0:
        print(f"Processados {i + 1}/{len(chunks)} chunks...")

# Salva no JSON
print(f"Salvando em {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

print(f"✅ Ingestão concluída! {len(chunks)} chunks salvos.")
