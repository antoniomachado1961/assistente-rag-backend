import requests
import uuid

API_URL = "http://127.0.0.1:5000/chat"
session_id = str(uuid.uuid4())

print("=== Assistente de Atendimento ===")
print("Digite 'sair' para encerrar\n")

while True:
    pergunta = input("Voce: ")
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("Ate logo!")
        break

    response = requests.post(API_URL, json={"message": pergunta, "session_id": session_id})

    if response.status_code == 200:
        resposta = response.json()["response"]
        print(f"\nAssistente: {resposta}\n")
    else:
        print(f"Erro: {response.status_code}")
