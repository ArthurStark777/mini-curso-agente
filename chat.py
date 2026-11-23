import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
modelo = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# O histórico guarda a conversa, para o modelo lembrar das perguntas anteriores
historico = [
    {
        "role": "system",
        "content": "Você é um assistente prestativo. Responda sempre em português.",
    }
]

print("Digite sua pergunta (ou 'sair' para encerrar).")

while True:
    pergunta = input("\nVocê: ").strip()

    if pergunta.lower() in ("sair", "exit", "quit"):
        print("Encerrando.")
        break
    if not pergunta:
        continue

    historico.append({"role": "user", "content": pergunta})

    stream = cliente.chat.completions.create(
        model=modelo,
        messages=historico,
        stream=True,
    )

    print("Assistente: ", end="", flush=True)
    resposta_completa = ""
    for pedaco in stream:
        if not pedaco.choices:
            continue
        texto = pedaco.choices[0].delta.content
        if texto:
            print(texto, end="", flush=True)
            resposta_completa += texto
    print()

    historico.append({"role": "assistant", "content": resposta_completa})