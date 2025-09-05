# Variáveis de Ambiente
import os
from dotenv import load_dotenv #
import google.generativeai as genai
load_dotenv()
# Bibliotecas do LangChain
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Cria templates e reserva históricos
from langchain_core.runnables.history import RunnableWithMessageHistory # gerencia histórico de msgs
# Armazena histórico em formato estruturado
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_community.chat_message_histories import ChatMessageHistory

# Se a chave da API não foi encontrada
if not os.getenv("GOOGLE_API_KEY"):
    print("\nErro na chave")
    exit(0)

try:
    template = """Você é um assistente de perito criminal. 
    voce ficará responsavel em analisar os documentos e laudos periciais e retornará os suspeitos ou informações
    relevantes a respeito do crime ocorrido.

    Histórico da conversa:
    {history}
    Entrada do Usuário:
    {input}"""

    # Configurar Agente
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Inicializar Modelo 
    llm = ChatGoogleGenerativeAI(temperature=0.4, model="gemini-2.5-flash")
    chain = prompt | llm
    # Configurar histórico
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    def iniciar_assistente():
        print("\nBoas vindas! digite sair para encerrar\n")
        while True:
            pergunta = input("Digite aqui: ")
            if pergunta.lower() in ["sair", "exit"]:
                break
            if not pergunta.strip():
                continue
            try:
                resposta = chain_with_history.invoke(
                    {'input': pergunta},
                    config={'configurable': {'session_id': 'user123'}}
                )
                print("\nAssistente criminal: ", resposta.content)

            except Exception as e:
                print("\nOcorreu um erro ao processar sua pergunta f{e}")

except Exception as e:
    print(f"\nAconteceu um erro: f{e}")

if __name__ == '__main__':
    iniciar_assistente()


