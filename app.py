import os
from datetime import datetime
import traceback
from zoneinfo import ZoneInfo
import random
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

from flask import Flask, request, jsonify
from flask_cors import CORS

from pymongo import MongoClient
import pytz

from mongo_tools import query_registros

load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


tz_sp = pytz.timezone("America/Sao_Paulo")
today = datetime.now(tz_sp)

app = Flask(__name__)
CORS(app)


store = {}


def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

system_prompt_roteador = ("system",
                          """
                              ### PERSONA SISTEMA
                              - Você é o Igestinha, o assistente virtual do usuário, você tem que ajudar as pessoas com as contagens de suas respectivas empresas
                              - Evite jargões.
                              - Evite ser prolixo.
                              - Não invente dados.
                              - Respostas não precisam ser necessariamente curtas, mas procure não falar demais.
                              - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
                          
                              ### PAPEL
                              - Acolher o usuário e manter o foco em CONTAGENS da empresa.
                              - Decidir a rota: {{especialista}}.
                              - Responder diretamente em:
                              (a) saudações/small talk, ou 
                              - Seu objetivo é conversar de forma amigável com o usuário e trazer informações sobre as contagens, que estão no banco MongoDB.
                              - Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.
                          
                              ### REGRAS
                              - Seja breve, educado e objetivo.
                              - Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
                              - Responda de forma textual.
                          
                          
                              ### PROTOCOLO DE ENCAMINHAMENTO (texto puro)
                              ROUTE=<especialista>
                              PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
                              PERSONA=<copie o bloco "PERSONA SISTEMA" daqui>
                              CLARIFY=<pergunta mínima se precisar; senão deixe vazio>
                          
                          
                              ### SAÍDAS POSSÍVEIS
                              - Resposta direta (texto curto) quando for saudação.
                              - Encaminhamento ao especialista usando exatamente o protocolo acima.
                          
                          
                              ### HISTÓRICO DA CONVERSA
                              {chat_history}
                              """
                          )

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

shots_roteador = [
    # 1) Saudação -> resposta direta
    {
        "human": "Oi, tudo bem?",
        "ai": "Olá! Posso te ajudar com quais consultas hoje?"
    },
    # 2) Pergunta sobre contagens -> encaminhar ao especialista
    {
        "human": "Quais foram as condenas mais frequentes na semana passada?",
        "ai": "ROUTE=especialista\nPERGUNTA_ORIGINAL=Quais foram as condenas mais frequentes na semana passada?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 3) Quem é você? -> resposta direta
    {
        "human": "Quem é você?",
        "ai": "Eu sou o Igestinha, seu assistente virtual para ajudar com as contagens da sua empresa. Como posso ajudar hoje?"
    }
]

fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)

# -------------------
# - PROMPTS ESPECIALISTAS --------------------
system_prompt_especialista = ("system",
                              """
                              ### OBJETIVO
                              Interpretar a PERGUNTA_ORIGINAL sobre consultas e operar as tools de 'queries' para responder.
                              A saída SEMPRE é JSON (contrato abaixo).
                          
                          
                              ### TAREFAS
                              - Consultar as condenas, as quantidades, os tipos (total, por turno, por dia).
                              - SOMENTE CONSULTAS, você não vai inserir nada no MongoDB.
                              - Resumir a situação atual das contagens, onde que está o maior problema.
                          
                          
                          
                              ### CONTEXTO
                              - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
                              - Entrada vem do Roteador via protocolo:
                              - ROUTE=especialista
                              - PERGUNTA_ORIGINAL=...
                              - PERSONA=...   (use como diretriz de concisão/objetividade)
                              - CLARIFY=...   (se preenchido, priorize responder esta dúvida antes de prosseguir)
                          
                          
                              ### REGRAS
                              - Use o {chat_history} para resolver referências ao contexto recente.
                          
                          
                          
                              ### SAÍDA (JSON)
                              Campos mínimos para enviar para o roteador de volta:
                              # Obrigatórios:
                               - dominio   : "especialista"
                               - intencao  : "consultar" | "resumo"
                               - resposta  : uma frase objetiva e apontamentos
                               - recomendacao : o que está pior no caso (pode ser string vazia se não houver)
                              # Opcionais (incluir só se necessário):
                               - acompanhamento : texto curto de follow-up/próximo passo
                               - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
                               - escrita        : {{"operacao":"adicionar|atualizar|deletar","id":123}}
                               - janela_tempo   : {{"de":"YYYY-MM-DD","ate":"YYYY-MM-DD","rotulo":'mês passado'}}
                               - indicadores    : {{chaves livres e numéricas úteis ao log}}
                          
                          
                              ### HISTÓRICO DA CONVERSA
                              {chat_history}
                              """
                              )

# Especialista financeiro (mesmo example_prompt_pair)
shots_especialista = [
    {
        "human": "ROUTE=especialista\nPERGUNTA_ORIGINAL=Quais foram as condenas mais frequentes na semana passada\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"especialista","intencao":"consultar","resposta":"Semana passada houve x condenas de Aero Saculite e Y de Sangria inadequada","recomendacao":"Tome mais cuidado com a Aero Saculite e Sangria inadequada"}}"""
    },
    {
        "human": "ROUTE=especialista\nPERGUNTA_ORIGINAL=Quais foram as condenas contabilizadas hoje?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"especialista","intencao":"consultar","resposta":"Hoje contaram: Sangria inadequada, Y, Z. Com uma grande frequência em Z","recomendacao":"Focar na redução da frequência de Z, que apresentou maior incidência hoje."}}""",
    },
    {
        "human": "ROUTE=especialista\nPERGUNTA_ORIGINAL=Quero um resumo da semana passada\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"especialista","intencao":"resumo","resposta":"Condenas mais contabilizadas: X, Y, Z, F. As que mais houveram casos: Z e X","recomendacao":""}}"""
    },
]

fewshots_especialista = FewShotChatMessagePromptTemplate(
    examples=shots_especialista,
    example_prompt=example_prompt_base,
)

# Orquestrador
system_prompt_orquestrador = ("system",
                              """
                            ### PAPEL
                            Você é o Agente Orquestrador do IGestinha. Sua função é entregar a resposta final ao usuário **somente** quando o Especialista retornar o JSON.
                        
                        
                            ### ENTRADA
                            - ESPECIALISTA_JSON contendo chaves como:
                            dominio, intencao, resposta, recomendacao (opcional, pode vir vazio), acompanhamento (opcional),
                            esclarecer (opcional), janela_tempo (opcional), evento (opcional), escrita (opcional), indicadores (opcional).
                        
                        
                            ### REGRAS
                            - Use **exatamente** `resposta` do especialista como a **primeira linha** do output.
                            - Não reescreva números/datas se já vierem prontos. Não invente dados. Seja conciso.
                            - Não retorne JSON; **sempre** retorne no FORMATO DE SAÍDA.
                        
                        
                            ### FORMATO DE SAÍDA (sempre ao usuário)
                            <sua resposta será 1 frase objetiva sobre a situação>
                            - Se houver `recomendacao`, adicione uma seção:
                            - Recomendo <recomendacao>
                        
                        
                            ### HISTÓRICO DA CONVERSA
                            {chat_history}
                            """
                              )

shots_orquestrador = [
    # 1) Especialista — consultar
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"especialista","intencao":"consultar","resposta":"Semana passada houve x condenas de Aero Saculite e Y de Sangria inadequada","recomendacao":"Tome mais cuidado com a Aero Saculite e Sangria inadequada","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}""",
        "ai": "Semana passada houve x condenas de Aero Saculite e Y de Sangria inadequada\n- *Recomendação*:\nTome mais cuidado com a Aero Saculite e Sangria inadequada"
    },

    # 2) Especialista — consultar
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"especialista","intencao":"consultar","resposta":"Hoje contaram: Sangria inadequada, Y, Z. Com uma grande frequência em Z.","recomendacao":"Focar na redução da frequência de Z, que apresentou maior incidência hoje."}}""",
        "ai": """Hoje contaram: Sangria inadequada, Y, Z. Com uma grande frequência em Z.\n- *Recomendação*: Focar na redução da frequência de Z, que apresentou maior incidência hoje."""
    },

    # 3) Especialista — Resumo
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"especialista","intencao":"resumo","resposta":"Condenas mais contabilizadas: X, Y, Z, F. As que mais houveram casos: Z e X.","recomendacao":""}}""",
        "ai": """Condenas mais contabilizadas: X, Y, Z, F. As que mais houveram casos: Z e X.\n- *Recomendação*:\n"""
    },
]

fewshots_orquestrador = FewShotChatMessagePromptTemplate(
    examples=shots_orquestrador,
    example_prompt=example_prompt_base
)

prompt_roteador = ChatPromptTemplate.from_messages([
    system_prompt_roteador,
    fewshots_roteador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

prompt_especialista = ChatPromptTemplate.from_messages([
    system_prompt_especialista,
    fewshots_roteador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
]).partial(today_local=today.isoformat())

prompt_orquestrador = ChatPromptTemplate.from_messages([
    system_prompt_orquestrador,
    fewshots_orquestrador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

MONGO_TOOLS = [query_registros]

especialista_agent = create_tool_calling_agent(llm_fast, MONGO_TOOLS, prompt_especialista)

especialista_executor_base = AgentExecutor(
    agent=especialista_agent,
    tools=MONGO_TOOLS,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

especialista_executor = RunnableWithMessageHistory(
    especialista_executor_base,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

roteador_chain = RunnableWithMessageHistory(
    prompt_roteador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

orquestrador_chain = RunnableWithMessageHistory(
    prompt_orquestrador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

@app.route("/<unidade>/chat", methods=["POST"])
def chat(unidade):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Dados não fornecidos ou formato inválido!"}), 400

    user_message = data.get("usuario", "")

    if not user_message:
        return jsonify({"error": "A mensagem do usuário está vazia!"}), 400

    try:
        resposta_roteador = roteador_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": "DISNARA"}}
        )
        if 'ROUTE=' not in resposta_roteador:
            return jsonify({"resposta": resposta_roteador})
        elif 'ROUTE=especialista' in resposta_roteador:
            resposta_especialista = especialista_executor.invoke(
                {"input": resposta_roteador,
                        "unidade": unidade},
                config={"configurable": {"session_id": "DISNARA"}}
            )
            resposta_orquestrador = orquestrador_chain.invoke(
                {"input": resposta_especialista},
                config={"configurable": {"session_id": "DISNARA"}}
            )
            return jsonify({"resposta": resposta_orquestrador})

    except Exception as e:
        print(f"Erro ao consumir a API: {e}")
        return jsonify({"error": "Erro ao processar a solicitação."}), 500


@app.route("/historico/<unidade>/<cargo>/<id_user>/<id_chat>", methods=["GET"])
def historico(unidade, cargo, id_user, id_chat):
    try:
        client = MongoClient(MONGO_URL)
        db = client["igestaDB"]
        coll = db["chatbot"]
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Erro ao conectar ao banco de dados."}), 500

    try:
        cursor = coll.find({
            "unidade": unidade,
            "cargo": cargo,
            "id_user": id_user,
            "id_chat": id_chat
        })

        print(cursor)
        print("-----")

        data = list(cursor)

        print(data)

        if not data:
            return jsonify({"error": "Dados não encontrados!"}), 404

        for d in data:
            d["_id"] = str(d["_id"])

        return jsonify({"historico": data}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)