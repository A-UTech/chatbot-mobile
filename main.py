import os
from datetime import datetime
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

load_dotenv()
TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

store = {}


def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

system_prompt_roteador = ("system",
"""
    ### PERSONA SISTEMA
    Você é o Assessor.AI — um assistente pessoal de compromissos e finanças. É objetivo, responsável, confiável e empático, com foco em utilidade imediata. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.
    - Evite jargões.
    - Evite ser prolixo.
    - Não invente dados.
    - Respostas sempre curtas e aplicáveis.
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    
    
    ### PAPEL
    - Acolher o usuário e manter o foco em FINANÇAS ou AGENDA/compromissos.
    - Decidir a rota: {{financeiro | agenda | fora_escopo | faq}}.
    - Responder diretamente em:
    (a) saudações/small talk, ou 
    (b) fora de escopo (redirecionando para finanças/agenda).
    - Seu objetivo é conversar de forma amigável com o usuário e tentar identificar se ele menciona algo sobre finanças ou agenda.
    - Em fora_escopo: ofereça 1–2 sugestões práticas para voltar ao seu escopo (ex.: agendar algo, registrar/consultar um gasto).
    - Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.
    
    
    ### REGRAS
    - Seja breve, educado e objetivo.
    - Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
    - Responda de forma textual.
    
    
    ### PROTOCOLO DE ENCAMINHAMENTO (texto puro)
    ROUTE=<financeiro|agenda>
    PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
    PERSONA=<copie o bloco "PERSONA SISTEMA" daqui>
    CLARIFY=<pergunta mínima se precisar; senão deixe vazio>
    
    
    ### SAÍDAS POSSÍVEIS
    - Resposta direta (texto curto) quando saudação ou fora de escopo.
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
        "ai": "Olá! Posso te ajudar com finanças ou agenda; por onde quer começar?"
    },
    # 2) Fora de escopo -> recusar e redirecionar
    {
        "human": "Me conta uma piada.",
        "ai": "Consigo ajudar apenas com finanças ou agenda. Prefere olhar seus gastos ou marcar um compromisso?"
    },
    # 3) Finanças -> encaminhar (protocolo textual)
    {
        "human": "Quanto gastei com mercado no mês passado?",
        "ai": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 4) Ambíguo -> pedir 1 clarificação mínima (texto direto, sem encaminhar)
    {
        "human": "Agendar pagamento amanhã às 9h",
        "ai": "Você quer lançar uma transação (finanças) ou criar um compromisso no calendário (agenda)?"
    },
    # 5) Agenda -> encaminhar (protocolo textual) — exemplo explícito
    {
        "human": "Tenho reunião amanhã às 9h?",
        "ai": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho reunião amanhã às 9h?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 6) FAQ -> Resposta com base no documento FAQ
    {
        "human": "Qual o objetivo do Assessor.AI?",
        "ai": "ROUTE=faq\nPERGUNTA_ORIGINAL=Qual o objetivo do Assessor.AI?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 7) FAQ -> Resposta com base no documento FAQ
    {
        "human": "Qual o telefone de contato?",
        "ai": "ROUTE=faq\nPERGUNTA_ORIGINAL=Qual o telefone de contato?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    }
]

fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)

# -------------------- PROMPTS ESPECIALISTAS --------------------
# prompt do agente financeiro
# system_prompt_financeiro = ("system",
#                             """
#                             ### OBJETIVO
#                             Interpretar a PERGUNTA_ORIGINAL sobre finanças e operar as tools de `transactions` para responder.
#                             A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.
#
#
#                             ### TAREFAS
#                             - Consultar gastos/entradas/dívidas (totais, por categoria, por estabelecimento, etc).
#                             - Inserir/atualizar/deletar lançamentos financeiros.
#                             - Resumir saúde financeira (entradas, gastos, dívidas, saldo, tendências).
#
#
#
#                             ### CONTEXTO
#                             - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
#                             - Entrada vem do Roteador via protocolo:
#                             - ROUTE=financeiro
#                             - PERGUNTA_ORIGINAL=...
#                             - PERSONA=...   (use como diretriz de concisão/objetividade)
#                             - CLARIFY=...   (se preenchido, priorize responder esta dúvida antes de prosseguir)
#
#
#                             ### REGRAS
#                             - Use o {chat_history} para resolver referências ao contexto recente.
#
#
#
#                             ### SAÍDA (JSON)
#                             Campos mínimos para enviar para o orquestrador:
#                             # Obrigatórios:
#                              - dominio   : "financeiro"
#                              - intencao  : "consultar" | "inserir" | "atualizar" | "deletar" | "resumo"
#                              - resposta  : uma frase objetiva
#                              - recomendacao : ação prática (pode ser string vazia se não houver)
#                             # Opcionais (incluir só se necessário):
#                              - acompanhamento : texto curto de follow-up/próximo passo
#                              - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
#                              - escrita        : {{"operacao":"adicionar|atualizar|deletar","id":123}}
#                              - janela_tempo   : {{"de":"YYYY-MM-DD","ate":"YYYY-MM-DD","rotulo":'mês passado'}}
#                              - indicadores    : {{chaves livres e numéricas úteis ao log}}
#
#
#                             ### HISTÓRICO DA CONVERSA
#                             {chat_history}
#                             """
#                             )
#
# # Especialista financeiro (mesmo example_prompt_pair)
# shots_financeiro = [
#     {
#         "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
#         "ai": """{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}"""
#     },
#     {
#         "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Registrar almoço hoje R$ 45 no débito\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
#         "ai": """{{"dominio":"financeiro","intencao":"inserir","resposta":"Lancei R$ 45,00 em 'comida' hoje (débito).","recomendacao":"Deseja adicionar uma observação?","escrita":{{"operacao":"adicionar","id":2045}}}}"""
#     },
#     {
#         "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quero um resumo dos gastos\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
#         "ai": """{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}"""
#     },
# ]

# fewshots_financeiro = FewShotChatMessagePromptTemplate(
#     examples=shots_financeiro,
#     example_prompt=example_prompt_base,
# )

prompt_roteador = ChatPromptTemplate.from_messages([
    system_prompt_roteador,
    fewshots_roteador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

# prompt_financeiro = ChatPromptTemplate.from_messages([
#     system_prompt_financeiro,
#     fewshots_roteador,
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder("agent_scratchpad")
# ]).partial(today_local=today.isoformat())

# financeiro_agent = create_tool_calling_agent(llm, TOOLS, prompt_financeiro)
#
# financeiro_executor_base = AgentExecutor(
#     agent=financeiro_agent,
#     tools=TOOLS,
#     verbose=False,
#     handle_parsing_errors=True,
#     return_intermediate_steps=False
# )
#
# financeiro_executor = RunnableWithMessageHistory(
#     financeiro_executor_base,
#     get_session_history=get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history"
# )


roteador_chain = RunnableWithMessageHistory(
    prompt_roteador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

while True:
    user_input = input(">>> ")
    if user_input.lower() in ["biribinha"]:
        print("Tchau!")
        break
    try:
        resposta_roteador = roteador_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "DISNARA"}}
        )
        print(resposta_roteador)
    except Exception as e:
        print(f"Erro ao consumir a API: {e}")