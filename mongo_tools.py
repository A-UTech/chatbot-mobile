import os
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Optional, List
from langchain.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import pytz

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

def get_conn():
    client = MongoClient(MONGO_URL)
    db = client["igestaDB"]
    coll = db["registros"]
    return coll

class QueryRegistrosModel (BaseModel):
    start_date: Optional[str] = Field(default=None, description="Data de início para a consulta (formato YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, description="Data de fim para a consulta (formato YYYY-MM-DD).")
    unidade: str = Field(Default=None, description="Unidade da empresa para filtrar os registros.")
    gestor: str = Field(default=None, description="Nome do gestor para filtrar os registros.")

@tool("query_registros", args_schema=QueryRegistrosModel)
def query_registros(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    unidade: str = None,
    gestor: str = None
) -> list[dict]:
    """
    Consulta registros na coleção 'registros' do MongoDB com filtros de unidade e gestor, tipo e datas locais (America/Sao_Paulo).

    """
    coll = get_conn()
    query = {}
    if gestor:
        query["gestor"] = {"$regex": gestor, "$options": "i"}
    if unidade:
        query["unidade"] = {"$regex": unidade, "$options": "i"}

    tz = pytz.timezone("America/Sao_Paulo")

    registros = []

    for doc in coll.find(query):
        registros.append({
            "id": str(doc.get("_id")),
            "condenas": doc.get("condenas"),
            "data": doc.get("data").astimezone(tz).strftime("%Y-%m-%d %H:%M:%S"),
            "unidade": doc.get("unidade"),
            "gestor": doc.get("gestor"),
            "lote": doc.get("lote")
        })

    return {"status": "success", "data": registros, "count": len(registros)}

TOOLS = [query_registros]
