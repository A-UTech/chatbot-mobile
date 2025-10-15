import os
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Optional, List
from langchain.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

def get_conn():
    client = MongoClient(MONGO_URL)
    db = client['igestaDB']
    coll = db['registros']
    return coll

class QueryRegistrosArgs(BaseModel):
    start_date: Optional[str] = Field(default=None, description="Data de início para a consulta (formato YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, description="Data de fim para a consulta (formato YYYY-MM-DD).")
    unidade: Optional[str] = Field(default=None, description="Nome da unidade para filtrar.")
    limit: int = Field(default=10, description="Número máximo de registros a retornar.")

@tool("query_registros", args_schema=QueryRegistrosArgs)
def query_registros(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        unidade: Optional[str] = None,
        limit: int = 10,
) -> dict:
    """Consulta registros de condena com filtros."""
    coll = get_conn()
    try:
        query = {}
        if start_date:
            query["data"] = {"$gte": datetime.fromisoformat(start_date)}
        if end_date:
            if "data" not in query:
                query["data"] = {}
            query["data"]["$lte"] = datetime.fromisoformat(end_date)
        if unidade:
            query["unidade"] = unidade

        registros = []
        for r in coll.find(query).limit(limit):
            r['_id'] = str(r['_id'])
            registros.append(r)
        return {"status": "ok", "registros": registros}

    except Exception as e:
        return {"status": "error", "message": str(e)}
