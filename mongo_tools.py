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
    return client['igesthinha']

class Condena(BaseModel):
    nome: str
    tipo: str
    quantidade: int

class AddRegistroArgs(BaseModel):
    condenas: List[Condena] = Field(..., description="Lista de condenas do registro.")
    data: Optional[str] = Field(default=None, description="Data do registro no formato ISO 8601. Se ausente, usa a data atual.")
    empresa: str = Field(..., description="Nome da empresa.")
    unidade: str = Field(..., description="Nome da unidade da empresa.")
    gestor: str = Field(..., description="Nome do gestor responsável.")
    turno: int = Field(..., description="Turno do registro.")
    lote: str = Field(..., description="Lote do registro.")

@tool("add_registro", args_schema=AddRegistroArgs)
def add_registro(
        condenas: List[Condena],
        empresa: str,
        unidade: str,
        gestor: str,
        turno: int,
        lote: str,
        data: Optional[str] = None,
) -> dict:
    """Insere um registro de condena no banco de dados MongoDB."""
    db = get_conn()
    try:
        registro = {
            "condenas": [c.dict() for c in condenas],
            "data": datetime.fromisoformat(data) if data else datetime.utcnow(),
            "empresa": empresa,
            "unidade": unidade,
            "gestor": gestor,
            "turno": turno,
            "lote": lote
        }
        result = db.registros.insert_one(registro)
        return {"status": "ok", "id": str(result.inserted_id)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

class QueryRegistrosArgs(BaseModel):
    start_date: Optional[str] = Field(default=None, description="Data de início para a consulta (formato YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, description="Data de fim para a consulta (formato YYYY-MM-DD).")
    empresa: Optional[str] = Field(default=None, description="Nome da empresa para filtrar.")
    unidade: Optional[str] = Field(default=None, description="Nome da unidade para filtrar.")
    limit: int = Field(default=10, description="Número máximo de registros a retornar.")

@tool("query_registros", args_schema=QueryRegistrosArgs)
def query_registros(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        empresa: Optional[str] = None,
        unidade: Optional[str] = None,
        limit: int = 10,
) -> dict:
    """Consulta registros de condena com filtros."""
    db = get_conn()
    try:
        query = {}
        if start_date:
            query["data"] = {"$gte": datetime.fromisoformat(start_date)}
        if end_date:
            if "data" not in query:
                query["data"] = {}
            query["data"]["$lte"] = datetime.fromisoformat(end_date)
        if empresa:
            query["empresa"] = empresa
        if unidade:
            query["unidade"] = unidade

        registros = []
        for r in db.registros.find(query).limit(limit):
            r['_id'] = str(r['_id'])
            registros.append(r)
        return {"status": "ok", "registros": registros}

    except Exception as e:
        return {"status": "error", "message": str(e)}
