import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conexao:
    total = conexao.execute(text("select count(*) from alunos")).scalar()
    print("Conectou! Alunos no banco:", total)