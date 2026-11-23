import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

load_dotenv(override=True)
base = make_url(os.getenv("DATABASE_URL"))

for porta in (5432, 6543):
    for ssl in ("prefer", "require"):
        engine = create_engine(
            base.set(port=porta),
            connect_args={"connect_timeout": 10, "sslmode": ssl},
        )
        try:
            with engine.connect() as conexao:
                n = conexao.execute(text("select count(*) from alunos")).scalar()
            print(f"OK     porta {porta} sslmode={ssl} -> alunos: {n}")
        except Exception as erro:
            resumo = str(erro).splitlines()[0][:160]
            print(f"FALHA  porta {porta} sslmode={ssl} -> {resumo}")