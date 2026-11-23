import os

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

load_dotenv(override=True)
url = make_url(os.getenv("DATABASE_URL"))

print("usuário :", url.username)
print("host    :", url.host)
print("porta   :", url.port)
print("banco   :", url.database)
print("senha   :", len(url.password or ""), "caracteres")
print("senha só com letras e números:", (url.password or "").isalnum())