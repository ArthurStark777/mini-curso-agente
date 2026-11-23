# pylint: disable=broad-exception-caught, redefined-outer-name, line-too-long, missing-function-docstring
import os
import re
from collections import Counter

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv(override=True)

cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
modelo = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMPRESA_ID = os.getenv("EMPRESA_ID") or "todas"
engine = create_engine(os.getenv("DATABASE_URL"))

MOSTRAR_SQL = False   # mostra a consulta que o modelo montou e os erros (bom para conferir)
MAX_LINHAS = 200     # máximo de linhas que uma consulta SQL pode devolver
N_RESULTADOS = 20    # quantos registros a busca por significado traz

modelo_emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma = chromadb.PersistentClient(path="chroma_db")
colecao = chroma.get_collection("meu_banco")

# tipo no Chroma -> (nome da tabela, campos, apelidos que o usuário pode usar)
CATALOGO = {
    "alunos": ("alunos", "escola, nome, matrícula, turma, turno, ano letivo e se está ativo",
               ["alunos"]),
    "observacoes": ("observacoes_alunos", "aluno, turma, data, título, texto e autor",
                    ["observacoes", "observações"]),
    "avisos": ("avisos_escola", "título, mensagem, turma e data",
               ["avisos"]),
    "notas": ("notas_alunos", "aluno, turma, disciplina, etapa, bimestre, tipo de avaliação, atividade, nota e observação",
              ["notas"]),
    "faltas": ("presencas_sala", "aluno, turma, matéria, data, se foi justificada, justificativa e observação (só as faltas)",
               ["presencas", "presenças", "faltas"]),
    "turmas": ("turmas", "escola, unidade, nome, turno, ano letivo e se está ativa",
               ["turmas"]),
    "professores": ("turma_professores", "turma, turno, professor, matéria, dias da semana, início e fim",
                    ["professores"]),
    "horarios": ("horarios_turmas", "turma, turno, dia da semana, entrada e saída",
                 ["horarios", "horários"]),
    "feriados": ("feriados", "data, descrição e abrangência",
                 ["feriados"]),
    "escolas": ("empresas", "nome, razão social, endereço, telefone e se está ativa",
                ["escolas"]),
    "unidades": ("filiais", "escola, nome da unidade, endereço, tipo e se está ativa",
                 ["unidades"]),
    "departamentos": ("departamentos", "escola, nome e se está ativo",
                      ["departamentos"]),
    "funcionarios": ("funcionarios", "escola, nome, matrícula, cargo, departamento, unidade, datas de admissão e demissão e se está ativo (sem CPF nem PIS)",
                     ["funcionarios", "funcionários"]),
}

# palavras na pergunta -> tipo de registro onde buscar (busca por significado)
FILTROS = {
    "notas": r"\bnotas?\b",
    "faltas": r"\bfaltas?\b",
    "avisos": r"\bavisos?\b",
    "feriados": r"\bferiados?\b",
    "professores": r"\bprofessor(es)?\b",
    "horarios": r"\bhor[áa]rios?\b",
    "funcionarios": r"\bfuncion[áa]rios?\b|\bcargos?\b",
    "departamentos": r"\bdepartamentos?\b",
    "turmas": r"\bturmas\b",
    "unidades": r"\bunidades\b",
    "escolas": r"\bescolas\b",
}

# "o que você tem?", "quais tabelas você tem?", "que dados você tem?"
PADRAO_LISTA = (
    r"^(o ?que|oq|que dados|quais (dados|tabelas|informa[çc][õo]es))\b"
    r"[^?]{0,25}\b(tem|possui|conhece|sabe|existem?)\b"
    r"(\s+(no|na|em|nesse|neste)\s+[\w ]{1,25})?\s*\??\s*$"
)

# "quantos alunos tem?", "quantas turmas existem?"
PADRAO_QUANTOS = (
    r"^quant[oa]s\s+(?P<coisa>\w+)"
    r"(\s+(vc|você|voce|tem|têm|temos|existem|existe|há|ha|cadastrad[oa]s|"
    r"ativ[oa]s|no banco|no sistema|no total|ao todo|de dados))*\s*\??\s*$"
)

TEM_ATIVO = {"alunos", "turmas", "escolas", "unidades", "departamentos", "funcionarios"}

OBS = {
    "professores": "Cada registro é um professor em uma turma e matéria, então o mesmo professor pode contar mais de uma vez.",
    "faltas": "Conta só as faltas, não as presenças.",
    "avisos": "Conta só os avisos ativos.",
}

NOMES = "\n".join(f"- {info[0]}" for info in CATALOGO.values())
DETALHES = "\n".join(f"- {info[0]}: {info[1]}" for info in CATALOGO.values())


def descrever(tipo):
    tabela, campos, _ = CATALOGO[tipo]
    qtd = len(colecao.get(where={"tipo": tipo}, include=[])["ids"])
    return f"Tabela {tabela} ({qtd} registros)\nCampos: {campos}"


def achar_tabela(texto):
    """Devolve o tipo da tabela citada (o nome mais longo vence)."""
    texto = texto.lower().replace("_", " ")
    melhor, tamanho = None, 0
    for tipo, (tabela, _, apelidos) in CATALOGO.items():
        for nome in [tabela.replace("_", " ")] + apelidos:
            if re.search(rf"\b{re.escape(nome)}\b", texto) and len(nome) > tamanho:
                melhor, tamanho = tipo, len(nome)
    return melhor


def contar(tipo):
    tabela = CATALOGO[tipo][0]
    docs = colecao.get(where={"tipo": tipo}, include=["documents"])["documents"]
    linhas = [f"Tabela {tabela}: {len(docs)} registros no total."]

    if tipo in TEM_ATIVO:
        ativos = sum(1 for d in docs if re.search(r"Ativ[oa]: sim", d))
        linhas.append(f"Destes, {ativos} estão ativos.")

    if tipo != "escolas":
        por_escola = Counter()
        for d in docs:
            achado = re.search(r"Escola: ([^|]+)", d)
            if achado:
                por_escola[achado.group(1).strip()] += 1
        if len(por_escola) > 1:
            linhas.append("Por escola:")
            linhas += [f"- {nome}: {qtd}" for nome, qtd in sorted(por_escola.items())]

    if tipo in OBS:
        linhas.append(OBS[tipo])
    return "\n".join(linhas)


# ---------- Memória da conversa ----------
historico = []


def registrar(pergunta, resposta):
    historico.append((pergunta, resposta[:600]))
    del historico[:-4]


def texto_historico():
    if not historico:
        return "(vazio)"
    return "\n".join(f"Usuário: {p}\nAssistente: {r}" for p, r in historico)


def atalho(pergunta, resposta):
    print("Assistente:", resposta)
    registrar(pergunta, resposta)


def transmitir(mensagens):
    """Chama o modelo mostrando a resposta aos poucos e devolve o texto completo."""
    stream = cliente.chat.completions.create(model=modelo, messages=mensagens, stream=True)
    print("Assistente: ", end="", flush=True)
    partes = []
    for pedaco in stream:
        if not pedaco.choices:
            continue
        parte = pedaco.choices[0].delta.content
        if parte:
            print(parte, end="", flush=True)
            partes.append(parte)
    print()
    return "".join(partes)


# ---------- Caminho SQL ----------
def valores(sql):
    try:
        with engine.connect() as con:
            return [str(r[0]) for r in con.execute(text(sql)) if r[0] is not None]
    except Exception:
        return []


def carregar_esquema():
    """Lê do banco só as tabelas e colunas que o usuário curso_leitor pode ver."""
    sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND has_column_privilege(current_user,
                quote_ident(table_schema) || '.' || quote_ident(table_name),
                column_name, 'SELECT')
        ORDER BY table_name, ordinal_position
    """
    tabelas = {}
    try:
        with engine.connect() as con:
            for tabela, coluna, tipo in con.execute(text(sql)):
                tipo = tipo.replace("USER-DEFINED", "enum")
                tabelas.setdefault(tabela, []).append(f"{coluna} {tipo}")
    except Exception:
        return ""
    return "\n".join(f"- {t}({', '.join(cols)})" for t, cols in tabelas.items())


ESQUEMA = carregar_esquema()
if not ESQUEMA:
    print("Aviso: não consegui ler as tabelas do banco; o SQL não vai funcionar.")
elif MOSTRAR_SQL:
    print(f"[sql] {len(ESQUEMA.splitlines())} tabelas liberadas para consulta")

ESCOLAS = valores("SELECT DISTINCT COALESCE(nome_fantasia, razao_social) FROM empresas ORDER BY 1")
TURMAS = valores("SELECT DISTINCT nome FROM turmas ORDER BY 1 LIMIT 60")
DISCIPLINAS = valores("SELECT DISTINCT disciplina FROM notas_alunos ORDER BY 1 LIMIT 60")
ETAPAS = valores("SELECT DISTINCT etapa FROM notas_alunos ORDER BY 1 LIMIT 30")

FILTRO_ESCOLA = (
    ""
    if EMPRESA_ID == "todas"
    else f"- Considere SOMENTE a escola com empresa_id = '{EMPRESA_ID}' (filtre por empresa_id).\n"
)

REGRAS_SQL = f"""Você converte perguntas em consultas SQL (PostgreSQL) para um banco de dados escolar.
Responda com UMA destas opções, sem explicação e sem markdown:
1) uma única consulta SELECT;
2) somente a palavra BUSCA, quando a pergunta é sobre o conteúdo de textos livres
   (observações, avisos, justificativas) ou quando não dá para responder com SQL.

TABELAS E COLUNAS PERMITIDAS (use só estas):
{ESQUEMA}

RELAÇÕES:
- alunos.turma_id -> turmas.id; notas_alunos.aluno_id -> alunos.id;
  presencas_sala.aluno_id -> alunos.id e presencas_sala.turma_id -> turmas.id
- turma_professores.turma_id -> turmas.id; turma_professores.professor_id -> usuarios.id
- horarios_turmas.turma_id -> turmas.id; funcionarios.departamento_id -> departamentos.id
- filiais.empresa_id -> empresas.id; as demais tabelas têm empresa_id -> empresas.id

VALORES CONHECIDOS:
- Escolas: {", ".join(ESCOLAS) or "(desconhecido)"}
- Turmas: {", ".join(TURMAS) or "(desconhecido)"}
- Disciplinas (notas_alunos.disciplina): {", ".join(DISCIPLINAS) or "(desconhecido)"}
- Etapas (notas_alunos.etapa): {", ".join(ETAPAS) or "(desconhecido)"}

REGRAS:
{FILTRO_ESCOLA}- Nunca use SELECT *; liste as colunas.
- Compare textos com ILIKE e use CAST(coluna AS text) em colunas do tipo enum (como turno).
- Para achar aluno, professor ou turma pelo nome, use ILIKE '%nome%'.
- Para o nome da escola, junte com empresas e use COALESCE(nome_fantasia, razao_social).
- Em listas, inclua ORDER BY e LIMIT {MAX_LINHAS}.
- Nota vai de 0 a 10; para média use ROUND(AVG(nota), 2).
- Faltas: presencas_sala com presente = false.
- Matérias: turma_professores.materia (dadas pelos professores), notas_alunos.disciplina (notas) e presencas_sala.materia (faltas). O valor 'null' em materia significa vazio e deve ser ignorado.
- Perguntas do tipo "quais matérias/turmas/cargos existem?" se respondem com SELECT DISTINCT (use UNION para juntar tabelas), nunca com BUSCA.
- Considere apenas registros ativos (ativo = true) quando a pergunta falar de "ativos".
- Use a conversa anterior para entender "deles", "dela", "essa turma" etc.
"""

REGRAS_RESPOSTA = (
    "Você responde em português, de forma direta e organizada, usando APENAS o resultado "
    "da consulta fornecida. Não invente dados. Se o resultado estiver vazio, diga que não "
    "encontrou. Para listas, use um item por linha. Se o resultado foi limitado, avise."
)

PROIBIDAS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|copy|pg_sleep)\b",
    re.IGNORECASE,
)


def gerar_sql(pergunta, erro=None, sql_anterior=None):
    mensagens = [
        {"role": "system", "content": REGRAS_SQL},
        {
            "role": "user",
            "content": f"Conversa até agora:\n{texto_historico()}\n\nPergunta: {pergunta}",
        },
    ]
    if erro:
        mensagens.append({"role": "assistant", "content": sql_anterior})
        mensagens.append(
            {
                "role": "user",
                "content": f"Essa consulta deu erro: {erro}\nCorrija e responda só com o SQL (ou BUSCA).",
            }
        )
    resp = cliente.chat.completions.create(model=modelo, messages=mensagens, temperature=0)
    sql = resp.choices[0].message.content.strip()
    return re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()


def executar_sql(sql):
    """Executa só SELECT, em transação somente leitura e com limite de tempo."""
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql or not re.match(r"(?is)^(select|with)\b", sql) or PROIBIDAS.search(sql):
        raise ValueError("Só é permitida uma consulta SELECT.")
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET LOCAL statement_timeout = 15000")
        cur.execute(sql)
        colunas = [c[0] for c in cur.description]
        linhas = cur.fetchmany(MAX_LINHAS + 1)
    finally:
        conn.rollback()
        conn.close()
    return colunas, linhas[:MAX_LINHAS], len(linhas) > MAX_LINHAS


def tabela_texto(colunas, linhas):
    saida = [" | ".join(colunas)]
    for linha in linhas:
        saida.append(" | ".join("" if v is None else str(v) for v in linha))
    return "\n".join(saida)


def tentar_sql(pergunta):
    """Devolve a resposta (já mostrada na tela) ou None se deve usar a busca por significado."""
    sql = gerar_sql(pergunta)
    for tentativa in range(2):
        if sql.strip().upper().startswith("BUSCA"):
            if MOSTRAR_SQL:
                print("[sql] o modelo escolheu a busca por significado")
            return None
        try:
            colunas, linhas, truncado = executar_sql(sql)
            break
        except Exception as erro:
            motivo = str(erro).strip().splitlines()[0][:300]
            if MOSTRAR_SQL:
                print("[sql] erro:", motivo, "| consulta:", " ".join(sql.split()))
            if tentativa == 1:
                return None
            sql = gerar_sql(pergunta, motivo, sql)

    if MOSTRAR_SQL:
        print("[consulta]", " ".join(sql.split()))
    resultado = tabela_texto(colunas, linhas) if linhas else "(nenhuma linha)"
    aviso = f"\n(Resultado limitado às primeiras {MAX_LINHAS} linhas.)" if truncado else ""
    return transmitir(
        [
            {"role": "system", "content": REGRAS_RESPOSTA},
            {
                "role": "user",
                "content": (
                    f"Pergunta: {pergunta}\n\nConsulta executada:\n{sql}\n\n"
                    f"Resultado:\n{resultado}{aviso}"
                ),
            },
        ]
    )


# ---------- Caminho de busca por significado ----------
REGRAS = (
    "Você é um assistente que responde perguntas sobre os dados de um banco de dados escolar. "
    "Você conhece APENAS os dados abaixo e o contexto que vier em cada pergunta.\n\n"
    f"TABELAS DISPONÍVEIS:\n{NOMES}\n\n"
    f"CAMPOS DE CADA TABELA (use só se o usuário perguntar sobre uma tabela específica):\n{DETALHES}\n\n"
    "Se o usuário perguntar o que você tem, quais tabelas ou quais dados você tem, "
    "liste SOMENTE os nomes das tabelas, sem descrever os campos. "
    "Só descreva os campos de uma tabela se o usuário perguntar especificamente sobre ela. "
    "Não diga que tem outras tabelas além dessas. "
    "Para perguntas sobre registros específicos, use APENAS o contexto fornecido. "
    "Se a resposta não estiver no contexto, diga: "
    "'Não encontrei essa informação no banco de dados.'"
)


def buscar_e_responder(pergunta, texto):
    vetor = modelo_emb.encode([pergunta]).tolist()
    tipo_filtro = next(
        (t for t, padrao in FILTROS.items() if re.search(padrao, texto)), None
    )
    filtro = {"tipo": tipo_filtro} if tipo_filtro else None
    busca = colecao.query(query_embeddings=vetor, n_results=N_RESULTADOS, where=filtro)
    contexto = "\n".join(f"- {doc}" for doc in busca["documents"][0])
    return transmitir(
        [
            {"role": "system", "content": REGRAS},
            {
                "role": "user",
                "content": (
                    f"Conversa até agora:\n{texto_historico()}\n\n"
                    f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
                ),
            },
        ]
    )


# ---------- Loop principal ----------
print("Pronto! Digite sua pergunta (ou 'sair' para encerrar).")

while True:
    pergunta = input("\nVocê: ").strip()

    if pergunta.lower() in ("sair", "exit", "quit"):
        print("Encerrando.")
        break
    if not pergunta:
        continue

    texto = pergunta.lower()

    # 0) "quantos alunos tem?": conta direto no banco vetorial (sem OpenAI)
    casou = re.match(PADRAO_QUANTOS, texto)
    if casou:
        tipo = achar_tabela(casou.group("coisa"))
        if tipo:
            atalho(pergunta, contar(tipo))
            continue

    # 1) Pergunta sobre uma tabela específica: mostra os campos dela
    fala_de_tabela = re.search(r"\b(tabela|colunas?|campos?|atributos?)\b", texto) or any(
        info[0] in texto for info in CATALOGO.values() if "_" in info[0]
    )
    if fala_de_tabela:
        tipo = achar_tabela(texto)
        if tipo:
            atalho(pergunta, descrever(tipo))
            continue
        if re.search(r"\btabela\b", texto) and re.search(
            r"\b(tem|possui|cont[ée]m|colunas?|campos?|atributos?)\b", texto
        ):
            atalho(pergunta, "Não tenho essa tabela. As tabelas que tenho são:\n" + NOMES)
            continue

    # 2) Pergunta geral ("o que você tem?"): mostra só os nomes das tabelas
    if re.search(PADRAO_LISTA, texto):
        atalho(
            pergunta,
            "Eu tenho estas tabelas:\n" + NOMES
            + "\n\nPergunte 'o que tem na tabela X?' para ver os campos de uma delas.",
        )
        continue

    # 3) Contagens com filtro, médias, listas: o modelo monta o SQL e o banco calcula
    resposta = tentar_sql(pergunta)

    # 4) Textos livres ou se o SQL não serviu: busca por significado
    if resposta is None:
        resposta = buscar_e_responder(pergunta, texto)

    registrar(pergunta, resposta)
    