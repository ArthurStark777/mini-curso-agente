import os
from decimal import Decimal

import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv(override=True)

# "todas" (ou vazio) = todas as escolas; um UUID = só aquela escola
EMPRESA_ID = os.getenv("EMPRESA_ID") or "todas"
print("Escopo:", "todas as escolas" if EMPRESA_ID == "todas" else f"escola {EMPRESA_ID}")

FONTES = {
    "alunos": {
        "rotulo": "Cadastro de aluno",
        "sql": """
            SELECT a.id, a.empresa_id, a.id AS aluno_id, a.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   a.nome, a.matricula, t.nome AS turma,
                   CAST(t.turno AS text) AS turno, t.ano_letivo, a.ativo
            FROM alunos a
            LEFT JOIN empresas e ON e.id = a.empresa_id
            LEFT JOIN turmas t ON t.id = a.turma_id
            WHERE (:empresa_id = 'todas' OR CAST(a.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Aluno", "nome"), ("Matrícula", "matricula"),
                   ("Turma", "turma"), ("Turno", "turno"), ("Ano letivo", "ano_letivo"),
                   ("Ativo", "ativo")],
    },
    "observacoes": {
        "rotulo": "Observação sobre aluno",
        "sql": """
            SELECT o.id, o.empresa_id, o.aluno_id, a.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   a.nome AS aluno, t.nome AS turma,
                   o.titulo, o.texto, o.autor_nome, CAST(o.created_at AS date) AS data
            FROM observacoes_alunos o
            JOIN alunos a ON a.id = o.aluno_id
            LEFT JOIN empresas e ON e.id = o.empresa_id
            LEFT JOIN turmas t ON t.id = a.turma_id
            WHERE (:empresa_id = 'todas' OR CAST(o.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Aluno", "aluno"), ("Turma", "turma"),
                   ("Data", "data"), ("Título", "titulo"), ("Texto", "texto"),
                   ("Autor", "autor_nome")],
    },
    "avisos": {
        "rotulo": "Aviso da escola",
        "sql": """
            SELECT av.id, av.empresa_id, av.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   t.nome AS turma,
                   av.titulo, av.mensagem, CAST(av.publicado_em AS date) AS data
            FROM avisos_escola av
            LEFT JOIN empresas e ON e.id = av.empresa_id
            LEFT JOIN turmas t ON t.id = av.turma_id
            WHERE av.ativo
              AND (:empresa_id = 'todas' OR CAST(av.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Título", "titulo"), ("Mensagem", "mensagem"),
                   ("Turma", "turma"), ("Data", "data")],
    },
    "notas": {
        "rotulo": "Nota de aluno",
        "sql": """
            SELECT n.id, n.empresa_id, n.aluno_id, a.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   a.nome AS aluno, t.nome AS turma,
                   n.disciplina, n.etapa, n.bimestre, n.tipo_avaliacao,
                   n.atividade, n.nota, n.observacao
            FROM notas_alunos n
            JOIN alunos a ON a.id = n.aluno_id
            LEFT JOIN empresas e ON e.id = n.empresa_id
            LEFT JOIN turmas t ON t.id = a.turma_id
            WHERE (:empresa_id = 'todas' OR CAST(n.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Aluno", "aluno"), ("Turma", "turma"),
                   ("Disciplina", "disciplina"), ("Etapa", "etapa"),
                   ("Bimestre", "bimestre"), ("Avaliação", "tipo_avaliacao"),
                   ("Atividade", "atividade"), ("Nota", "nota"),
                   ("Observação", "observacao")],
    },
    "faltas": {
        "rotulo": "Falta de aluno",
        "sql": """
            SELECT p.id, p.empresa_id, p.aluno_id, p.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   a.nome AS aluno, t.nome AS turma,
                   p.materia, p.data, p.falta_justificada, p.justificativa, p.observacao
            FROM presencas_sala p
            JOIN alunos a ON a.id = p.aluno_id
            JOIN turmas t ON t.id = p.turma_id
            LEFT JOIN empresas e ON e.id = p.empresa_id
            WHERE NOT p.presente
              AND (:empresa_id = 'todas' OR CAST(p.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Aluno", "aluno"), ("Turma", "turma"),
                   ("Matéria", "materia"), ("Data", "data"),
                   ("Falta justificada", "falta_justificada"),
                   ("Justificativa", "justificativa"), ("Observação", "observacao")],
    },
    "turmas": {
        "rotulo": "Turma",
        "sql": """
            SELECT t.id, t.empresa_id, t.id AS turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   f.nome AS unidade,
                   t.nome, CAST(t.turno AS text) AS turno, t.ano_letivo, t.ativo
            FROM turmas t
            LEFT JOIN empresas e ON e.id = t.empresa_id
            LEFT JOIN filiais f ON f.id = t.filial_id
            WHERE (:empresa_id = 'todas' OR CAST(t.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Unidade", "unidade"), ("Turma", "nome"),
                   ("Turno", "turno"), ("Ano letivo", "ano_letivo"), ("Ativa", "ativo")],
    },
    "professores": {
        "rotulo": "Professor da turma",
        "sql": """
            SELECT tp.id, tp.empresa_id, tp.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   t.nome AS turma, CAST(t.turno AS text) AS turno,
                   u.nome AS professor, tp.materia,
                   (SELECT string_agg(d, ', ')
                      FROM jsonb_array_elements_text(tp.dias_semana) AS d) AS dias,
                   to_char(tp.hora_inicio, 'HH24:MI') AS inicio,
                   to_char(tp.hora_fim, 'HH24:MI') AS fim
            FROM turma_professores tp
            JOIN turmas t ON t.id = tp.turma_id
            JOIN usuarios u ON u.id = tp.professor_id
            LEFT JOIN empresas e ON e.id = tp.empresa_id
            WHERE tp.ativo
              AND (:empresa_id = 'todas' OR CAST(tp.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Turma", "turma"), ("Turno", "turno"),
                   ("Professor", "professor"), ("Matéria", "materia"),
                   ("Dias da semana", "dias"), ("Início", "inicio"), ("Fim", "fim")],
    },
    "horarios": {
        "rotulo": "Horário da turma",
        "sql": """
            SELECT h.id, h.empresa_id, h.turma_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   t.nome AS turma, CAST(t.turno AS text) AS turno,
                   h.dia_semana,
                   to_char(h.hora_entrada, 'HH24:MI') AS entrada,
                   to_char(h.hora_saida, 'HH24:MI') AS saida
            FROM horarios_turmas h
            JOIN turmas t ON t.id = h.turma_id
            LEFT JOIN empresas e ON e.id = h.empresa_id
            WHERE h.ativo
              AND (:empresa_id = 'todas' OR CAST(h.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Turma", "turma"), ("Turno", "turno"),
                   ("Dia da semana", "dia_semana"), ("Entrada", "entrada"),
                   ("Saída", "saida")],
    },
    "feriados": {
        "rotulo": "Feriado",
        "sql": """
            SELECT f.id, f.empresa_id, f.data, f.descricao,
                   CAST(f.abrangencia AS text) AS abrangencia
            FROM feriados f
            WHERE (:empresa_id = 'todas' OR f.empresa_id IS NULL
                   OR CAST(f.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Data", "data"), ("Descrição", "descricao"),
                   ("Abrangência", "abrangencia")],
    },
    "escolas": {
        "rotulo": "Escola",
        "sql": """
            SELECT e.id, e.id AS empresa_id,
                   e.razao_social, e.nome_fantasia, e.endereco, e.telefone, e.ativo
            FROM empresas e
            WHERE (:empresa_id = 'todas' OR CAST(e.id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "nome_fantasia"), ("Razão social", "razao_social"),
                   ("Endereço", "endereco"), ("Telefone", "telefone"), ("Ativa", "ativo")],
    },
    "unidades": {
        "rotulo": "Unidade",
        "sql": """
            SELECT f.id, f.empresa_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   f.nome, f.endereco, CAST(f.tipo AS text) AS tipo, f.ativo
            FROM filiais f
            LEFT JOIN empresas e ON e.id = f.empresa_id
            WHERE (:empresa_id = 'todas' OR CAST(f.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Unidade", "nome"), ("Endereço", "endereco"),
                   ("Tipo", "tipo"), ("Ativa", "ativo")],
    },
    "departamentos": {
        "rotulo": "Departamento",
        "sql": """
            SELECT d.id, d.empresa_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   d.nome, d.ativo
            FROM departamentos d
            LEFT JOIN empresas e ON e.id = d.empresa_id
            WHERE (:empresa_id = 'todas' OR CAST(d.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Departamento", "nome"), ("Ativo", "ativo")],
    },
    "funcionarios": {
        "rotulo": "Funcionário",
        "sql": """
            SELECT fu.id, fu.empresa_id,
                   COALESCE(e.nome_fantasia, e.razao_social) AS escola,
                   fi.nome AS unidade,
                   d.nome AS departamento,
                   fu.nome, fu.matricula, fu.cargo,
                   fu.data_admissao, fu.data_demissao, fu.ativo
            FROM funcionarios fu
            LEFT JOIN empresas e ON e.id = fu.empresa_id
            LEFT JOIN filiais fi ON fi.id = fu.filial_id
            LEFT JOIN departamentos d ON d.id = fu.departamento_id
            WHERE (:empresa_id = 'todas' OR CAST(fu.empresa_id AS text) = :empresa_id)
        """,
        "campos": [("Escola", "escola"), ("Funcionário", "nome"),
                   ("Matrícula", "matricula"), ("Cargo", "cargo"),
                   ("Departamento", "departamento"), ("Unidade", "unidade"),
                   ("Admissão", "data_admissao"), ("Demissão", "data_demissao"),
                   ("Ativo", "ativo")],
    },
}


def formatar(valor):
    if isinstance(valor, bool):
        return "sim" if valor else "não"
    if isinstance(valor, Decimal):
        valor = float(valor)
    if isinstance(valor, float):
        return str(int(valor)) if valor.is_integer() else f"{valor:g}"
    return str(valor).strip()


def montar_texto(registro, rotulo, campos):
    partes = []
    for nome_campo, coluna in campos:
        valor = registro.get(coluna)
        if valor is None or pd.isna(valor):
            continue
        valor = formatar(valor)
        if valor.lower() in ("", "null"):
            continue
        partes.append(f"{nome_campo}: {valor}")
    return f"{rotulo}. " + " | ".join(partes) if partes else ""


engine = create_engine(os.getenv("DATABASE_URL"))

ids, textos, metadados = [], [], []
for nome, fonte in FONTES.items():
    df = pd.read_sql(text(fonte["sql"]), engine, params={"empresa_id": EMPRESA_ID})
    print(f"{nome}: {len(df)} linhas")
    for reg in df.to_dict("records"):
        texto = montar_texto(reg, fonte["rotulo"], fonte["campos"])
        if not texto:
            continue
        meta = {"tipo": nome}
        for col in ("empresa_id", "aluno_id", "turma_id"):
            if reg.get(col) is not None and not pd.isna(reg[col]):
                meta[col] = str(reg[col])
        ids.append(f"{nome}:{reg['id']}")
        textos.append(texto)
        metadados.append(meta)

if not textos:
    raise SystemExit("Nenhuma linha encontrada. Confira o EMPRESA_ID.")

print("Exemplo de texto:", textos[0])

modelo_emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = modelo_emb.encode(textos, batch_size=32, show_progress_bar=True).tolist()

chroma = chromadb.PersistentClient(path="chroma_db")
try:
    chroma.delete_collection("meu_banco")
except Exception:
    pass
colecao = chroma.create_collection("meu_banco")

TAMANHO_LOTE = 500
for i in range(0, len(ids), TAMANHO_LOTE):
    colecao.upsert(
        ids=ids[i : i + TAMANHO_LOTE],
        documents=textos[i : i + TAMANHO_LOTE],
        embeddings=embeddings[i : i + TAMANHO_LOTE],
        metadatas=metadados[i : i + TAMANHO_LOTE],
    )

print("Vetorização concluída:", colecao.count(), "itens na coleção")