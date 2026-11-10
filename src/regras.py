"""Baselines sem modelo: classe majoritaria e regras de palavras-chave escritas olhando SO o vocabulario do treino."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from .prompt import CHAVES

PALAVRAS_DA_CATEGORIA = {
    "acesso": ["senha", "login", "acesso", "bloqueado", "entrar no sistema"],
    "rede": ["internet", "wifi", "vpn", "rede"],
    "hardware": ["notebook", "monitor", "teclado", "mouse"],
    "software": ["programa", "sistema de notas", "antivirus", "aplicativo"],
    "email": ["email", "caixa de entrada", "calendario"],
    "impressora": ["impressora", "papel", "impressao"],
}
PALAVRAS_DA_PRIORIDADE = [
    ("critica", ["parou tudo", "ninguem consegue", "fora do ar", "cliente importante"]),
    ("alta", ["ate hoje", "entrega hoje", "atrapalha muito"]),
    ("baixa", ["sem pressa", "quando der", "nao e urgente", "so estou avisando"]),
]
PRIORIDADE_PADRAO = "media"
CATEGORIA_PADRAO = "software"
SETOR_PADRAO = "ti"


def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def prever_por_regras(texto: str) -> str:
    t = sem_acento(texto)
    categoria = next((c for c, palavras in PALAVRAS_DA_CATEGORIA.items() if any(p in t for p in palavras)),
                     CATEGORIA_PADRAO)
    prioridade = next((p for p, palavras in PALAVRAS_DA_PRIORIDADE if any(x in t for x in palavras)),
                      PRIORIDADE_PADRAO)
    setor = next((s for s in ("financeiro", "rh", "comercial", "juridico", "logistica")
                  if re.search(rf"\b{s}\b", t)), SETOR_PADRAO)
    return json.dumps({"categoria": categoria, "prioridade": prioridade, "setor": setor})


def prever_classe_majoritaria(treino: list[dict]):
    mais_comum = {k: Counter(c[k] for c in treino).most_common(1)[0][0] for k in CHAVES}
    resposta = json.dumps(mais_comum)
    return lambda texto: resposta
