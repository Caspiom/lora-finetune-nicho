"""Prompt da tarefa e leitura da resposta do modelo."""
from __future__ import annotations

import json

from .dados import CATEGORIAS, PRIORIDADES, SETORES

CHAVES = ("categoria", "prioridade", "setor")

INSTRUCAO = (
    "Leia o chamado de suporte de TI e responda SOMENTE com um JSON com as chaves "
    f"categoria ({'|'.join(CATEGORIAS)}), prioridade ({'|'.join(PRIORIDADES)}) "
    f"e setor ({'|'.join(SETORES)}).\n\nChamado: "
)


def pedido(texto: str) -> str:
    return INSTRUCAO + texto


def resposta_esperada(chamado: dict) -> str:
    return json.dumps({k: chamado[k] for k in CHAVES}, ensure_ascii=False)


def mensagens(texto: str, exemplos: list[dict] | None = None) -> list[dict]:
    historico = []
    for e in exemplos or []:
        historico += [{"role": "user", "content": pedido(e["texto"])},
                      {"role": "assistant", "content": resposta_esperada(e)}]
    return historico + [{"role": "user", "content": pedido(texto)}]


def ler_resposta(bruto: str) -> dict | None:
    """Devolve o dicionario com as 3 chaves ou None se a resposta nao for um JSON valido com elas."""
    inicio, fim = bruto.find("{"), bruto.rfind("}")
    if inicio == -1 or fim <= inicio:
        return None
    try:
        dado = json.loads(bruto[inicio:fim + 1])
    except ValueError:
        return None
    if not isinstance(dado, dict) or not all(isinstance(dado.get(k), str) for k in CHAVES):
        return None
    return {k: dado[k].strip().lower() for k in CHAVES}
