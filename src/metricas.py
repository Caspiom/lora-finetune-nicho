"""Metricas da extracao: validade do JSON, acerto por campo e acerto exato, com IC por bootstrap pareado."""
from __future__ import annotations

import random

from .dados import CATEGORIAS, PRIORIDADES, SETORES
from .prompt import CHAVES, ler_resposta

VALORES_PERMITIDOS = {"categoria": CATEGORIAS, "prioridade": PRIORIDADES, "setor": SETORES}


def pontuar(previsto_bruto: str | None, gabarito: dict) -> dict:
    lido = ler_resposta(previsto_bruto) if previsto_bruto is not None else None
    campos = {k: bool(lido) and lido[k] == gabarito[k] for k in CHAVES}
    no_esquema = lido is not None and all(lido[k] in VALORES_PERMITIDOS[k] for k in CHAVES)
    return {"valido": lido is not None, "no_esquema": no_esquema, **campos, "exato": all(campos.values())}


def resumir(pontuacoes: list[dict]) -> dict:
    n = len(pontuacoes)
    return {nome: round(sum(p[nome] for p in pontuacoes) / n, 3) for nome in ("valido", "no_esquema", *CHAVES, "exato")}


def diferenca_pareada(a: list[bool], b: list[bool], reamostragens: int = 5000, semente: int = 0) -> dict:
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    rng = random.Random(semente)
    medias = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reamostragens))
    return {"diff": round(sum(diffs) / len(diffs), 3),
            "ic95": [round(medias[int(0.025 * reamostragens)], 3), round(medias[int(0.975 * reamostragens) - 1], 3)],
            "vitorias": sum(d > 0 for d in diffs), "derrotas": sum(d < 0 for d in diffs)}
