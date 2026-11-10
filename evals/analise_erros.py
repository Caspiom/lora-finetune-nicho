"""Onde o LoRA erra no teste_novo: acerto por frase de problema e por frase de urgencia, e pares de confusao.
Le resultados/previsoes.jsonl (gerado pelo run_eval). Uso: python -m evals.analise_erros"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.dados import PROBLEMAS, URGENCIA, carregar
from src.prompt import ler_resposta


def analisar() -> dict:
    gabarito = {c["id"]: c for c in carregar("teste_novo")}
    por_problema, por_urgencia, confusoes = defaultdict(list), defaultdict(list), Counter()
    for linha in Path("resultados/previsoes.jsonl").read_text().splitlines():
        r = json.loads(linha)
        if r["teste"] != "teste_novo" or not r["sistema"].startswith("lora_"):
            continue
        c, lido = gabarito[r["id"]], ler_resposta(r["previsto"]) or {}
        frase = next(f for f in PROBLEMAS["novo"][c["categoria"]] if f in c["texto"])
        urg = next(f for f in URGENCIA["novo"][c["prioridade"]] if f in c["texto"])
        por_problema[(c["categoria"], frase)].append(lido.get("categoria") == c["categoria"])
        por_urgencia[(c["prioridade"], urg)].append(lido.get("prioridade") == c["prioridade"])
        if lido.get("categoria") != c["categoria"]:
            confusoes[(c["categoria"], lido.get("categoria"))] += 1
    taxa = lambda xs: round(sum(xs) / len(xs), 3)
    return {
        "categoria_por_frase_nova": {f"{c}: {f}": taxa(v) for (c, f), v in sorted(por_problema.items())},
        "prioridade_por_frase_nova": {f"{p}: {f}": taxa(v) for (p, f), v in sorted(por_urgencia.items())},
        "confusoes_de_categoria": {f"{a} -> {b}": n for (a, b), n in confusoes.most_common(8)},
    }


if __name__ == "__main__":
    print(json.dumps(analisar(), indent=2, ensure_ascii=False))
