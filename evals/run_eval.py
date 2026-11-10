"""Compara 5 sistemas na extracao de categoria/prioridade/setor de chamados de TI, em dois testes:
  teste_igual: mesmos moldes do treino      teste_novo: frases, aberturas e vocabulario que o treino nunca viu

Sistemas: classe majoritaria, regras de palavras-chave, Qwen base zero-shot, Qwen base few-shot (8 exemplos)
e Qwen + LoRA (uma linha por semente de treino). Grava resultados/metricas.json e resultados/previsoes.jsonl.

Uso: python -m evals.run_eval --adaptadores adaptadores/semente1 adaptadores/semente2 adaptadores/semente3"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.dados import CATEGORIAS, PRIORIDADES, carregar
from src.metricas import diferenca_pareada, pontuar, resumir
from src.regras import prever_classe_majoritaria, prever_por_regras

TESTES = ("teste_igual", "teste_novo")
EXEMPLOS_NO_PROMPT = 8


def escolher_exemplos(treino: list[dict], n: int = EXEMPLOS_NO_PROMPT, semente: int = 0) -> list[dict]:
    rng = random.Random(semente)
    embaralhado = rng.sample(treino, len(treino))
    cobertura, escolhidos = set(), []
    for c in embaralhado:
        chaves = {("categoria", c["categoria"]), ("prioridade", c["prioridade"])}
        if len(escolhidos) < n and not chaves <= cobertura:
            escolhidos.append(c)
            cobertura |= chaves
    sobras = [c for c in embaralhado if c not in escolhidos]
    return escolhidos + sobras[:n - len(escolhidos)]


def avaliar_sistema(nome: str, previsoes: list[str], chamados: list[dict], extras: dict | None = None) -> dict:
    pontos = [pontuar(p, c) for p, c in zip(previsoes, chamados)]
    return {"sistema": nome, **resumir(pontos), **(extras or {}), "_exato": [p["exato"] for p in pontos],
            "_previsoes": previsoes}


def rodar_modelos(args, treino, testes):
    from src.modelo import carregar_modelo, carregar_tokenizador, gerar
    tok = carregar_tokenizador()
    exemplos = escolher_exemplos(treino)
    saida = {}
    base = carregar_modelo()
    for nome, ex in (("base_zero_shot", None), ("base_few_shot_8", exemplos)):
        for teste, chamados in testes.items():
            previsoes, tempo = gerar(base, tok, [c["texto"] for c in chamados], exemplos=ex)
            saida[(nome, teste)] = avaliar_sistema(nome, previsoes, chamados, tempo)
    del base
    import torch
    torch.cuda.empty_cache()
    for adaptador in args.adaptadores:
        modelo = carregar_modelo(adaptador)
        nome = f"lora_{Path(adaptador).name}"
        for teste, chamados in testes.items():
            previsoes, tempo = gerar(modelo, tok, [c["texto"] for c in chamados])
            saida[(nome, teste)] = avaliar_sistema(nome, previsoes, chamados, tempo)
        del modelo
        torch.cuda.empty_cache()
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptadores", nargs="*", default=[])
    ap.add_argument("--sem-modelo", action="store_true", help="so os baselines sem modelo")
    args = ap.parse_args()
    treino = carregar("treino")
    testes = {t: carregar(t) for t in TESTES}
    resultados = {}
    majoritaria = prever_classe_majoritaria(treino)
    for teste, chamados in testes.items():
        for nome, prever in (("classe_majoritaria", majoritaria), ("regras_palavras_chave", prever_por_regras)):
            resultados[(nome, teste)] = avaliar_sistema(nome, [prever(c["texto"]) for c in chamados], chamados)
    if not args.sem_modelo:
        resultados.update(rodar_modelos(args, treino, testes))
    gravar(resultados, testes)


def gravar(resultados: dict, testes: dict) -> None:
    Path("resultados").mkdir(exist_ok=True)
    nomes = list(dict.fromkeys(n for n, _ in resultados))
    lora = [n for n in nomes if n.startswith("lora_")]
    resumo = {t: {n: {k: v for k, v in resultados[(n, t)].items() if not k.startswith("_")} for n in nomes} for t in TESTES}
    for t in TESTES:
        if lora:
            media = {k: round(sum(resultados[(n, t)][k] for n in lora) / len(lora), 3) for k in ("valido", "no_esquema", "categoria", "prioridade", "setor", "exato")}
            minimo = {k: min(resultados[(n, t)][k] for n in lora) for k in ("exato",)}
            maximo = {k: max(resultados[(n, t)][k] for n in lora) for k in ("exato",)}
            resumo[t]["lora_media_das_sementes"] = {**media, "exato_min": minimo["exato"], "exato_max": maximo["exato"], "n_sementes": len(lora)}
            for outro in ("base_few_shot_8", "regras_palavras_chave"):
                if (outro, t) in resultados:
                    resumo[t][f"lora_{lora[0].split('_', 1)[1]}_menos_{outro}"] = diferenca_pareada(
                        resultados[(lora[0], t)]["_exato"], resultados[(outro, t)]["_exato"])
    Path("resultados/metricas.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    with open("resultados/previsoes.jsonl", "w") as f:
        for (n, t), r in resultados.items():
            for c, p, ok in zip(testes[t], r["_previsoes"], r["_exato"]):
                f.write(json.dumps({"sistema": n, "teste": t, "id": c["id"], "previsto": p, "exato": ok}, ensure_ascii=False) + "\n")
    imprimir(resumo)


def imprimir(resumo: dict) -> None:
    for t, sistemas in resumo.items():
        print(f"\n== {t}")
        print(f"{'sistema':<30} {'valido':>7} {'esquema':>8} {'categ':>7} {'prior':>7} {'setor':>7} {'exato':>7}")
        for n, r in sistemas.items():
            if "exato" in r and "valido" in r:
                print(f"{n:<30} {r['valido']:>7} {r['no_esquema']:>8} {r['categoria']:>7} {r['prioridade']:>7} {r['setor']:>7} {r['exato']:>7}")
            else:
                print(f"{n:<30} IC95% da diferenca de acerto exato: {r['diff']:+.3f} {r['ic95']}  vitorias {r['vitorias']} derrotas {r['derrotas']}")


if __name__ == "__main__":
    main()
