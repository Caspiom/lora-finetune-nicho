"""Treino LoRA: perda so nos tokens da resposta, laco simples em PyTorch.

Uso: python -m src.treinar --saida adaptadores/semente1 --semente 1"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .dados import carregar
from .modelo import carregar_modelo, carregar_tokenizador, dispositivo
from .prompt import mensagens, resposta_esperada

MODULOS_ALVO = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CONFIG_LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=MODULOS_ALVO, task_type="CAUSAL_LM")


def tokenizar_exemplo(tok, chamado: dict, comprimento_maximo: int = 256) -> dict:
    pedido = tok.apply_chat_template(mensagens(chamado["texto"]), tokenize=False, add_generation_prompt=True)
    resposta = resposta_esperada(chamado) + tok.eos_token
    ids_pedido = tok(pedido, add_special_tokens=False)["input_ids"]
    ids_resposta = tok(resposta, add_special_tokens=False)["input_ids"]
    ids = (ids_pedido + ids_resposta)[:comprimento_maximo]
    rotulos = ([-100] * len(ids_pedido) + ids_resposta)[:comprimento_maximo]
    return {"input_ids": ids, "labels": rotulos}


def montar_lote(exemplos: list[dict], id_de_preenchimento: int, aparelho: str) -> dict:
    maior = max(len(e["input_ids"]) for e in exemplos)
    ids = [e["input_ids"] + [id_de_preenchimento] * (maior - len(e["input_ids"])) for e in exemplos]
    rot = [e["labels"] + [-100] * (maior - len(e["labels"])) for e in exemplos]
    mascara = [[1] * len(e["input_ids"]) + [0] * (maior - len(e["input_ids"])) for e in exemplos]
    return {"input_ids": torch.tensor(ids, device=aparelho), "labels": torch.tensor(rot, device=aparelho),
            "attention_mask": torch.tensor(mascara, device=aparelho)}


def perda_media(modelo, lotes: list[dict]) -> float:
    modelo.eval()
    with torch.no_grad():
        return sum(float(modelo(**l).loss) for l in lotes) / len(lotes)


def acumular_gradientes(modelo, grupo: list[dict], micro_lote: int, id_de_preenchimento: int, aparelho: str) -> float:
    perda_do_passo = 0.0
    for j in range(0, len(grupo), micro_lote):
        parte = grupo[j:j + micro_lote]
        perda = modelo(**montar_lote(parte, id_de_preenchimento, aparelho)).loss * len(parte) / len(grupo)
        perda.backward()
        perda_do_passo += float(perda.detach())
    return perda_do_passo


def treinar(saida: str, semente: int, epocas: int = 2, tamanho_do_lote: int = 8, micro_lote: int = 2, taxa: float = 2e-4,
            modelo=None, tok=None, treino=None, validacao=None) -> dict:
    random.seed(semente)
    torch.manual_seed(semente)
    tok = tok or carregar_tokenizador()
    base = modelo or carregar_modelo()
    modelo = get_peft_model(base, LoraConfig(**CONFIG_LORA))
    aparelho = dispositivo()
    treino = treino if treino is not None else carregar("treino")
    validacao = validacao if validacao is not None else carregar("validacao")
    tok_treino = [tokenizar_exemplo(tok, c) for c in treino]
    lotes_validacao = [montar_lote([tokenizar_exemplo(tok, c) for c in validacao[i:i + 4]], tok.pad_token_id, aparelho)
                       for i in range(0, len(validacao), 4)]
    otimizador = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=taxa, weight_decay=0.0)
    passos_totais = epocas * ((len(tok_treino) + tamanho_do_lote - 1) // tamanho_do_lote)
    agenda = torch.optim.lr_scheduler.LambdaLR(otimizador, lambda p: min(1.0, (p + 1) / 10) * max(0.0, 1 - p / passos_totais))
    historico = {"perda_validacao_inicial": round(perda_media(modelo, lotes_validacao), 4), "perda_treino": []}
    if aparelho == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    passo = 0
    for _ in range(epocas):
        random.shuffle(tok_treino)
        modelo.train()
        for i in range(0, len(tok_treino), tamanho_do_lote):
            grupo = tok_treino[i:i + tamanho_do_lote]
            perda_do_passo = acumular_gradientes(modelo, grupo, micro_lote, tok.pad_token_id, aparelho)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agenda.step()
            otimizador.zero_grad(set_to_none=True)
            passo += 1
            if passo % 10 == 0:
                historico["perda_treino"].append(round(perda_do_passo, 4))
    historico.update(
        perda_validacao_final=round(perda_media(modelo, lotes_validacao), 4),
        segundos_de_treino=round(time.perf_counter() - inicio, 1), passos=passo, semente=semente,
        memoria_maxima_mb=round(torch.cuda.max_memory_allocated() / 2**20) if aparelho == "cuda" else None,
        parametros_treinaveis=sum(p.numel() for p in modelo.parameters() if p.requires_grad))
    Path(saida).mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(saida)
    historico["tamanho_do_adaptador_mb"] = round(sum(f.stat().st_size for f in Path(saida).glob("*.safetensors")) / 2**20, 1)
    Path(saida, "historico.json").write_text(json.dumps(historico, indent=2))
    return historico


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--epocas", type=int, default=2)
    args = ap.parse_args()
    print(json.dumps(treinar(args.saida, args.semente, args.epocas), indent=2))
