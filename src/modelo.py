"""Carregar o modelo base (com ou sem adaptador LoRA) e gerar respostas em lote."""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import mensagens

MODELO_BASE = os.environ.get("MODELO_BASE", "Qwen/Qwen2.5-0.5B-Instruct")


def dispositivo() -> str:
    return os.environ.get("DISPOSITIVO") or ("cuda" if torch.cuda.is_available() else "cpu")


def carregar_tokenizador():
    tok = AutoTokenizer.from_pretrained(MODELO_BASE)
    tok.padding_side = "left"
    return tok


def carregar_modelo(adaptador: str | None = None):
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, dtype=torch.bfloat16).to(dispositivo())
    if adaptador:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, adaptador)
    return modelo.eval()


def texto_do_pedido(tok, texto: str, exemplos: list[dict] | None = None) -> str:
    return tok.apply_chat_template(mensagens(texto, exemplos), tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gerar(modelo, tok, textos: list[str], exemplos: list[dict] | None = None, lote: int = 16,
          max_novos_tokens: int = 48) -> tuple[list[str], dict]:
    saidas, tokens_gerados = [], 0
    inicio = time.perf_counter()
    for i in range(0, len(textos), lote):
        pedidos = [texto_do_pedido(tok, t, exemplos) for t in textos[i:i + lote]]
        entrada = tok(pedidos, return_tensors="pt", padding=True).to(modelo.device)
        gerado = modelo.generate(**entrada, max_new_tokens=max_novos_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        novos = gerado[:, entrada["input_ids"].shape[1]:]
        tokens_gerados += int((novos != tok.pad_token_id).sum())
        saidas += tok.batch_decode(novos, skip_special_tokens=True)
    segundos = time.perf_counter() - inicio
    return saidas, {"segundos": round(segundos, 2), "ms_por_exemplo": round(1000 * segundos / len(textos), 1),
                    "tokens_gerados": tokens_gerados}
