"""Decodificacao restrita ao esquema: o modelo so pode escolher entre as 144 respostas validas.

Constroi uma arvore (trie) com a tokenizacao das 6 x 4 x 6 respostas possiveis (`resposta_esperada` + fim de
turno), a MESMA tokenizacao usada no treino (`tokenizar_exemplo` tokeniza a resposta separada do pedido). A cada
passo, so os tokens que continuam algum caminho da arvore podem ser escolhidos; no fim da resposta so o token de fim.

O que isso garante: JSON valido e rotulo DENTRO do esquema (zera "rotulo inventado" por construcao).
O que NAO garante: que o rotulo escolhido esteja certo. A restricao escolhe o melhor caminho valido, nao o certo.
Limite: depende do modelo gerar com a tokenizacao canonica; um modelo que prefira outra segmentacao do mesmo
texto fica preso nos caminhos canonicos (e a avaliacao de validade mede isso, nao esconde).
"""
from __future__ import annotations

from itertools import product

from .dados import CATEGORIAS, PRIORIDADES, SETORES
from .prompt import resposta_esperada


class Arvore:
    def __init__(self, tok):
        self.eos = tok.eos_token_id
        self.filhos: dict[int, dict] = {}
        self.respostas: list[str] = []
        for c, p, s in product(CATEGORIAS, PRIORIDADES, SETORES):
            texto = resposta_esperada({"categoria": c, "prioridade": p, "setor": s})
            self.respostas.append(texto)
            ids = tok(texto, add_special_tokens=False)["input_ids"] + [self.eos]
            no = self.filhos
            for i in ids:
                no = no.setdefault(i, {})

    def permitidos(self, gerados: list[int]) -> list[int]:
        """Tokens validos depois do prefixo `gerados`. Prefixo fora da arvore ou resposta terminada: so o fim."""
        no = self.filhos
        for i in gerados:
            if i not in no:
                return [self.eos]
            no = no[i]
        return list(no) or [self.eos]

    def caminhos(self):
        """Todas as sequencias completas (sem o fim). Usado nos testes."""
        def dfs(no, prefixo):
            if not no:
                yield prefixo[:-1]  # tira o eos
                return
            for i, filho in no.items():
                yield from dfs(filho, prefixo + [i])
        yield from dfs(self.filhos, [])


def funcao_de_prefixo(arvore: Arvore, tamanho_do_pedido: int):
    """`prefix_allowed_tokens_fn` do `generate`: o pedido tem `tamanho_do_pedido` tokens (com padding a esquerda)."""
    def permitidos(_lote: int, ids) -> list[int]:
        return arvore.permitidos(ids[tamanho_do_pedido:].tolist())
    return permitidos
