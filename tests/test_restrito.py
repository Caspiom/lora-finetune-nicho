import json
from itertools import product

import pytest
import torch
from tests.test_treino import modelo_minusculo, tok  # noqa: F401  (fixture do tokenizador)

from src.dados import CATEGORIAS, PRIORIDADES, SETORES
from src.modelo import gerar
from src.prompt import ler_resposta
from src.restrito import Arvore


@pytest.fixture(scope="module")
def arvore(tok):
    return Arvore(tok)


def test_a_arvore_aceita_exatamente_as_144_respostas_validas(arvore, tok):
    decodificadas = {tok.decode(c) for c in arvore.caminhos()}
    assert len(decodificadas) == len(CATEGORIAS) * len(PRIORIDADES) * len(SETORES) == 144
    esperadas = {json.dumps({"categoria": c, "prioridade": p, "setor": s}, ensure_ascii=False)
                 for c, p, s in product(CATEGORIAS, PRIORIDADES, SETORES)}
    assert decodificadas == esperadas  # nenhuma a mais (rotulo inventado) e nenhuma a menos


def test_cada_resposta_valida_e_aceita_passo_a_passo_e_termina_no_fim(arvore, tok):
    for texto in arvore.respostas[::17]:
        ids = tok(texto, add_special_tokens=False)["input_ids"]
        for i, token in enumerate(ids):
            assert token in arvore.permitidos(ids[:i])
        assert arvore.permitidos(ids) == [tok.eos_token_id]  # resposta completa: so o fim


def test_token_fora_da_arvore_e_barrado_em_todos_os_passos(arvore, tok):
    inventado = tok("{\"categoria\": \"nuvem", add_special_tokens=False)["input_ids"]
    permitidos = [i for i in range(len(inventado)) if inventado[i] not in arvore.permitidos(inventado[:i])]
    assert permitidos, "o rotulo inventado precisa ser barrado em algum passo"


def test_sempre_ha_ao_menos_um_token_permitido(arvore):
    assert arvore.permitidos([]) and arvore.permitidos([10**9]) == [arvore.eos]


def test_modelo_aleatorio_com_restricao_sempre_responde_dentro_do_esquema(tok):
    modelo = modelo_minusculo(tok).eval()
    textos = [f"chamado numero {i} do setor" for i in range(6)]
    livre, _ = gerar(modelo, tok, textos, max_novos_tokens=40)
    restrito, _ = gerar(modelo, tok, textos, max_novos_tokens=40, restrito=True)
    lidas = [ler_resposta(r) for r in restrito]
    assert all(l is not None for l in lidas)
    assert all(l["categoria"] in CATEGORIAS and l["prioridade"] in PRIORIDADES and l["setor"] in SETORES for l in lidas)
    assert any(ler_resposta(r) is None for r in livre), "controle: sem restricao o modelo aleatorio NAO acerta o formato"


def test_restricao_funciona_com_padding_a_esquerda_em_lote_de_textos_de_tamanhos_diferentes(tok):
    modelo = modelo_minusculo(tok).eval()
    textos = ["curto", "um chamado bem mais longo " * 8]
    sozinho = [gerar(modelo, tok, [t], max_novos_tokens=40, restrito=True)[0][0] for t in textos]
    junto, _ = gerar(modelo, tok, textos, max_novos_tokens=40, restrito=True)
    assert all(ler_resposta(r) is not None for r in junto + sozinho)
