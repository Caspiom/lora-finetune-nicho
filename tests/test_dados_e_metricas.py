import json
from collections import Counter

import pytest

from src.dados import (ABERTURAS, CATEGORIAS, MOLDES, PRIORIDADES, PROBLEMAS, SETOR_MOLDES, SETORES, TAMANHOS,
                       URGENCIA, carregar, gerar_conjunto, gravar_conjuntos)
from src.metricas import diferenca_pareada, pontuar, resumir
from src.prompt import ler_resposta, mensagens, resposta_esperada
from src.regras import prever_classe_majoritaria, prever_por_regras
from evals.run_eval import escolher_exemplos


def test_geracao_e_deterministica_e_tem_os_tamanhos_certos():
    for nome, n in TAMANHOS.items():
        a, b = gerar_conjunto(nome), gerar_conjunto(nome)
        assert a == b and len(a) == n


def test_arquivos_em_data_batem_com_o_gerador(tmp_path):
    gravar_conjuntos(tmp_path)
    for nome in TAMANHOS:
        assert carregar(nome, tmp_path) == carregar(nome)


def test_teste_novo_nao_divide_vocabulario_nem_moldes_com_o_treino():
    for tabela in (PROBLEMAS, URGENCIA):
        treino = {frase for grupo in tabela["treino"].values() for frase in grupo}
        novo = {frase for grupo in tabela["novo"].values() for frase in grupo}
        assert not treino & novo
    for tabela in (ABERTURAS, SETOR_MOLDES, MOLDES):
        assert not set(tabela["treino"]) & set(tabela["novo"])


def test_nenhum_texto_do_teste_aparece_no_treino():
    treino = {c["texto"] for c in carregar("treino")} | {c["texto"] for c in carregar("validacao")}
    for nome in ("teste_igual", "teste_novo"):
        repetidos = [c["id"] for c in carregar(nome) if c["texto"] in treino]
        if nome == "teste_novo":
            assert repetidos == []
    ids = [c["id"] for nome in TAMANHOS for c in carregar(nome)]
    assert len(ids) == len(set(ids))


def test_rotulos_cobrem_todas_as_classes_sem_desbalanco_grosseiro():
    for nome in TAMANHOS:
        for campo, classes in (("categoria", CATEGORIAS), ("prioridade", PRIORIDADES), ("setor", SETORES)):
            contagem = Counter(c[campo] for c in carregar(nome))
            assert set(contagem) == set(classes)
            assert max(contagem.values()) < 2.2 * min(contagem.values())


def test_cada_texto_contem_o_problema_e_o_sinal_de_urgencia_do_rotulo():
    for nome, grupo in (("treino", "treino"), ("teste_novo", "novo")):
        for c in carregar(nome):
            assert any(f in c["texto"] for f in PROBLEMAS[grupo][c["categoria"]])
            assert any(f in c["texto"] for f in URGENCIA[grupo][c["prioridade"]])


@pytest.mark.parametrize("bruto,esperado", [
    ('{"categoria": "rede", "prioridade": "alta", "setor": "ti"}', {"categoria": "rede", "prioridade": "alta", "setor": "ti"}),
    ('```json\n{"categoria": "Rede", "prioridade": "ALTA", "setor": " ti "}\n```', {"categoria": "rede", "prioridade": "alta", "setor": "ti"}),
    ('Resposta: {"categoria": "rede", "prioridade": "alta", "setor": "ti"} fim', {"categoria": "rede", "prioridade": "alta", "setor": "ti"}),
])
def test_ler_resposta_aceita_formatos_comuns(bruto, esperado):
    assert ler_resposta(bruto) == esperado


@pytest.mark.parametrize("bruto", ["", "nao sei", "{quebrado", '{"categoria": "rede"}', '{"categoria": 1, "prioridade": "a", "setor": "b"}', "[1]"])
def test_ler_resposta_rejeita_o_que_nao_e_json_completo(bruto):
    assert ler_resposta(bruto) is None


def test_pontuar_conta_json_invalido_como_erro_em_todos_os_campos():
    gab = {"categoria": "rede", "prioridade": "alta", "setor": "ti"}
    p = pontuar("lixo", gab)
    assert p == {"valido": False, "no_esquema": False, "categoria": False, "prioridade": False, "setor": False,
                 "exato": False}
    assert pontuar(resposta_esperada(gab), gab)["exato"] is True
    parcial = pontuar('{"categoria": "rede", "prioridade": "baixa", "setor": "ti"}', gab)
    assert parcial["categoria"] and parcial["setor"] and not parcial["prioridade"] and not parcial["exato"]
    assert pontuar(None, gab)["valido"] is False


def test_rotulo_inventado_e_json_valido_mas_fora_do_esquema():
    gab = {"categoria": "rede", "prioridade": "alta", "setor": "ti"}
    p = pontuar('{"categoria": "video", "prioridade": "alta", "setor": "ti"}', gab)
    assert p["valido"] and not p["no_esquema"] and not p["categoria"] and p["prioridade"] and not p["exato"]
    assert pontuar(resposta_esperada(gab), gab)["no_esquema"]


def test_resumir_e_diferenca_pareada():
    assert resumir([{"valido": True, "no_esquema": True, "categoria": True, "prioridade": False, "setor": True, "exato": False},
                    {"valido": True, "no_esquema": False, "categoria": True, "prioridade": True, "setor": True, "exato": True}])["exato"] == 0.5
    d = diferenca_pareada([True] * 10, [False] * 10)
    assert d["diff"] == 1.0 and d["ic95"][0] > 0 and d["vitorias"] == 10
    assert diferenca_pareada([True, False], [True, False])["diff"] == 0
    assert diferenca_pareada([True, False, True], [False, False, True]) == diferenca_pareada([True, False, True], [False, False, True])


def test_regras_acertam_o_molde_do_treino_e_erram_o_vocabulario_novo():
    igual = [pontuar(prever_por_regras(c["texto"]), c)["exato"] for c in carregar("teste_igual")]
    novo = [pontuar(prever_por_regras(c["texto"]), c)["exato"] for c in carregar("teste_novo")]
    assert sum(igual) / len(igual) > 0.95 and sum(novo) / len(novo) < 0.3


def test_classe_majoritaria_devolve_sempre_a_mesma_resposta():
    prever = prever_classe_majoritaria(carregar("treino"))
    assert prever("a") == prever("b") and ler_resposta(prever("x")) is not None


def test_mensagens_com_exemplos_alternam_usuario_e_assistente_e_terminam_no_pedido():
    ex = carregar("treino")[:2]
    m = mensagens("texto novo", ex)
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant", "user"]
    assert m[-1]["content"].endswith("texto novo") and m[1]["content"] == resposta_esperada(ex[0])


def test_exemplos_do_few_shot_sao_do_treino_e_cobrem_categorias():
    treino = carregar("treino")
    ex = escolher_exemplos(treino)
    assert len(ex) == 8 and all(e in treino for e in ex) and ex == escolher_exemplos(treino)
    assert len({e["categoria"] for e in ex}) == 6 and len({e["prioridade"] for e in ex}) == 4
    assert len({e["id"] for e in ex}) == 8
