"""A fiacao do --restrito no avaliador: a flag chega ao `gerar` e os resultados vao para a pasta certa."""
import sys
from types import SimpleNamespace

from evals import run_eval
from src.dados import carregar


def test_rodar_modelos_repassa_restrito_para_todo_sistema(monkeypatch):
    chamadas = []
    monkeypatch.setattr("src.modelo.carregar_tokenizador", lambda: None)
    monkeypatch.setattr("src.modelo.carregar_modelo", lambda adaptador=None: object())

    def falso(modelo, tok, textos, exemplos=None, restrito=False, **kw):
        chamadas.append(restrito)
        return ['{"categoria":"rede","prioridade":"alta","setor":"ti"}'] * len(textos), {}

    monkeypatch.setattr("src.modelo.gerar", falso)
    treino, testes = carregar("treino"), {"teste_igual": carregar("teste_igual")[:5]}
    for flag in (True, False):
        chamadas.clear()
        run_eval.rodar_modelos(SimpleNamespace(restrito=flag, adaptadores=["a1"]), treino, testes)
        assert chamadas == [flag] * 3  # zero-shot, few-shot e o adaptador


def test_main_grava_restrito_em_pasta_propria_e_normal_na_padrao(monkeypatch):
    pastas = []
    monkeypatch.setattr(run_eval, "gravar", lambda resultados, testes, pasta="resultados": pastas.append(pasta))
    for argv, esperada in ((["x", "--sem-modelo", "--restrito"], "resultados/restrito"), (["x", "--sem-modelo"], "resultados")):
        monkeypatch.setattr(sys, "argv", argv)
        run_eval.main()
        assert pastas[-1] == esperada


def test_gravar_escreve_na_pasta_pedida_e_nao_na_padrao(tmp_path, monkeypatch):
    testes = {t: carregar(t)[:3] for t in run_eval.TESTES}  # antes do chdir: `carregar` usa caminho relativo
    monkeypatch.chdir(tmp_path)
    res = {(n, t): run_eval.avaliar_sistema(n, ['{"categoria":"rede","prioridade":"alta","setor":"ti"}'] * 3, testes[t])
           for n in ("regras_palavras_chave",) for t in testes}
    run_eval.gravar(res, testes, str(tmp_path / "saida" / "restrito"))
    assert (tmp_path / "saida" / "restrito" / "metricas.json").exists()
    assert (tmp_path / "saida" / "restrito" / "previsoes.jsonl").exists()
    assert not (tmp_path / "resultados").exists()
