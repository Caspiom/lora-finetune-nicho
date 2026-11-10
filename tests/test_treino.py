import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.dados import carregar
from src.modelo import carregar_tokenizador
from src.treinar import montar_lote, tokenizar_exemplo, treinar


@pytest.fixture(scope="module")
def tok():
    try:
        return carregar_tokenizador()
    except OSError:
        pytest.skip("tokenizador do Qwen nao esta no cache e nao ha rede")


def modelo_minusculo(tok):
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(config)


def test_perda_so_conta_os_tokens_da_resposta(tok):
    ex = tokenizar_exemplo(tok, carregar("treino")[0])
    mascarados = sum(r == -100 for r in ex["labels"])
    assert 0 < mascarados < len(ex["labels"])
    assert ex["labels"][:mascarados] == [-100] * mascarados
    resposta = tok.decode([r for r in ex["labels"] if r != -100])
    assert resposta.startswith('{"categoria"') and resposta.endswith(tok.eos_token)


def test_montar_lote_preenche_com_pad_e_ignora_o_pad_na_perda(tok):
    a, b = (tokenizar_exemplo(tok, c) for c in carregar("treino")[:2])
    lote = montar_lote([a, b], tok.pad_token_id, "cpu")
    assert lote["input_ids"].shape == lote["labels"].shape == lote["attention_mask"].shape
    curto = a if len(a["input_ids"]) < len(b["input_ids"]) else b
    i = 0 if curto is a else 1
    n = len(curto["input_ids"])
    assert int(lote["attention_mask"][i].sum()) == n and (lote["labels"][i][n:] == -100).all()


def test_treino_de_fumaca_so_treina_o_adaptador_e_a_perda_cai(tok, tmp_path, monkeypatch):
    monkeypatch.setenv("DISPOSITIVO", "cpu")
    base = modelo_minusculo(tok)
    pesos_antes = {n: p.detach().clone() for n, p in base.named_parameters()}
    h = treinar(str(tmp_path / "ad"), semente=1, epocas=3, tamanho_do_lote=8, micro_lote=2, taxa=5e-3,
                modelo=base, tok=tok, treino=carregar("treino")[:32], validacao=carregar("validacao")[:8])
    assert h["perda_validacao_final"] < h["perda_validacao_inicial"]
    assert 0 < h["parametros_treinaveis"] < sum(p.numel() for p in base.parameters()) / 2
    assert (tmp_path / "ad" / "adapter_config.json").exists()
    assert list((tmp_path / "ad").glob("*.safetensors"))
    for nome, p in base.named_parameters():
        original = nome.replace(".base_layer", "")
        if "lora" not in nome:
            assert torch.equal(pesos_antes[original], p.detach()), nome
    assert json.loads((tmp_path / "ad" / "historico.json").read_text())["passos"] == h["passos"]


def test_acumulacao_de_gradiente_e_igual_ao_lote_inteiro(tok):
    from peft import LoraConfig, get_peft_model
    from src.treinar import CONFIG_LORA, acumular_gradientes
    grupo = [tokenizar_exemplo(tok, c) for c in carregar("treino")[:8]]
    base = modelo_minusculo(tok)
    modelo = get_peft_model(base, LoraConfig(**CONFIG_LORA | {"lora_dropout": 0.0}))
    for p in modelo.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.05)
    modelo.eval()
    gradientes, perdas = [], []
    for micro in (8, 2):
        modelo.zero_grad()
        perdas.append(acumular_gradientes(modelo, grupo, micro, tok.pad_token_id, "cpu"))
        gradientes.append(torch.cat([p.grad.flatten() for p in modelo.parameters() if p.grad is not None]))
    assert perdas[0] == pytest.approx(perdas[1], rel=1e-3)
    erro_relativo = float((gradientes[0] - gradientes[1]).norm() / gradientes[0].norm())
    assert erro_relativo < 0.05, erro_relativo
    assert float(gradientes[0].norm()) > 0
