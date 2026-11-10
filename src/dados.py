"""Gerador de chamados de suporte de TI (sinteticos e fictícios) com rotulos conhecidos.

Cada chamado tem tres rotulos: categoria, prioridade e setor. O texto e montado a partir de moldes.
O conjunto de TREINO usa um grupo de moldes, aberturas, problemas e sinais de urgencia; o conjunto
`teste_novo` usa grupos DIFERENTES de tudo isso, para medir se o modelo entendeu a tarefa ou so decorou
as frases. O sorteio e deterministico (semente fixa)."""
from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIAS = ["acesso", "rede", "hardware", "software", "email", "impressora"]
PRIORIDADES = ["baixa", "media", "alta", "critica"]
SETORES = ["financeiro", "rh", "comercial", "ti", "juridico", "logistica"]
NOME_DO_SETOR = {"financeiro": "financeiro", "rh": "RH", "comercial": "comercial", "ti": "TI",
                 "juridico": "juridico", "logistica": "logistica"}

PROBLEMAS = {
    "treino": {
        "acesso": ["nao consigo entrar no sistema", "minha senha expirou e nao consigo trocar",
                   "o login diz usuario bloqueado", "perdi o acesso a pasta compartilhada"],
        "rede": ["a internet esta caindo toda hora", "o wifi do andar nao conecta",
                 "a vpn nao fecha conexao", "a rede esta muito lenta"],
        "hardware": ["o notebook nao liga", "o monitor ficou preto",
                     "o teclado parou de responder", "o mouse esta falhando"],
        "software": ["o programa de planilhas fecha sozinho", "o sistema de notas fiscais da erro ao salvar",
                     "o antivirus pede atualizacao e trava", "o aplicativo de videoconferencia nao abre"],
        "email": ["nao estou recebendo emails", "minha caixa de entrada esta cheia",
                  "os emails que envio voltam", "o calendario nao sincroniza"],
        "impressora": ["a impressora esta sem comunicacao", "o papel esta enroscando",
                       "a impressao sai com listras", "nao encontro a impressora do andar"],
    },
    "novo": {
        "acesso": ["apareceu acesso negado quando abro a planilha da equipe",
                   "o aplicativo de autenticacao nao gera o codigo"],
        "rede": ["o ponto de acesso da sala de reuniao sumiu da lista",
                 "os videos travam so no meu setor"],
        "hardware": ["a tela do computador pisca e fica listrada",
                     "o carregador do notebook esquenta demais e desliga"],
        "software": ["o editor de documentos perdeu meu arquivo ao salvar",
                     "o aplicativo de relatorios mostra tela branca"],
        "email": ["as respostas automaticas pararam de funcionar",
                  "sumiram convites de reuniao da agenda"],
        "impressora": ["o scanner da multifuncional nao envia para o email",
                       "a impressora imprime paginas em branco"],
    },
}

URGENCIA = {
    "treino": {
        "critica": ["parou tudo aqui", "ninguem consegue trabalhar", "o sistema de vendas esta fora do ar",
                    "tem cliente importante esperando agora"],
        "alta": ["preciso resolver ate hoje", "tenho uma entrega hoje", "isso atrapalha muito meu trabalho"],
        "media": ["preciso resolver esta semana", "atrapalha um pouco", "da para trabalhar mas e chato"],
        "baixa": ["sem pressa", "quando der", "nao e urgente", "so estou avisando"],
    },
    "novo": {
        "critica": ["a operacao inteira esta parada", "estamos perdendo pedidos por causa disso"],
        "alta": ["o prazo da proposta vence hoje", "preciso disso funcionando antes da reuniao de hoje"],
        "media": ["pode ser resolvido ate sexta", "nao e bloqueante mas incomoda"],
        "baixa": ["nao tem urgencia nenhuma", "se sobrar tempo"],
    },
}

ABERTURAS = {
    "treino": ["Bom dia,", "Oi,", "Pessoal,", "Ola time de suporte,"],
    "novo": ["Boa tarde, equipe.", "Ei,", "Atencao suporte:"],
}

SETOR_MOLDES = {
    "treino": ["aqui no {s}", "sou do {s}", "no setor {s}"],
    "novo": ["pela equipe de {s}", "falando em nome do {s}"],
}

MOLDES = {
    "treino": [
        "{abertura} {setor}, {problema}. {urgencia}.",
        "{abertura} {problema}. {urgencia}. Falo {setor}.",
        "{abertura} {setor} {problema}, {urgencia}.",
        "{abertura} abrindo chamado: {problema}. {setor}. {urgencia}.",
        "{abertura} {urgencia}. {setor}, {problema}.",
    ],
    "novo": [
        "{abertura} {problema}, {setor}. {urgencia}.",
        "{abertura} {urgencia}: {problema}. Escrevo {setor}.",
        "{abertura} {setor}: {problema} e {urgencia}.",
        "{abertura} sobre o chamado, {problema}. {urgencia}. {setor}.",
    ],
}

TAMANHOS = {"treino": 900, "validacao": 100, "teste_igual": 300, "teste_novo": 300}
SEMENTES = {"treino": 1, "validacao": 2, "teste_igual": 3, "teste_novo": 4}
GRUPO_DE_TEXTO = {"treino": "treino", "validacao": "treino", "teste_igual": "treino", "teste_novo": "novo"}


def gerar_chamado(rng: random.Random, grupo: str) -> dict:
    categoria = rng.choice(CATEGORIAS)
    prioridade = rng.choice(PRIORIDADES)
    setor = rng.choice(SETORES)
    texto = rng.choice(MOLDES[grupo]).format(
        abertura=rng.choice(ABERTURAS[grupo]),
        setor=rng.choice(SETOR_MOLDES[grupo]).format(s=NOME_DO_SETOR[setor]),
        problema=rng.choice(PROBLEMAS[grupo][categoria]),
        urgencia=rng.choice(URGENCIA[grupo][prioridade]),
    )
    return {"texto": texto, "categoria": categoria, "prioridade": prioridade, "setor": setor}


def gerar_conjunto(nome: str) -> list[dict]:
    rng = random.Random(SEMENTES[nome])
    grupo = GRUPO_DE_TEXTO[nome]
    return [{"id": f"{nome}-{i:04d}", **gerar_chamado(rng, grupo)} for i in range(TAMANHOS[nome])]


def gravar_conjuntos(pasta: str | Path = "data") -> None:
    pasta = Path(pasta)
    pasta.mkdir(exist_ok=True)
    for nome in TAMANHOS:
        linhas = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in gerar_conjunto(nome))
        (pasta / f"{nome}.jsonl").write_text(linhas)


def carregar(nome: str, pasta: str | Path = "data") -> list[dict]:
    return [json.loads(l) for l in (Path(pasta) / f"{nome}.jsonl").read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    gravar_conjuntos()
