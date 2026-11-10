# lora-finetune-nicho

> **Pilar do portfólio: Sábado** · parte de uma série de 7 projetos de Engenharia de Software de IA (1h/dia).

## O problema
Triagem de chamados de suporte de TI: ler o texto livre e devolver um JSON com **categoria**, **prioridade** e **setor**. Um modelo pequeno (Qwen2.5-0.5B-Instruct) é adaptado com LoRA e comparado com 4 baselines, em dois testes: um com as mesmas frases do treino e outro com frases, aberturas e vocabulário que o treino nunca viu.

**Os dados são sintéticos** (gerados por moldes, com semente fixa, em `src/dados.py`). Isso permite rótulo exato e teste fora da distribuição, mas também significa que **os números abaixo não dizem nada sobre chamados reais**.

## Stack
Python 3.13 · PyTorch (bf16, RTX 3060 de 6 GB) · Transformers · PEFT · laço de treino próprio (sem Trainer). Tudo local, custo zero de API.

## Resultados medidos
Acerto exato = os 3 campos certos. 300 chamados por teste. LoRA: 3 sementes de treino.

| Sistema | JSON válido | Rótulo dentro do esquema | Acerto exato (igual ao treino) | Acerto exato (frases novas) |
|---|---|---|---|---|
| Classe majoritária | 1,00 | 1,00 | 0,007 | 0,010 |
| Regras de palavras-chave | 1,00 | 1,00 | **1,000** | 0,117 |
| Qwen base, zero-shot | 1,00 | 0,74 / 0,82 | 0,040 | 0,023 |
| Qwen base, few-shot (8 exemplos) | 1,00 | 0,90 / 0,95 | 0,150 | 0,120 |
| **Qwen + LoRA (média de 3 sementes)** | 1,00 | 1,00 / 0,95 | **1,000** | **0,546** (0,497 a 0,583) |

Acerto por campo nas frases novas (média das 3 sementes): categoria 0,66, prioridade 0,81, setor 1,00.

**Diferença pareada no acerto exato (frases novas), semente 1, IC 95% por bootstrap:** LoRA menos few-shot +0,437 [0,367, 0,503]; LoRA menos regras +0,440 [0,373, 0,507]. O LoRA ganha de 152 a 150 chamados e perde em 21 a 18.

**Custo (RTX 3060, 6 GB):** cada treino leva ~110 s (226 passos, 2 épocas), usa 2,4 GB de GPU no pico, treina 8,8 milhões de parâmetros (1,75% do modelo) e o adaptador pesa 33,6 MB. Inferência: ~80 ms por chamado com o LoRA (sem fundir os pesos), ~45 ms zero-shot, ~100 ms few-shot (prompt maior).

### Leitura honesta dos números
- **No teste igual ao treino as regras também dão 100%.** A tarefa sintética é resolvível por tabela de palavras. Esse teste só prova que o LoRA aprendeu os moldes; não prova nada sobre generalização.
- **O resultado que importa é o das frases novas, e ele é fraco: 55% de acerto exato.** O LoRA é bem melhor que as regras e que o few-shot (que ficam em 12%), mas **erra quase metade** quando o vocabulário muda.
- **Onde erra (`python -m evals.analise_erros`):** nenhuma das frases novas de rede acerta (*"o ponto de acesso da sala de reuniao sumiu da lista"* e *"os videos travam so no meu setor"*: 0%), vira `acesso`; o e-mail *"as respostas automaticas pararam de funcionar"* acerta 13% (vira `software`); prioridade media *"pode ser resolvido ate sexta"* acerta 33%. O modelo se apoia em palavras do treino em vez de entender a frase.
- **O modelo inventa rótulo:** 5% das respostas das frases novas têm JSON válido mas rótulo fora do esquema; o caso típico é `video` como categoria, copiado de *"os videos travam"* (44 casos, 3 sementes somadas). "JSON válido" sozinho engana: 100% válido, 95% dentro do esquema.
- **A variação entre sementes é grande** (0,497 a 0,583 de acerto exato), então uma semente só poderia mostrar 0,50 ou 0,58. Por isso o README usa a média e o intervalo.
- **A perda de validação cai a 0,0001:** o modelo decorou os moldes. Mais épocas não ajudariam; o que falta é diversidade nos dados.
- **O few-shot da primeira rodada usava 6 exemplos em vez de 8** (bug na escolha dos exemplos, achado por teste); os números acima são da rodada corrigida.

## Limites
Dados sintéticos de 12 problemas de categoria para treino e 12 para teste; um único modelo base (0,5B); sem busca de hiperparâmetros (r=16, 2 épocas, taxa 2e-4, valores comuns, sem ajuste); acurácia medida só em acerto exato e por campo, sem F1 por classe; adaptador não fundido nos pesos (a latência de inferência incluiria menos overhead se fundido).

## Validação do próprio código
**26 testes** (CPU, ~20 s): geração determinística, ausência de vazamento de vocabulário entre treino e `teste_novo`, balanceamento de classes, leitura de JSON, métricas, e treino de fumaça com um Qwen minúsculo (a perda só conta os tokens da resposta, só o adaptador muda, a acumulação de gradiente bate com o lote inteiro).
**Mutação, 8 mutantes, todos pegos:** perda sem máscara do pedido, rótulo inventado aceito, JSON inválido não contado como erro, few-shot sem completar exemplos, teste novo reaproveitando vocabulário do treino, acumulação sem dividir pela fração, bootstrap com sinal trocado, leitura sem normalizar caixa. Um mutante **sobreviveu na primeira versão** da suíte (acumulação sem escala): o AdamW é quase invariante à escala do gradiente, então o teste que comparava o passo não distinguia; troquei por comparar os gradientes diretamente. Esse mesmo teste mostrou que a acumulação difere do lote inteiro em ~1%, porque a média da perda é por token em cada micro-lote e as respostas têm tamanhos diferentes; é aproximação conhecida, não bug.

## Como rodar
```bash
make setup     # venv + torch, transformers, peft (baixa ~5 GB e o Qwen 0,5B do Hugging Face)
make all       # dados, 3 treinos (~2 min cada na RTX 3060), eval, análise de erros, testes
make teste     # só os testes, em CPU, sem baixar o modelo grande (baixa só o tokenizador)
```
No NixOS o `Makefile` acha sozinho a `libstdc++` no store para o PyTorch carregar; em Linux comum não faz nada. Os adaptadores treinados ficam em `adaptadores/` (101 MB, fora do git); `resultados/` traz métricas e as previsões de cada sistema.

## Estrutura
- `src/dados.py` gerador sintético · `src/prompt.py` prompt e leitura do JSON · `src/treinar.py` laço LoRA · `src/modelo.py` carga e geração · `src/regras.py` baselines sem modelo · `src/metricas.py` métricas e bootstrap
- `evals/run_eval.py` os 5 sistemas nos 2 testes · `evals/analise_erros.py` erros por frase
- `data/` os 4 conjuntos (treino 900, validação 100, teste igual 300, teste novo 300) · `resultados/` métricas e previsões

## Status
- [x] Esqueleto / setup roda em < 15 min
- [x] Núcleo funcional (treino LoRA + inferência)
- [x] Avaliação / métricas medidas (5 sistemas, 2 testes, 3 sementes, IC por bootstrap)
- [x] README com números, latência e custo
- [ ] Dados com mais diversidade (ou chamados reais anonimizados) para o teste novo subir de 55%
- [ ] Ablação de r, épocas e taxa; modelo base maior (1,5B) para ver quanto do erro é capacidade
- [ ] Deploy / demo reproduzível

---
Autor: Gappy · Série montada com apoio do Hermes
